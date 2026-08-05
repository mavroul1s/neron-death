"""Task loop, checkpointing, logging.

Computes no metrics of its own: every number written to disk comes from
``src.probes`` (CLAUDE.md §4). If you find yourself about to write ``.mean()``
in this file, put it in probes.py instead.

Timeline of one task, in order, because the order is load-bearing:

    1. reset the gradient window (so "last 100 steps" cannot span a boundary)
    2. stream one pass over the training set; recycling events fire on the
       global step counter, mid-task, every F steps
    3. probe -- current-task batch and reference batch -- and write the
       per-neuron rows for this boundary
    4. shrink-and-perturb, if enabled (after the measurement, so the row
       describes the network the task produced, not the perturbed one)
    5. every `checkpoint.every_tasks`: flush log shards, THEN checkpoint

Step 5's order matters on resume: a shard ahead of the checkpoint is dropped and
rewritten, whereas a checkpoint ahead of the shards would silently lose rows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from . import probes
from .config import (
    assert_config_unchanged,
    config_hash,
    load_config,
    resolve_config,
    resolve_device,
    save_config,
    set_determinism,
)
from .data import build_dataset
from .interventions import (
    Recycler,
    RecyclerConfig,
    ShrinkPerturbConfig,
    l2_penalty,
    shrink_and_perturb,
)
from .logs import NEURON_SCHEMA, ShardedParquetLog, cleanup_shard_dir
from .models import build_model
from .probes import GradientTracker, ProbeConfig

#: Salts so each independent stream gets its own generator.
_SALT_MODEL_INIT = 0x1111
_SALT_SP = 0x2222

INIT_TASK_IDX = -1  # the probe taken before any training


def build_optimizer(cfg: dict, model) -> torch.optim.Optimizer:
    o = cfg["optim"]
    name = str(o["name"]).lower()
    wd = float(o.get("weight_decay", 0.0))
    if name != "adamw" and wd != 0.0:
        # Two routes to the same regulariser is how a run silently gets twice
        # the L2 it was supposed to have.
        raise ValueError(
            "optim.weight_decay is only meaningful for AdamW (decoupled decay). "
            "For coupled L2 set l2.lambda, which is added explicitly to the loss."
        )
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(o["lr"]), momentum=float(o["momentum"]))
    if name == "adam":
        return torch.optim.Adam(
            params, lr=float(o["lr"]), betas=tuple(o["betas"]), eps=float(o["eps"])
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(o["lr"]),
            betas=tuple(o["betas"]),
            eps=float(o["eps"]),
            weight_decay=wd,
        )
    raise ValueError(f"unknown optimizer {name!r}; known: sgd, adam, adamw")


class Trainer:
    """One config, one run directory, resumable by run_id alone."""

    def __init__(self, cfg: dict, runs_root="runs", device: Optional[str] = None):
        self.cfg = resolve_config(cfg)
        self.run_id: str = self.cfg["run_id"]
        self.hash: str = config_hash(self.cfg)
        self.run_dir = Path(runs_root) / self.run_id
        self.device_spec = device or self.cfg["device"]

        self.global_step = 0
        self.wall_time_s = 0.0
        self._t_start = 0.0

    # -- setup ----------------------------------------------------------------

    def setup(self) -> None:
        assert_config_unchanged(self.cfg, self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        det = self.cfg["determinism"]
        self.determinism_report = set_determinism(
            int(self.cfg["seed"]), bool(det["strict"]), bool(det["warn_only"])
        )
        self.device = resolve_device(self.device_spec)

        # Model init uses a dedicated CPU generator so the initial weights are
        # identical whether the run starts on a T4 or on a laptop.
        init_gen = torch.Generator()
        init_gen.manual_seed(int(self.cfg["seed"]) ^ _SALT_MODEL_INIT)
        self.model = build_model(self.cfg["model"], generator=init_gen).to(self.device)

        self.optimizer = build_optimizer(self.cfg, self.model)
        self.dataset = build_dataset(self.cfg["data"], self.device)
        self.probe_cfg = ProbeConfig.from_dict(self.cfg["probe"])

        self.recycler = Recycler(
            RecyclerConfig.from_dict(self.cfg["recycling"]),
            seed=int(self.cfg["seed"]),
            probe_cfg=self.probe_cfg,
            run_id=self.run_id,
        )
        self.sp = ShrinkPerturbConfig.from_dict(self.cfg["shrink_perturb"])
        self._sp_gen = torch.Generator()
        self._sp_gen.manual_seed(int(self.cfg["seed"]) ^ _SALT_SP)

        self.grad_tracker = GradientTracker(
            list(self.model.linears),
            window=self.probe_cfg.grad_window,
            device=self.device,
        )

        self.l2_lambda = float(self.cfg["l2"]["lambda"])
        self.l2_include_bias = bool(self.cfg["l2"]["include_bias"])

        # The reference probe batch is built once and never rebuilt.
        self.ref_x, self.ref_y = self.dataset.reference_batch()

        self.logs = {
            "tasks": ShardedParquetLog(self.run_dir, "tasks"),
            "metrics": ShardedParquetLog(
                self.run_dir,
                "metrics",
                # saturated_frac is None for unbounded activations; without an
                # explicit type the first all-None shard would be null-typed.
                type_overrides={
                    "saturated_frac": pa.float64(),
                    "saturated_frac_ref": pa.float64(),
                },
            ),
            "neurons": ShardedParquetLog(self.run_dir, "neurons", schema=NEURON_SCHEMA),
            "recycling": ShardedParquetLog(self.run_dir, "recycling"),
        }

        save_config(self.cfg, self.run_dir / "config.json")
        with open(self.run_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_hash": self.hash,
                    "device": str(self.device),
                    "determinism": self.determinism_report,
                    "n_parameters": self.model.n_parameters(),
                    "steps_per_task": self.dataset.steps_per_task,
                    "model": self.model.describe(),
                },
                f,
                indent=2,
                default=str,
            )
            f.write("\n")

    # -- the loop -------------------------------------------------------------

    def run(self, resume: bool = True) -> Path:
        self.setup()
        n_tasks = int(self.cfg["data"]["n_tasks"])
        ckpt_every = int(self.cfg["checkpoint"]["every_tasks"])

        start_task = 0
        if resume:
            resumed = self._resume()
            if resumed is not None:
                start_task = resumed + 1
                print(f"[{self.run_id}] resuming at task {start_task}", flush=True)

        if start_task == 0:
            # Pristine network, before any gradient has been taken. Uses task
            # 0's permutation for the "current" batch.
            self._boundary(INIT_TASK_IDX, "init", 0, None)

        self._t_start = time.perf_counter()
        for task_idx in range(start_task, n_tasks):
            self._run_task(task_idx)
            is_last = task_idx == n_tasks - 1
            if (task_idx + 1) % ckpt_every == 0 or is_last:
                for log in self.logs.values():
                    log.flush()  # shards first...
                self._save_checkpoint(task_idx)  # ...then the checkpoint

        return self._finalize(n_tasks)

    def _run_task(self, task_idx: int) -> None:
        t0 = time.perf_counter()
        ds, model, opt = self.dataset, self.model, self.optimizer

        self.grad_tracker.reset()
        was_recycled = [np.zeros(h, dtype=bool) for h in model.hidden_dims]
        probe_x, _ = ds.probe_batch(task_idx)

        model.train()
        # Accumulate on-device: a .item() per step would sync the GPU 469 times
        # per task for numbers we only need at the boundary.
        correct = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        seen = 0
        n_recycled = 0

        for xb, yb in ds.task_batches(task_idx):
            logits = model(xb)
            with torch.no_grad():
                # Online accuracy: the prediction the network makes on each
                # example *before* being trained on it (Dohare et al.'s online
                # measure). Taken from the training-mode forward pass, so for a
                # dropout arm it is measured under dropout -- as in Dohare et
                # al. The eval-mode probe accuracy below is the clean companion.
                correct += (logits.argmax(dim=1) == yb).sum()
            loss = F.cross_entropy(logits, yb)
            total = loss
            if self.l2_lambda > 0.0:
                total = total + self.l2_lambda * l2_penalty(model, self.l2_include_bias)

            opt.zero_grad(set_to_none=True)
            total.backward()
            self.grad_tracker.update()  # after backward, before step
            opt.step()

            self.global_step += 1
            bs = int(yb.shape[0])
            loss_sum += loss.detach().to(torch.float64) * bs
            seen += bs

            if self.recycler.due(self.global_step):
                score_x, _ = ds.score_batch(
                    task_idx,
                    self.recycler.event_idx,
                    n=self.recycler.cfg.score_batch_size,
                )
                event = self.recycler.run_event(
                    model,
                    opt,
                    score_x=score_x,
                    probe_x=probe_x,
                    step=self.global_step,
                    task_idx=task_idx,
                    ref_x=self.ref_x,
                )
                self.logs["recycling"].add_rows(event.rows)
                for layer_idx, idx in event.recycled.items():
                    if idx.size:
                        was_recycled[layer_idx][idx] = True
                n_recycled += event.total_recycled

        online_acc = float(correct.item()) / seen
        mean_loss = float(loss_sum.item()) / seen

        self._boundary(
            task_idx,
            "task_end",
            n_recycled,
            was_recycled,
            online_accuracy=online_acc,
            mean_loss=mean_loss,
            task_wall_s=time.perf_counter() - t0,
        )

        # Applied after the boundary measurement so the logged row describes the
        # network this task produced.
        if self.sp.enabled and (task_idx + 1) % self.sp.every_tasks == 0:
            shrink_and_perturb(model, self.sp.shrink, self.sp.perturb, self._sp_gen)

    # -- measurement ----------------------------------------------------------

    @torch.no_grad()
    def _boundary(
        self,
        task_idx: int,
        probe_point: str,
        n_recycled: int,
        was_recycled: Optional[List[np.ndarray]],
        online_accuracy: float = float("nan"),
        mean_loss: float = float("nan"),
        task_wall_s: float = float("nan"),
    ) -> None:
        """Probe both batches and write one boundary's worth of rows."""
        model, ds = self.model, self.dataset
        was_training = model.training
        model.eval()
        try:
            cur_task = max(task_idx, 0)  # the init probe uses task 0's permutation
            probe_x, probe_y = ds.probe_batch(cur_task)
            cur, cur_logits = probes.probe_model_and_logits(model, probe_x, self.probe_cfg)
            ref, ref_logits = probes.probe_model_and_logits(
                model, self.ref_x, self.probe_cfg
            )

            metric_rows = []
            for li in range(model.n_hidden):
                lin_in = model.incoming_linear(li)
                lin_out = model.outgoing_linear(li)
                w_in_norm, w_out_norm = probes.neuron_weight_norms(
                    lin_in.weight, lin_out.weight
                )
                # Per-layer weight stats cover the layer's own (W, b). Norm-layer
                # affine parameters are excluded here and counted in the
                # model-level total on the tasks row.
                wl2, wmean = probes.weight_stats([lin_in.weight, lin_in.bias])
                metric_rows.append(
                    probes.layer_metric_row(
                        run_id=self.run_id,
                        task_idx=task_idx,
                        probe_point=probe_point,
                        layer_idx=li,
                        cur=cur[li],
                        ref=ref[li],
                        weight_l2=wl2,
                        weight_mean_abs=wmean,
                        grad_norm_layer=self.grad_tracker.layer_mean(li),
                        grad_window_steps=self.grad_tracker.n_steps_in_window,
                    )
                )
                self.logs["neurons"].add_columns(
                    probes.neuron_rows(
                        run_id=self.run_id,
                        task_idx=task_idx,
                        probe_point=probe_point,
                        layer_idx=li,
                        cur=cur[li],
                        ref=ref[li],
                        w_in_norm=w_in_norm,
                        w_out_norm=w_out_norm,
                        bias=lin_in.bias.detach().to(torch.float64).cpu().numpy(),
                        grad_norm_neuron=self.grad_tracker.neuron_mean(li),
                        was_recycled=(
                            was_recycled[li]
                            if was_recycled is not None
                            else np.zeros(cur[li].n_neurons, dtype=bool)
                        ),
                    )
                )
            self.logs["metrics"].add_rows(metric_rows)

            model_l2, model_mean_abs = probes.weight_stats(
                [p for p in model.parameters()]
            )
            self.logs["tasks"].add_rows(
                [
                    {
                        "run_id": self.run_id,
                        "task_idx": task_idx,
                        "probe_point": probe_point,
                        "global_step": self.global_step,
                        "online_accuracy": online_accuracy,
                        "mean_loss": mean_loss,
                        "probe_accuracy": probes.accuracy(cur_logits, probe_y),
                        "probe_accuracy_ref": probes.accuracy(ref_logits, self.ref_y),
                        "model_weight_l2": model_l2,
                        "model_weight_mean_abs": model_mean_abs,
                        "grad_norm_output_layer": self.grad_tracker.layer_mean(
                            model.n_hidden
                        ),
                        "n_recycled": int(n_recycled),
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "task_wall_s": float(task_wall_s),
                    }
                ]
            )
        finally:
            if was_training:
                model.train()

    # -- checkpointing --------------------------------------------------------

    def _checkpoint_path(self, task_idx: int) -> Path:
        return self.run_dir / "checkpoints" / f"ck_{task_idx:05d}.pt"

    def _save_checkpoint(self, task_idx: int) -> Path:
        self.wall_time_s += time.perf_counter() - self._t_start
        self._t_start = time.perf_counter()
        state = {
            "task_idx": task_idx,  # last COMPLETED task
            "global_step": self.global_step,
            "config_hash": self.hash,
            "wall_time_s": self.wall_time_s,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "recycler": self.recycler.state_dict(),
            "sp_gen": self._sp_gen.get_state(),
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "shard_dir": str(self.run_dir / "_shards"),
        }
        path = self._checkpoint_path(task_idx)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic: a killed save never leaves a torn file

        keep = int(self.cfg["checkpoint"]["keep_last"])
        existing = sorted((self.run_dir / "checkpoints").glob("ck_*.pt"))
        for old in existing[:-keep] if keep > 0 else []:
            old.unlink()
        return path

    def _latest_checkpoint(self) -> Optional[Path]:
        cks = sorted((self.run_dir / "checkpoints").glob("ck_*.pt"))
        return cks[-1] if cks else None

    def _resume(self) -> Optional[int]:
        path = self._latest_checkpoint()
        if path is None:
            return None
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("config_hash") != self.hash:
            raise RuntimeError(
                f"{path} was written by config hash {state.get('config_hash')} "
                f"but this run has hash {self.hash}. Refusing to resume."
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.recycler.load_state_dict(state["recycler"])
        self._sp_gen.set_state(state["sp_gen"])
        rng = state["rng"]
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        self.global_step = int(state["global_step"])
        self.wall_time_s = float(state.get("wall_time_s", 0.0))

        last_task = int(state["task_idx"])
        for log in self.logs.values():
            log.rewind_to(last_task)
        return last_task

    # -- finish ---------------------------------------------------------------

    def _finalize(self, n_tasks: int) -> Path:
        self.wall_time_s += time.perf_counter() - self._t_start
        paths = {name: log.finalize() for name, log in self.logs.items()}
        cleanup_shard_dir(self.run_dir)
        self._verify_per_neuron_log(n_tasks, paths.get("neurons"))

        summary = {
            "run_id": self.run_id,
            "config_hash": self.hash,
            "n_tasks": n_tasks,
            "global_step": self.global_step,
            "wall_time_s": self.wall_time_s,
            "gpu_hours": self.wall_time_s / 3600.0,
            "device": str(self.device),
            "status": "complete",
            "outputs": {k: (str(v) if v else None) for k, v in paths.items()},
        }
        tasks_path = paths.get("tasks")
        if tasks_path is not None:
            t = pq.read_table(tasks_path).to_pydict()
            accs = [
                a
                for a, p in zip(t["online_accuracy"], t["probe_point"])
                if p == "task_end"
            ]
            summary["final_10_task_online_accuracy"] = (
                float(np.mean(accs[-10:])) if accs else None
            )
            summary["first_10_task_online_accuracy"] = (
                float(np.mean(accs[:10])) if accs else None
            )
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        return self.run_dir

    def _verify_per_neuron_log(self, n_tasks: int, path: Optional[Path]) -> None:
        """A run without the per-neuron log is a wasted run (CLAUDE.md §5.4).

        The dataset cannot be reconstructed after the fact, so this is a hard
        failure, never a warning.
        """
        if path is None or not Path(path).exists():
            raise RuntimeError(
                f"[{self.run_id}] per-neuron log was not written. The C4 dataset "
                "is not recoverable retrospectively; treat this run as lost."
            )
        expected = (n_tasks + 1) * sum(self.model.hidden_dims)  # +1 for the init probe
        actual = pq.read_metadata(path).num_rows
        if actual != expected:
            raise RuntimeError(
                f"[{self.run_id}] neurons.parquet has {actual} rows, expected "
                f"{expected} = ({n_tasks} tasks + 1 init) x "
                f"{sum(self.model.hidden_dims)} neurons. Boundary rows are "
                "missing or duplicated; do not analyse this run."
            )


def run_config(config_path, runs_root="runs", device=None, resume=True) -> Path:
    """Entry point. This is the single function a Kaggle notebook calls."""
    cfg = load_config(config_path)
    return Trainer(cfg, runs_root=runs_root, device=device).run(resume=resume)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one neuron-death experiment config.")
    ap.add_argument("--config", required=True, help="path to a run config JSON")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--device", default=None, help="override config device")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    run_dir = run_config(
        args.config,
        runs_root=args.runs_root,
        device=args.device,
        resume=not args.no_resume,
    )
    print(f"{run_dir}  ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
