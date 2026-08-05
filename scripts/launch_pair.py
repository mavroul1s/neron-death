"""Run configs two at a time, one per T4.

CLAUDE.md §3: the MLPs never saturate a single T4, so two independent configs
run in parallel via ``CUDA_VISIBLE_DEVICES``. Each child sees exactly one GPU
and cannot be written to assume otherwise.

    python scripts/launch_pair.py configs/gate/*.json --budget-hours 10.5

Stops launching new work once ``--budget-hours`` has elapsed, so a queue can be
pointed at a Kaggle session without risking the unannounced 12-hour kill
landing mid-run. Anything not started is simply left for the next session;
anything interrupted resumes from its own checkpoint by run_id.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _launch(config: Path, gpu: int, runs_root: str, data_root: str = "") -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Set before the child creates its cuBLAS handle; see config.set_determinism.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(config),
        "--runs-root",
        runs_root,
        "--device",
        "cuda",  # the child sees exactly one GPU, which is always cuda:0 to it
    ]
    if data_root:
        cmd += ["--data-root", data_root]
    print(f"[gpu{gpu}] {config.name}", flush=True)
    return subprocess.Popen(cmd, cwd=ROOT, env=env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="config JSON paths (globs expanded by the shell)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument(
        "--budget-hours",
        type=float,
        default=10.5,
        help="stop launching new runs after this much wall time (session cap is 12h)",
    )
    ap.add_argument("--skip-complete", action="store_true", default=True)
    args = ap.parse_args(argv)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    queue = [Path(c) for c in args.configs]
    if args.skip_complete:
        remaining = []
        for c in queue:
            run_id = json.loads(c.read_text(encoding="utf-8"))["run_id"]
            summary = Path(args.runs_root) / run_id / "summary.json"
            if summary.exists() and json.loads(summary.read_text())["status"] == "complete":
                continue
            remaining.append(c)
        print(f"{len(queue) - len(remaining)} already complete, {len(remaining)} to run")
        queue = remaining

    t0 = time.monotonic()
    running = {}  # gpu -> (Popen, config)
    failures = []
    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            if (time.monotonic() - t0) / 3600.0 >= args.budget_hours:
                if queue:
                    print(
                        f"budget reached; {len(queue)} config(s) left for the next session",
                        flush=True,
                    )
                    queue = []
                break
            cfg = queue.pop(0)
            running[gpu] = (_launch(cfg, gpu, args.runs_root), cfg)

        time.sleep(2.0)
        for gpu, (proc, cfg) in list(running.items()):
            if proc.poll() is None:
                continue
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"[gpu{gpu}] {cfg.name}: {status}", flush=True)
            if proc.returncode != 0:
                failures.append(cfg.name)
            del running[gpu]

    elapsed = (time.monotonic() - t0) / 3600.0
    print(f"done in {elapsed:.2f}h; {len(failures)} failure(s)")
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
