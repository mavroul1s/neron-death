"""Source package for the neuron-death continual-learning study.

Module boundaries are load-bearing (see CLAUDE.md §4):

* ``probes``        -- every metric definition. Nothing else computes a metric.
* ``models``        -- the MLP; activation and normalisation are config fields.
* ``data``          -- permuted MNIST, label-shuffled CIFAR-10.
* ``interventions`` -- ReDo, random-matched, inverse-matched, L2, shrink-and-perturb.
* ``train``         -- task loop, checkpointing, logging.
* ``analysis``      -- post-hoc only. MUST NOT be imported by training code.
"""
