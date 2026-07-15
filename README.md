# DEW-P: Dynamic Entanglement-Weighted Pruning for Quantum Federated Unlearning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A reference implementation of **entanglement-weighted parameter pruning**
for selective client forgetting in quantum federated learning (QFL),
applied to a supply-chain risk classification task. Given a federated
variational quantum classifier and a "forget this client" request, this
repository computes a per-parameter importance score that combines
**quantum Fisher information** (how sensitive the model's state is to that
parameter, conditioned on the forgotten client's data) with **circuit
entanglement structure** (how structurally load-bearing that parameter's
gate is), prunes the least important parameters, and briefly re-optimizes
the survivors, producing a model statistically indistinguishable from a
full from-scratch retrain, at a fraction of the compute cost.

This repository is the companion code for the paper *"Dynamic
Entanglement-Weighted Pruning for QFL-Based Supply Chain Risk Unlearning."*
It is built to be run, not just read: every number in `results/` is
produced by the code in `src/`, with no hand-edited figures or tables.

---

## Table of contents

- [What's in this repository](#whats-in-this-repository)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Reproducing the full result set](#reproducing-the-full-result-set)
- [Running on an HPC cluster](#running-on-an-hpc-cluster)
- [Method summary](#method-summary)
- [Results at a glance](#results-at-a-glance)
- [Extending this codebase](#extending-this-codebase)
- [Reproducibility and honesty notes](#reproducibility-and-honesty-notes)
- [Citation](#citation)
- [License](#license)

---

## What's in this repository

- **`src/qflewp/`**   the full method implementation: a data re-uploading
  variational quantum circuit simulated in NumPy, a FedAvg federated
  trainer using exact parameter-shift gradients, a parameter-shift diagonal
  Quantum Fisher Information estimator, a per-gate von Neumann entanglement
  weighting scheme, the pruning/unlearning algorithms (proposed method +
  5 baselines), and a full evaluation suite (utility, membership-inference
  based forgetting, retrain-distance).
- **`scripts/`**   thin, documented CLI entry points that call into
  `src/qflewp/`   no logic lives in the scripts themselves.
- **`hpc/slurm/`**   a four-stage SLURM job chain for running the full
  experiment suite at publication scale on a shared cluster.
- **`results/`**   every figure (`figures/`, PNG + PDF), table
  (`tables/`, CSV + a combined Markdown export), and raw run artifact
  (`json/`) referenced in the paper, already generated and version
  controlled so you can inspect them without running anything.
- **`tests/`**   fast (~15 s) correctness checks: circuit unitarity, QFIM
  client-dependence, entanglement-weight differentiation, and exact
  pruning-fraction behavior.
- **`docs/METHODOLOGY.md`**   the exact formulas and configuration schema
  used by the code, for readers extending or auditing it.

## Quickstart

```bash
git clone https://github.com/<your-username>/dew-p-qfl-unlearning.git
cd dew-p-qfl-unlearning

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

make install                    # pip install -r requirements.txt + editable install
make test                       # ~15 s, verifies the install is correct
```

Run a small end-to-end pass (a few minutes, 1 seed, reduced budget) to
confirm everything works before committing to a full run:

```bash
python3 scripts/run_main_experiment.py --seeds 0 --n-rounds 3 --local-maxiter 8
```

## Repository layout

```
dew-p-qfl-unlearning/
├── src/qflewp/                  # method implementation (see docs/METHODOLOGY.md)
│   ├── circuit.py                 # data re-uploading VQC, NumPy statevector sim
│   ├── data.py                    # synthetic non-IID supply-chain dataset generator
│   ├── federated.py                # FedAvg + exact parameter-shift gradients
│   ├── qfim.py                    # parameter-shift diagonal QFIM estimator
│   ├── entanglement.py            # per-gate von Neumann entanglement weights
│   ├── pruning.py                 # EWP + 5 baselines
│   ├── evaluate.py                # utility / forgetting / retrain-distance metrics
│   ├── pipeline.py                # main N-seed benchmark orchestration
│   ├── sweeps.py                  # pruning-threshold sweep + ablations
│   ├── reconstruct.py             # per-method predictions, confusion matrices, timing
│   └── q1_deliverables.py         # all 16 figures + 10 tables, from saved JSON/CSV only
├── scripts/                     # CLI wrappers around src/qflewp/
├── hpc/slurm/                   # SLURM batch chain for cluster runs
├── results/
│   ├── figures/                   # fig01-fig16, .png (400 DPI) + .pdf (vector)
│   ├── tables/                    # table01-table10, .csv + ALL_TABLES.md
│   └── json/                      # main_experiment_raw.json, extended_experiment.json
├── tests/                       # pytest suite
├── docs/METHODOLOGY.md          # formulas + config schema reference
├── pyproject.toml / requirements.txt / environment.yml
└── Makefile
```

## Reproducing the full result set

The whole pipeline is four sequential stages, each independently runnable
and independently cached to disk:

```bash
bash scripts/run_full_pipeline.sh
```

which is equivalent to:

```bash
python3 scripts/run_main_experiment.py --seeds 0 1 2       # trains the federated model, runs EWP + baselines
python3 scripts/run_sweeps.py --stage all                  # pruning-threshold sweep + robustness ablations
python3 scripts/run_reconstruction.py                      # confusion matrices, ROC curves, wall-clock timing
python3 scripts/generate_deliverables.py                   # all 16 figures + 10 tables from the above
```

Every stage reads/writes plain JSON or CSV under `results/`, so you can
inspect, diff, or version-control intermediate results, and re-run only
the stage you changed. All experiment hyperparameters are CLI flags on
`run_main_experiment.py`   see `docs/METHODOLOGY.md#7` for the full schema.

Default configuration runs in roughly 30-45 minutes on a single modern CPU
core. Scale seeds/rounds/samples up for tighter confidence intervals; see
[Reproducibility and honesty notes](#reproducibility-and-honesty-notes).

## Running on an HPC cluster

See [`hpc/README.md`](hpc/README.md) for a SLURM job chain covering the
same four stages, with dependency-aware submission (`hpc/slurm/submit_all.sh`)
and a publication-scale default configuration (8 seeds, 8 FedAvg rounds).

## Method summary

Given a federated variational quantum classifier trained across `N`
clients and a forget request from client `j`, this repository computes,
for every trainable circuit parameter `k`:

```
s_k = w_ent(k) * F_kk^(j)
```

where `F_kk^(j)` is the diagonal Quantum Fisher Information entry for
parameter `k`, estimated via the exact parameter-shift rule and averaged
over client `j`'s own data, and `w_ent(k)` is the von Neumann entanglement
entropy generated at the point in the circuit where parameter `k`'s gate
sits. Parameters with the lowest `s_k` are pruned (set to zero, equivalent
to replacing the gate with the identity), and the surviving parameters are
briefly re-optimized on the retained clients only. Full formulas, the
gradient/QFIM derivations, and the five baseline methods this is compared
against are documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Results at a glance

Full numbers are in `results/tables/`; figures are in `results/figures/`.
Headline comparison (3 seeds, mean ± std), from
`results/tables/table03_baseline_comparison.csv`:

| Method | Accuracy | Unlearning time |
|---|---|---|
| Random Pruning | ~0.47 | < 1 ms |
| Fisher-only Pruning | ~0.47 | < 1 ms |
| Entanglement-only Pruning | ~0.40 | < 1 ms |
| Fine-Tune Only (no pruning) | ~0.87 | ~14 s |
| Full Retrain (oracle) | ~0.79 | ~65 s |
| **DEW-P (this method)** | **~0.84** | **~4 s** |

A paired t-test (`results/tables/table08_statistical_significance.csv`)
finds no statistically significant utility gap between DEW-P and the
full-retrain oracle, while DEW-P significantly outperforms all three
single-signal pruning baselines   at roughly 16x the speed of a full
retrain.

## Extending this codebase

- **New ansatz**: implement an alternative to `circuit.py`'s `VQC` class
  exposing the same `forward_full` / `predict_proba` interface; everything
  downstream (QFIM, entanglement, pruning, evaluation) is ansatz-agnostic.
- **New dataset**: replace `data.py`'s `generate_federated_dataset` with a
  loader returning a list of `ClientDataset` objects; no other file needs
  to change.
- **New baseline**: add a function to `pruning.py` following the
  `prune_by_score(theta, scores, fraction, method_name)` pattern used by
  the existing baselines, then register it in `pipeline.run_single_seed`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the PR checklist.

## Reproducibility and honesty notes

- The default configuration uses **3 random seeds** for the main benchmark
  (compute-budget constrained for interactive reproduction). The
  statistical-significance table is reported honestly at this sample size;
  treat p-values as indicative rather than conclusive, and prefer 8-10
  seeds (`hpc/slurm/01_run_main_experiment.slurm` already does this) before
  citing significance claims in follow-up work.
- The membership-inference forgetting metric is measured on a synthetic
  dataset with a few hundred samples per client; at this scale the metric
  has real seed-to-seed variance (visible directly in
  `table05_forgetting_comparison.csv`). This is disclosed rather than
  smoothed over, in line with the ablation-honesty goals in the paper.
- Every figure and table is regenerated from `results/json/*.json` by
  `scripts/generate_deliverables.py`   nothing under `results/figures/` or
  `results/tables/` is hand-edited. If you change the pipeline, regenerate
  rather than patching outputs directly.

## Citation

If this repository or its results are useful to you, please cite it using
the metadata in [`CITATION.cff`](CITATION.cff) (also exposed via GitHub's
"Cite this repository" button), and cite the accompanying paper once
published.

## License

Released under the [MIT License](LICENSE).
