# Running on an HPC / SLURM Cluster

This directory contains a four-stage SLURM job chain that reproduces every
result in `results/` at a larger, publication-scale compute budget than the
quick-look defaults used in `scripts/run_full_pipeline.sh`.

## Job chain

| Order | Script | Depends on | Typical wall time* |
|---|---|---|---|
| 1 | `01_run_main_experiment.slurm` |   | 45-90 min |
| 2 | `02_run_sweeps.slurm` | job 1 | 30-60 min |
| 3 | `03_run_reconstruction.slurm` | job 1 | 30-60 min |
| 4 | `04_generate_deliverables.slurm` | jobs 2 and 3 | < 5 min |

\* Single CPU core, 4 qubits, 8 seeds, 8 FedAvg rounds x 20 local
L-BFGS-B iterations. Everything here is a classical statevector
simulation (no external QPU calls), so wall time scales with
`n_seeds x n_rounds x local_maxiter x samples_per_client` and with
`2^n_qubits` for the simulated Hilbert space   see
`results/tables/table06_complexity.csv` / `scalability.csv` for measured
scaling.

## Submitting the chain

```bash
bash hpc/slurm/submit_all.sh
```

This submits all four jobs with SLURM `--dependency=afterok` chaining, so
job 2 and 3 will not start until job 1 finishes successfully, and job 4
waits on both. Monitor with `squeue -u $USER`; logs land in `logs/slurm/`.

## Submitting stages individually

Each `.slurm` file can also be submitted on its own with `sbatch`, e.g. to
re-run only the sweep stage after changing a hyperparameter:

```bash
sbatch hpc/slurm/02_run_sweeps.slurm
```

## Adjusting compute budget

Edit the `python3 scripts/run_main_experiment.py ...` invocation inside
`01_run_main_experiment.slurm` directly   every experimental hyperparameter
(qubit count, layers, clients, samples, FedAvg rounds, local optimizer
iterations, pruning threshold, random seeds) is a CLI flag. No source
changes are required to scale the experiment up or down.

## Environment

These scripts assume a `venv/` virtual environment has already been created
at the repository root (see the top-level `README.md` "Installation"
section) and that your cluster's module system exposes a `python/3.10` (or
compatible) module. Both the `module load` line and the `venv` activation
are wrapped in `|| true` / `|| echo ...` fallbacks so the scripts degrade
gracefully to the system Python if your site is configured differently  
adjust these two lines for your cluster's conventions before submitting.
