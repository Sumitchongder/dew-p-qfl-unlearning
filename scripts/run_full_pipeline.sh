#!/usr/bin/env bash
# End-to-end reproduction of every result, table, and figure in this
# repository, from a clean checkout.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Expected wall-clock time on a single CPU core: ~30-45 minutes for the
# default configuration (3 seeds, 4 qubits, 6 FedAvg rounds). Scale via the
# --seeds / --n-rounds / --local-maxiter flags on run_main_experiment.py for
# a larger, publication-grade run (see hpc/slurm/ for a batch-queue version
# of this same sequence).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== [1/4] Main N-seed benchmark =="
python3 scripts/run_main_experiment.py --seeds 0 1 2

echo "== [2/4] Pruning-threshold sweep + ablations =="
python3 scripts/run_sweeps.py --stage all

echo "== [3/4] Per-method reconstruction (confusion matrices, ROC, timings) =="
python3 scripts/run_reconstruction.py

echo "== [4/4] Figures + tables =="
python3 scripts/generate_deliverables.py

echo "All done. Results are in results/{figures,tables,json}/"
