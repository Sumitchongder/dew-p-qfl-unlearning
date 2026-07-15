#!/bin/bash
#SBATCH --job-name=dewp-main
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/dewp-main_%j.out
#SBATCH --error=logs/slurm/dewp-main_%j.err

set -euo pipefail
module purge
module load python/3.10 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

source venv/bin/activate 2>/dev/null || echo "No venv found; using system python3."

# Publication-scale run: more seeds and more optimization budget than the
# quick-look default in scripts/run_main_experiment.py. Adjust to your
# allocation.
python3 scripts/run_main_experiment.py \
    --seeds 0 1 2 3 4 5 6 7 \
    --n-qubits 4 --n-layers 3 \
    --n-clients 5 --samples-per-client 200 \
    --non-iid-strength 0.9 --noise 0.2 \
    --n-rounds 8 --local-maxiter 20 \
    --forget-client 0 --prune-fraction 0.2
