#!/bin/bash
#SBATCH --job-name=dewp-sweeps
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/dewp-sweeps_%j.out
#SBATCH --error=logs/slurm/dewp-sweeps_%j.err
#SBATCH --dependency=singleton

set -euo pipefail
module purge
module load python/3.10 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
source venv/bin/activate 2>/dev/null || echo "No venv found; using system python3."

python3 scripts/run_sweeps.py --stage all --seed 0
