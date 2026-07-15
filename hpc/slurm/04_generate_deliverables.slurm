#!/bin/bash
#SBATCH --job-name=dewp-deliverables
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/dewp-deliverables_%j.out
#SBATCH --error=logs/slurm/dewp-deliverables_%j.err

set -euo pipefail
module purge
module load python/3.10 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
source venv/bin/activate 2>/dev/null || echo "No venv found; using system python3."

python3 scripts/generate_deliverables.py
