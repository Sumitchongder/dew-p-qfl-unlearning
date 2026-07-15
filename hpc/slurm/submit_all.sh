#!/bin/bash
# Submits the full four-stage pipeline as a dependency chain:
#   01_run_main_experiment -> 02_run_sweeps -> 03_run_reconstruction -> 04_generate_deliverables
#
# Usage: bash hpc/slurm/submit_all.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs/slurm

JID1=$(sbatch --parsable hpc/slurm/01_run_main_experiment.slurm)
echo "Submitted 01_run_main_experiment: job $JID1"

JID2=$(sbatch --parsable --dependency=afterok:$JID1 hpc/slurm/02_run_sweeps.slurm)
echo "Submitted 02_run_sweeps: job $JID2 (after $JID1)"

JID3=$(sbatch --parsable --dependency=afterok:$JID1 hpc/slurm/03_run_reconstruction.slurm)
echo "Submitted 03_run_reconstruction: job $JID3 (after $JID1)"

JID4=$(sbatch --parsable --dependency=afterok:$JID2:$JID3 hpc/slurm/04_generate_deliverables.slurm)
echo "Submitted 04_generate_deliverables: job $JID4 (after $JID2 and $JID3)"

echo
echo "Monitor with: squeue -u \$USER"
