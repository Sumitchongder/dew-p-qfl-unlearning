#!/usr/bin/env python3
"""
Entry point: reconstruct per-method predictions, confusion matrices, ROC
curves, and wall-clock timings from the already-trained models saved by
run_main_experiment.py (results/json/main_experiment_raw.json).

No retraining of the shared federated model is performed here (theta_full /
theta_oracle / QFIM / entanglement are loaded, not recomputed) except for a
fresh timed pass of the full-retrain oracle, whose *cost* — not its
resulting parameters — is what this script measures.

Usage:
    python3 scripts/run_reconstruction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qflewp.reconstruct import run

if __name__ == "__main__":
    run()
    print("Done. See results/json/extended_experiment.json")
