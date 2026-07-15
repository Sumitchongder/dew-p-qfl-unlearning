#!/usr/bin/env python3
"""
Entry point: generate the complete deliverable set -- 10 CSV tables and
16 figures (PNG + PDF each) -- from results/json/main_experiment_raw.json
and results/json/extended_experiment.json.

Must be run after run_main_experiment.py, run_sweeps.py, and
run_reconstruction.py have produced their JSON/CSV artifacts.

Usage:
    python3 scripts/generate_deliverables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qflewp.q1_deliverables import generate_all

if __name__ == "__main__":
    generate_all()
    print("Done. See results/figures/*.{png,pdf} and results/tables/*.csv")
