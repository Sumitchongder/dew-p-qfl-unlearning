#!/usr/bin/env python3
"""
Entry point: pruning-threshold sweep + non-IID / client-count ablations
(Table 9, Table 10, Figures 10-11).

Usage:
    python3 scripts/run_sweeps.py --stage all
    python3 scripts/run_sweeps.py --stage pruning_fraction
    python3 scripts/run_sweeps.py --stage ablation_non_iid
    python3 scripts/run_sweeps.py --stage ablation_num_clients
    python3 scripts/run_sweeps.py --stage scalability
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qflewp.pipeline import DEFAULTS
from src.qflewp.sweeps import (
    pruning_fraction_sweep, ablation_non_iid, ablation_num_clients, scalability_sweep,
)

LIGHT_CFG = {**DEFAULTS, "n_rounds": 4, "local_maxiter": 10, "samples_per_client": 130}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=[
        "all", "pruning_fraction", "ablation_non_iid", "ablation_num_clients", "scalability",
    ], default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.stage in ("all", "pruning_fraction"):
        print("[run_sweeps] pruning-fraction sweep ...")
        pruning_fraction_sweep(DEFAULTS, seed=args.seed)

    if args.stage in ("all", "ablation_non_iid"):
        print("[run_sweeps] non-IID ablation ...")
        ablation_non_iid(LIGHT_CFG, seed=args.seed)

    if args.stage in ("all", "ablation_num_clients"):
        print("[run_sweeps] client-count ablation ...")
        ablation_num_clients(LIGHT_CFG, seed=args.seed, levels=(3, 5, 7))

    if args.stage in ("all", "scalability"):
        print("[run_sweeps] scalability sweep (instant, no training) ...")
        scalability_sweep(seed=args.seed)

    print("Done. See results/tables/*.csv")


if __name__ == "__main__":
    main()
