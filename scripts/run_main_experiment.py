#!/usr/bin/env python3
"""
Entry point: run the main N-seed benchmark (Table 3-8 source data).

Usage:
    python3 scripts/run_main_experiment.py --seeds 0 1 2 --n-rounds 6 \
        --local-maxiter 16 --samples-per-client 180

All arguments are optional; defaults match the configuration reported in
results/tables/table02_federated_training_configuration.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qflewp.pipeline import run_main_experiment, DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-qubits", type=int, default=DEFAULTS["n_qubits"])
    p.add_argument("--n-layers", type=int, default=DEFAULTS["n_layers"])
    p.add_argument("--n-clients", type=int, default=DEFAULTS["n_clients"])
    p.add_argument("--samples-per-client", type=int, default=DEFAULTS["samples_per_client"])
    p.add_argument("--non-iid-strength", type=float, default=DEFAULTS["non_iid_strength"])
    p.add_argument("--noise", type=float, default=DEFAULTS["noise"])
    p.add_argument("--n-rounds", type=int, default=DEFAULTS["n_rounds"])
    p.add_argument("--local-maxiter", type=int, default=DEFAULTS["local_maxiter"])
    p.add_argument("--forget-client", type=int, default=DEFAULTS["forget_client"])
    p.add_argument("--prune-fraction", type=float, default=DEFAULTS["prune_fraction"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(
        n_qubits=args.n_qubits, n_layers=args.n_layers, n_features=args.n_qubits,
        n_clients=args.n_clients, samples_per_client=args.samples_per_client,
        non_iid_strength=args.non_iid_strength, noise=args.noise,
        n_rounds=args.n_rounds, local_maxiter=args.local_maxiter,
        forget_client=args.forget_client, prune_fraction=args.prune_fraction,
    )
    print(f"[run_main_experiment] seeds={args.seeds} cfg={cfg}")
    results, df = run_main_experiment(seeds=tuple(args.seeds), cfg=cfg)
    print(df.groupby("method").agg(["mean", "std"]))
    print("\nWrote results/json/main_experiment_raw.json and results/tables/main_results_raw.csv")


if __name__ == "__main__":
    main()
