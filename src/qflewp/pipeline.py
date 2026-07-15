"""
End-to-end experiment pipeline: dataset -> federated training -> forget
request -> EWP + 5 baselines -> evaluation -> JSON/CSV artifacts.

This is the single source of truth that produces every number that ends up
in the paper's figures and tables. Run with:

    python3 -m src.qflewp.pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.qflewp.circuit import VQC
from src.qflewp.data import generate_federated_dataset
from src.qflewp.federated import fedavg_train
from src.qflewp.qfim import DiagonalQFIM
from src.qflewp.entanglement import EntanglementAnalyzer
from src.qflewp.pruning import (
    ewp_pruning, fisher_only_pruning, entanglement_only_pruning,
    random_pruning, fine_tune_only, recovery_fine_tune, full_retrain_oracle,
)
from src.qflewp.evaluate import evaluate_method

OUT = Path("results")
(OUT / "json").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

DEFAULTS = dict(
    n_qubits=4, n_layers=3, n_features=4,
    n_clients=5, samples_per_client=180, non_iid_strength=0.9, noise=0.2,
    n_rounds=6, local_maxiter=16,
    forget_client=0,
    prune_fraction=0.2,
)


def run_single_seed(seed: int, cfg: dict) -> dict:
    t0 = time.time()
    clients = generate_federated_dataset(
        n_clients=cfg["n_clients"], samples_per_client=cfg["samples_per_client"],
        n_features=cfg["n_features"], non_iid_strength=cfg["non_iid_strength"],
        noise=cfg["noise"], seed=1000 + seed,
    )
    vqc = VQC(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"], n_features=cfg["n_features"])

    theta_full, hist_full = fedavg_train(
        vqc, clients, n_rounds=cfg["n_rounds"], local_epochs=cfg["local_maxiter"], seed=seed,
    )

    forgotten = clients[cfg["forget_client"]]
    retained = [c for c in clients if c.client_id != cfg["forget_client"]]

    theta_oracle = full_retrain_oracle(
        vqc, clients, excluded_client=cfg["forget_client"],
        n_rounds=cfg["n_rounds"], local_epochs=cfg["local_maxiter"], seed=seed,
    )

    qfim = DiagonalQFIM(vqc)
    f_kk = qfim.estimate(theta_full, forgotten.X_train)

    ent = EntanglementAnalyzer(vqc)
    ent_result = ent.compute(theta_full, forgotten.X_train)
    w_ent = ent_result.gate_weights

    frac = cfg["prune_fraction"]

    outcomes = {}

    # ---- Random pruning ----
    rnd = random_pruning(theta_full, fraction=frac, seed=seed)
    outcomes["Random Pruning"] = evaluate_method(vqc, rnd.pruned_weights, theta_oracle, retained, forgotten, seed=seed)

    # ---- Fisher-only pruning ----
    fis = fisher_only_pruning(theta_full, f_kk, fraction=frac)
    outcomes["Fisher-only Pruning"] = evaluate_method(vqc, fis.pruned_weights, theta_oracle, retained, forgotten, seed=seed)

    # ---- Entanglement-only pruning ----
    ento = entanglement_only_pruning(theta_full, w_ent, fraction=frac)
    outcomes["Entanglement-only Pruning"] = evaluate_method(vqc, ento.pruned_weights, theta_oracle, retained, forgotten, seed=seed)

    # ---- Fine-tune only (no pruning) ----
    ft_theta = fine_tune_only(vqc, theta_full, retained, rounds=2, maxiter=10)
    outcomes["Fine-Tune Only"] = evaluate_method(vqc, ft_theta, theta_oracle, retained, forgotten, seed=seed)

    # ---- Full retrain oracle itself (upper bound reference) ----
    outcomes["Full Retrain (oracle)"] = evaluate_method(vqc, theta_oracle, theta_oracle, retained, forgotten, seed=seed)

    # ---- QFL-EWP (ours) ----
    ewp = ewp_pruning(theta_full, f_kk, w_ent, fraction=frac)
    ewp_recovered = recovery_fine_tune(vqc, ewp.pruned_weights, ewp.mask, retained, rounds=2, maxiter=10)
    outcomes["QFL-EWP"] = evaluate_method(vqc, ewp_recovered, theta_oracle, retained, forgotten, seed=seed)

    elapsed = time.time() - t0

    return dict(
        seed=seed,
        elapsed_seconds=elapsed,
        outcomes=outcomes,
        training_history=dict(loss=hist_full.global_loss, accuracy=hist_full.global_accuracy),
        qfim_diag=f_kk.tolist(),
        entanglement_weights=w_ent.tolist(),
        entanglement_per_layer_qubit=ent_result.per_layer_qubit.tolist(),
        ewp_scores=(f_kk * w_ent).tolist(),
        ewp_mask=ewp.mask.tolist(),
        ewp_threshold=ewp.threshold,
        theta_full=theta_full.tolist(),
        theta_oracle=theta_oracle.tolist(),
    )


def run_main_experiment(seeds=(0, 1, 2), cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    results = [run_single_seed(s, cfg) for s in seeds]

    with open(OUT / "json" / "main_experiment_raw.json", "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)

    rows = []
    for r in results:
        for method, m in r["outcomes"].items():
            rows.append({
                "seed": r["seed"], "method": method,
                "accuracy": m["utility_accuracy"], "auroc": m["utility_auroc"],
                "forgetting_score": m["forgetting_score"],
                "membership_advantage": m["membership_advantage"],
                "retrain_distance": m["retrain_distance"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "main_results_raw.csv", index=False)
    return results, df


if __name__ == "__main__":
    results, df = run_main_experiment()
    print(df.groupby("method").agg(["mean", "std"]))
