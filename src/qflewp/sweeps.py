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
    ewp_pruning, fisher_only_pruning, entanglement_only_pruning, random_pruning,
    full_retrain_oracle,
)
from src.qflewp.evaluate import evaluate_method

OUT = Path("results")


def pruning_fraction_sweep(cfg, seed=0, fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    clients = generate_federated_dataset(
        n_clients=cfg["n_clients"], samples_per_client=cfg["samples_per_client"],
        n_features=cfg["n_features"], non_iid_strength=cfg["non_iid_strength"],
        noise=cfg["noise"], seed=1000 + seed,
    )
    vqc = VQC(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"], n_features=cfg["n_features"])

    theta_full, _ = fedavg_train(vqc, clients, n_rounds=cfg["n_rounds"],
                                  local_epochs=cfg["local_maxiter"], seed=seed)

    forgotten = clients[cfg["forget_client"]]
    retained = [c for c in clients if c.client_id != cfg["forget_client"]]

    theta_oracle = full_retrain_oracle(vqc, clients, excluded_client=cfg["forget_client"],
                                        n_rounds=cfg["n_rounds"], local_epochs=cfg["local_maxiter"], seed=seed)

    qfim = DiagonalQFIM(vqc)
    f_kk = qfim.estimate(theta_full, forgotten.X_train)
    ent = EntanglementAnalyzer(vqc)
    w_ent = ent.compute(theta_full, forgotten.X_train).gate_weights

    rows = []
    methods = {
        "QFL-EWP": lambda frac: ewp_pruning(theta_full, f_kk, w_ent, fraction=frac),
        "Random Pruning": lambda frac: random_pruning(theta_full, fraction=frac, seed=seed),
        "Fisher-only Pruning": lambda frac: fisher_only_pruning(theta_full, f_kk, fraction=frac),
        "Entanglement-only Pruning": lambda frac: entanglement_only_pruning(theta_full, w_ent, fraction=frac),
    }

    for name, fn in methods.items():
        for frac in fractions:
            res = fn(frac)
            m = evaluate_method(vqc, res.pruned_weights, theta_oracle, retained, forgotten, seed=seed)
            rows.append({
                "method": name, "prune_fraction": frac,
                "accuracy": m["utility_accuracy"], "auroc": m["utility_auroc"],
                "forgetting_score": m["forgetting_score"],
                "retrain_distance": m["retrain_distance"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "pruning_fraction_sweep.csv", index=False)
    return df


def ablation_non_iid(cfg, seed=0, levels=(0.2, 0.9)):
    rows = []
    for level in levels:
        c2 = {**cfg, "non_iid_strength": level}
        clients = generate_federated_dataset(
            n_clients=c2["n_clients"], samples_per_client=c2["samples_per_client"],
            n_features=c2["n_features"], non_iid_strength=level,
            noise=c2["noise"], seed=2000 + seed,
        )
        vqc = VQC(n_qubits=c2["n_qubits"], n_layers=c2["n_layers"], n_features=c2["n_features"])
        theta_full, _ = fedavg_train(vqc, clients, n_rounds=c2["n_rounds"],
                                      local_epochs=c2["local_maxiter"], seed=seed)
        forgotten = clients[c2["forget_client"]]
        retained = [c for c in clients if c.client_id != c2["forget_client"]]
        theta_oracle = full_retrain_oracle(vqc, clients, excluded_client=c2["forget_client"],
                                            n_rounds=c2["n_rounds"], local_epochs=c2["local_maxiter"], seed=seed)
        qfim = DiagonalQFIM(vqc)
        f_kk = qfim.estimate(theta_full, forgotten.X_train)
        ent = EntanglementAnalyzer(vqc)
        w_ent = ent.compute(theta_full, forgotten.X_train).gate_weights
        ewp = ewp_pruning(theta_full, f_kk, w_ent, fraction=c2["prune_fraction"])
        m = evaluate_method(vqc, ewp.pruned_weights, theta_oracle, retained, forgotten, seed=seed)
        rows.append({"non_iid_strength": level, "accuracy": m["utility_accuracy"],
                      "forgetting_score": m["forgetting_score"], "auroc": m["utility_auroc"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "ablation_non_iid.csv", index=False)
    return df


def ablation_num_clients(cfg, seed=0, levels=(3, 5, 7)):
    rows = []
    for n in levels:
        c2 = {**cfg, "n_clients": n}
        clients = generate_federated_dataset(
            n_clients=n, samples_per_client=c2["samples_per_client"],
            n_features=c2["n_features"], non_iid_strength=c2["non_iid_strength"],
            noise=c2["noise"], seed=3000 + seed,
        )
        vqc = VQC(n_qubits=c2["n_qubits"], n_layers=c2["n_layers"], n_features=c2["n_features"])
        theta_full, _ = fedavg_train(vqc, clients, n_rounds=c2["n_rounds"],
                                      local_epochs=c2["local_maxiter"], seed=seed)
        forgotten = clients[0]
        retained = [c for c in clients if c.client_id != 0]
        theta_oracle = full_retrain_oracle(vqc, clients, excluded_client=0,
                                            n_rounds=c2["n_rounds"], local_epochs=c2["local_maxiter"], seed=seed)
        qfim = DiagonalQFIM(vqc)
        f_kk = qfim.estimate(theta_full, forgotten.X_train)
        ent = EntanglementAnalyzer(vqc)
        w_ent = ent.compute(theta_full, forgotten.X_train).gate_weights
        ewp = ewp_pruning(theta_full, f_kk, w_ent, fraction=c2["prune_fraction"])
        m = evaluate_method(vqc, ewp.pruned_weights, theta_oracle, retained, forgotten, seed=seed)
        rows.append({"n_clients": n, "accuracy": m["utility_accuracy"],
                      "forgetting_score": m["forgetting_score"], "auroc": m["utility_auroc"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "ablation_num_clients.csv", index=False)
    return df


def scalability_sweep(seed=0, qubit_levels=(3, 4, 5, 6)):
    """Circuit-evaluation cost vs qubit count (Figure 9 / Table 6)."""
    rows = []
    for nq in qubit_levels:
        vqc = VQC(n_qubits=nq, n_layers=3, n_features=nq)
        theta = vqc.initial_weights(seed=seed)
        X = np.random.default_rng(seed).uniform(-1, 1, size=(40, nq))

        t0 = time.time()
        _ = vqc.forward(theta, X)
        forward_time = time.time() - t0

        n_params = vqc.num_parameters
        # Analytic circuit-evaluation counts (not wall-clock, which is
        # simulator-specific): one parameter-shift gradient step needs
        # 2 circuit evaluations per parameter; one diagonal-QFIM estimate
        # needs 1 shifted evaluation per parameter (Algorithm 3).
        shift_gradient_evals = 2 * n_params
        qfim_evals = n_params
        retrain_evals = shift_gradient_evals  # per local optimization step

        rows.append({
            "n_qubits": nq, "n_params": n_params,
            "forward_pass_seconds": forward_time,
            "grad_step_circuit_evals": shift_gradient_evals,
            "qfim_circuit_evals": qfim_evals,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "scalability.csv", index=False)
    return df
