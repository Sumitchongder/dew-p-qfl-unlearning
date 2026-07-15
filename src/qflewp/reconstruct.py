"""
Reconstructs per-method predictions, confusion matrices, and timings from
the already-trained models saved in main_experiment_raw.json (theta_full,
theta_oracle, qfim_diag, entanglement_weights are all deterministic given
the stored seed, so no retraining of the shared federated model is
needed — only the cheap per-method pruning/eval steps are recomputed here,
plus a fresh timing pass for the full-retrain oracle since its cost has to
be measured, not just its resulting parameters).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from src.qflewp.circuit import VQC
from src.qflewp.data import generate_federated_dataset
from src.qflewp.federated import fedavg_train
from src.qflewp.pruning import (
    ewp_pruning, fisher_only_pruning, entanglement_only_pruning, random_pruning,
    fine_tune_only, recovery_fine_tune, full_retrain_oracle,
)
from src.qflewp.evaluate import (
    utility_accuracy, utility_auroc, retrain_distance, membership_inference_advantage,
    forgetting_score,
)

RESULTS = Path("results/json/main_experiment_raw.json")
OUT_JSON = Path("results/json/extended_experiment.json")


def reconstruct_seed(seed_result: dict, cfg: dict) -> dict:
    seed = seed_result["seed"]
    clients = generate_federated_dataset(
        n_clients=cfg["n_clients"], samples_per_client=cfg["samples_per_client"],
        n_features=cfg["n_features"], non_iid_strength=cfg["non_iid_strength"],
        noise=cfg["noise"], seed=1000 + seed,
    )
    vqc = VQC(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"], n_features=cfg["n_features"])

    theta_full = np.array(seed_result["theta_full"])
    theta_oracle = np.array(seed_result["theta_oracle"])
    f_kk = np.array(seed_result["qfim_diag"])
    w_ent = np.array(seed_result["entanglement_weights"])

    forgotten = clients[cfg["forget_client"]]
    retained = [c for c in clients if c.client_id != cfg["forget_client"]]
    X_ret = np.concatenate([c.X_test for c in retained])
    y_ret = np.concatenate([c.y_test for c in retained])

    frac = cfg["prune_fraction"]
    methods = {}

    # ---- pruning-based methods: cheap, deterministic given stored arrays ----
    def timed_eval(name, theta_method):
        t0 = time.perf_counter()
        prob = vqc.predict_proba(theta_method, X_ret)
        pred = (prob >= 0.5).astype(int)
        elapsed = time.perf_counter() - t0
        mi = membership_inference_advantage(vqc, theta_method, forgotten, retained, seed=seed)
        return {
            "y_true": y_ret.tolist(), "y_pred": pred.tolist(), "y_prob": prob.tolist(),
            "confusion_matrix": confusion_matrix(y_ret, pred, labels=[0, 1]).tolist(),
            "accuracy": utility_accuracy(vqc, theta_method, X_ret, y_ret),
            "auroc": utility_auroc(vqc, theta_method, X_ret, y_ret),
            "membership_advantage": mi["advantage"], "attack_auc": mi["attack_auc"],
            "forgetting_score": forgetting_score(mi), "roc": mi["roc"],
            "retrain_distance": retrain_distance(theta_method, theta_oracle),
            "inference_seconds": elapsed,
        }

    t0 = time.perf_counter()
    rnd = random_pruning(theta_full, fraction=frac, seed=seed)
    rnd_time = time.perf_counter() - t0
    r = timed_eval("Random Pruning", rnd.pruned_weights); r["unlearn_seconds"] = rnd_time
    methods["Random Pruning"] = r

    t0 = time.perf_counter()
    fis = fisher_only_pruning(theta_full, f_kk, fraction=frac)
    fis_time = time.perf_counter() - t0
    r = timed_eval("Fisher-only Pruning", fis.pruned_weights); r["unlearn_seconds"] = fis_time
    methods["Fisher-only Pruning"] = r

    t0 = time.perf_counter()
    ento = entanglement_only_pruning(theta_full, w_ent, fraction=frac)
    ento_time = time.perf_counter() - t0
    r = timed_eval("Entanglement-only Pruning", ento.pruned_weights); r["unlearn_seconds"] = ento_time
    methods["Entanglement-only Pruning"] = r

    t0 = time.perf_counter()
    ft_theta = fine_tune_only(vqc, theta_full, retained, rounds=2, maxiter=10)
    ft_time = time.perf_counter() - t0
    r = timed_eval("Fine-Tune Only", ft_theta); r["unlearn_seconds"] = ft_time
    methods["Fine-Tune Only"] = r

    t0 = time.perf_counter()
    ewp = ewp_pruning(theta_full, f_kk, w_ent, fraction=frac)
    ewp_theta = recovery_fine_tune(vqc, ewp.pruned_weights, ewp.mask, retained, rounds=2, maxiter=10)
    ewp_time = time.perf_counter() - t0
    r = timed_eval("QFL-EWP", ewp_theta); r["unlearn_seconds"] = ewp_time
    methods["QFL-EWP"] = r

    # ---- Full retrain oracle: the expensive baseline; time it for real ----
    t0 = time.perf_counter()
    theta_oracle_fresh = full_retrain_oracle(
        vqc, clients, excluded_client=cfg["forget_client"],
        n_rounds=cfg["n_rounds"], local_epochs=cfg["local_maxiter"], seed=seed,
    )
    oracle_time = time.perf_counter() - t0
    r = timed_eval("Full Retrain (oracle)", theta_oracle_fresh); r["unlearn_seconds"] = oracle_time
    methods["Full Retrain (oracle)"] = r

    return {"seed": seed, "methods": methods}


def run(cfg=None):
    with open(RESULTS) as f:
        raw = json.load(f)
    cfg = cfg or raw["config"]

    extended = {"config": cfg, "results": []}
    for seed_result in raw["results"]:
        print("reconstructing seed", seed_result["seed"])
        extended["results"].append(reconstruct_seed(seed_result, cfg))

    with open(OUT_JSON, "w") as f:
        json.dump(extended, f, indent=2)
    print("wrote", OUT_JSON)
    return extended


if __name__ == "__main__":
    run()
