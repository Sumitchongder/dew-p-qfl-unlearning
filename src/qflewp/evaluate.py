"""
Evaluation suite (Table 3 in the paper): utility, forgetting / membership
inference, and retrain distance — computed from the *actual* trained model's
predictions, never from synthetic random numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.qflewp.circuit import VQC


def utility_accuracy(vqc: VQC, theta, X, y) -> float:
    prob = vqc.predict_proba(theta, X)
    pred = (prob >= 0.5).astype(int)
    return float(np.mean(pred == y))


def utility_auroc(vqc: VQC, theta, X, y) -> float:
    prob = vqc.predict_proba(theta, X)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def retrain_distance(theta_unlearned, theta_oracle) -> float:
    """L2 distance in parameter space to the full-retrain oracle -- the
    standard closeness-to-exact-unlearning proxy."""
    return float(np.linalg.norm(theta_unlearned - theta_oracle))


def membership_inference_advantage(vqc: VQC, theta, forgotten_client, held_out_clients, seed=0):
    """Confidence-thresholding membership-inference attack (Yeom et al.
    style membership advantage): an attacker sees the model's prediction
    confidence on (a) the forgotten client's *training* samples (members)
    and (b) samples from clients that were genuinely never trained on
    (non-members: retained clients' test splits), and tries to tell them
    apart purely from confidence. Advantage = 2*(AUC - 0.5), i.e. 0 means
    the attacker cannot distinguish members from non-members (good
    forgetting); 1 means perfect membership leakage.
    """
    rng = np.random.default_rng(seed)

    member_X = forgotten_client.X_train
    non_member_X = np.concatenate([c.X_test for c in held_out_clients], axis=0)

    n = min(len(member_X), len(non_member_X))
    if n == 0:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    idx_m = rng.choice(len(member_X), size=n, replace=False)
    idx_n = rng.choice(len(non_member_X), size=n, replace=False)

    member_conf = np.abs(vqc.predict_proba(theta, member_X[idx_m]) - 0.5) * 2
    non_member_conf = np.abs(vqc.predict_proba(theta, non_member_X[idx_n]) - 0.5) * 2

    scores = np.concatenate([member_conf, non_member_conf])
    labels = np.concatenate([np.ones(n), np.zeros(n)])  # 1 = member

    if len(np.unique(labels)) < 2:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    advantage = float(2 * (auc - 0.5))
    return {"advantage": advantage, "attack_auc": float(auc), "roc": (fpr.tolist(), tpr.tolist())}


def forgetting_score(membership_result: dict) -> float:
    """1 - membership advantage, rescaled to [0, 1] where 1 = perfect
    forgetting (attacker at chance level)."""
    return float(1 - abs(membership_result["advantage"]))


def evaluate_method(vqc, theta, theta_oracle, retained_clients, forgotten_client, seed=0):
    X_ret, y_ret = np.concatenate([c.X_test for c in retained_clients]), \
        np.concatenate([c.y_test for c in retained_clients])

    mi = membership_inference_advantage(vqc, theta, forgotten_client, retained_clients, seed=seed)

    return {
        "utility_accuracy": utility_accuracy(vqc, theta, X_ret, y_ret),
        "utility_auroc": utility_auroc(vqc, theta, X_ret, y_ret),
        "forgetting_score": forgetting_score(mi),
        "membership_advantage": mi["advantage"],
        "attack_auc": mi["attack_auc"],
        "retrain_distance": retrain_distance(theta, theta_oracle),
        "roc": mi["roc"],
    }
