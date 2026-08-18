"""
Evaluation suite (Table 3 in the paper): utility, forgetting / membership
inference, and retrain distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    
    rng = np.random.default_rng(seed)

    member_X = forgotten_client.X_train
    non_member_X = np.concatenate([c.X_test for c in held_out_clients], axis=0)

    n = min(len(member_X), len(non_member_X))
    if n < 4:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    idx_m = rng.choice(len(member_X), size=n, replace=False)
    idx_n = rng.choice(len(non_member_X), size=n, replace=False)

    member_conf = np.abs(vqc.predict_proba(theta, member_X[idx_m]) - 0.5) * 2
    non_member_conf = np.abs(vqc.predict_proba(theta, non_member_X[idx_n]) - 0.5) * 2

    scores = np.concatenate([member_conf, non_member_conf]).reshape(-1, 1)
    labels = np.concatenate([np.ones(n), np.zeros(n)])  # 1 = member

    if len(np.unique(labels)) < 2:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    # Shuffle, then split into an attacker-training half and an
    # attacker-evaluation half (Shokri et al. shadow-model protocol:
    # the attacker is *fit*, then evaluated on held-out members/non-members).
    order = rng.permutation(len(labels))
    scores, labels = scores[order], labels[order]
    split = len(labels) // 2
    fit_X, fit_y = scores[:split], labels[:split]
    eval_X, eval_y = scores[split:], labels[split:]

    if len(np.unique(fit_y)) < 2 or len(np.unique(eval_y)) < 2:
        # Degenerate small-sample split: fall back to fitting and
        # evaluating on the full pool rather than failing the metric.
        fit_X, fit_y, eval_X, eval_y = scores, labels, scores, labels

    attacker = LogisticRegression()
    attacker.fit(fit_X, fit_y)
    attack_scores = attacker.predict_proba(eval_X)[:, 1]

    auc = roc_auc_score(eval_y, attack_scores)
    fpr, tpr, _ = roc_curve(eval_y, attack_scores)
    advantage = float(2 * (auc - 0.5))
    return {"advantage": advantage, "attack_auc": float(auc), "roc": (fpr.tolist(), tpr.tolist())}


def forgetting_score(vqc: VQC, theta, theta_oracle, forgotten_client) -> float:
   
    prob = vqc.predict_proba(theta, forgotten_client.X_test)
    prob_oracle = vqc.predict_proba(theta_oracle, forgotten_client.X_test)
    return float(np.mean(np.abs(prob - prob_oracle)))


def evaluate_method(vqc, theta, theta_oracle, retained_clients, forgotten_client, seed=0):
    X_ret, y_ret = np.concatenate([c.X_test for c in retained_clients]), \
        np.concatenate([c.y_test for c in retained_clients])

    mi = membership_inference_advantage(vqc, theta, forgotten_client, retained_clients, seed=seed)

    return {
        "utility_accuracy": utility_accuracy(vqc, theta, X_ret, y_ret),
        "utility_auroc": utility_auroc(vqc, theta, X_ret, y_ret),
        # forgetting_score = mean |p - p_oracle| on the forgotten client's
        # held-out data (Section 5.4 / Appendix D); see docstring above.
        "forgetting_score": forgetting_score(vqc, theta, theta_oracle, forgotten_client),
        "membership_advantage": mi["advantage"],
        "attack_auc": mi["attack_auc"],
        "retrain_distance": retrain_distance(theta, theta_oracle),
        "roc": mi["roc"],
    }
