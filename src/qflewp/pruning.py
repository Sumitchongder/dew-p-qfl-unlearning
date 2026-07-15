"""
Algorithm 2 (EWP) and the five baselines used throughout the paper:
full retrain, fine-tune only, random pruning, Fisher-only pruning,
entanglement-only pruning.

Pruning is implemented by zeroing the parameter, which is exactly
equivalent to replacing the gate with the identity (RY(0) = RZ(0) = I),
matching the "prune -> identity gate" rule described in the method section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PruneResult:
    method_name: str
    pruned_weights: np.ndarray
    mask: np.ndarray            # True = pruned (zeroed)
    scores: np.ndarray
    threshold: float
    fraction_pruned: float


def _threshold_for_fraction(scores: np.ndarray, fraction: float) -> float:
    if fraction <= 0:
        return -np.inf
    if fraction >= 1:
        return np.inf
    k = max(1, int(round(fraction * len(scores))))
    return float(np.sort(scores)[k - 1])


def prune_by_score(theta, scores, fraction, method_name):
    tau = _threshold_for_fraction(scores, fraction)
    mask = scores <= tau
    pruned = theta.copy()
    pruned[mask] = 0.0
    return PruneResult(
        method_name=method_name,
        pruned_weights=pruned,
        mask=mask,
        scores=scores,
        threshold=tau,
        fraction_pruned=float(mask.mean()),
    )


def ewp_pruning(theta, qfim_diag, entanglement_weights, fraction=0.25):
    """Algorithm 2: s_k = w_ent(k) * F_kk(j); prune lowest-scoring params."""
    scores = entanglement_weights * qfim_diag
    return prune_by_score(theta, scores, fraction, "QFL-EWP")


def fisher_only_pruning(theta, qfim_diag, fraction=0.25):
    return prune_by_score(theta, qfim_diag, fraction, "Fisher-only Pruning")


def entanglement_only_pruning(theta, entanglement_weights, fraction=0.25):
    return prune_by_score(theta, entanglement_weights, fraction, "Entanglement-only Pruning")


def random_pruning(theta, fraction=0.25, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.random(len(theta))
    return prune_by_score(theta, scores, fraction, "Random Pruning")


def fine_tune_only(vqc, theta, retained_clients, rounds=3, maxiter=10):
    """'Fine-tune only' baseline: no pruning at all, just a few extra
    federated rounds restricted to the retained clients. This is the
    naive unlearning approach the paper argues is insufficient on its
    own (it never explicitly removes the forgotten client's influence)."""
    from src.qflewp.federated import fedavg_train

    new_theta, _ = fedavg_train(
        vqc, retained_clients, n_rounds=rounds, local_epochs=maxiter,
        init_theta=theta,
    )
    return new_theta


def recovery_fine_tune(vqc, pruned_theta, mask, retained_clients, rounds=2, maxiter=10):
    """Optional recovery step after pruning (Section 4.5): re-optimize the
    *surviving* parameters only on retained-client data, keeping pruned
    parameters frozen at zero."""
    from scipy.optimize import minimize
    from src.qflewp.federated import bce_loss

    X = np.concatenate([c.X_train for c in retained_clients], axis=0)
    y = np.concatenate([c.y_train for c in retained_clients], axis=0)

    free_idx = np.where(~mask)[0]
    theta = pruned_theta.copy()

    if len(free_idx) == 0:
        return theta

    def loss_and_grad(free_vals):
        full = theta.copy()
        full[free_idx] = free_vals
        prob = vqc.predict_proba(full, X)
        L = bce_loss(y, prob)

        from src.qflewp.federated import _param_shift_gradient
        g_full = _param_shift_gradient(vqc, full, X, y)
        return L, g_full[free_idx]

    res = minimize(loss_and_grad, theta[free_idx], jac=True,
                    method="L-BFGS-B", options={"maxiter": maxiter})
    theta[free_idx] = res.x
    return theta


def full_retrain_oracle(vqc, clients, excluded_client, n_rounds=6, local_epochs=15, seed=42):
    """The gold-standard baseline: train from scratch with the forgotten
    client's data never having participated at all."""
    from src.qflewp.federated import fedavg_train

    theta, _ = fedavg_train(
        vqc, clients, n_rounds=n_rounds, local_epochs=local_epochs,
        seed=seed, excluded_client=excluded_client,
    )
    return theta
