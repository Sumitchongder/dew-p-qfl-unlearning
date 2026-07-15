"""
Federated training of the VQC risk model (Algorithm 1 in the paper).

Local updates use the exact parameter-shift rule (valid because every
trainable gate is a Pauli rotation with eigenvalues +-1/2), so gradients
are exact, not finite-difference approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.qflewp.circuit import VQC


def bce_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(-np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob)))


def _param_shift_gradient(vqc: VQC, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact gradient of the batch BCE loss w.r.t. every trainable parameter."""
    n_params = vqc.num_parameters
    grad = np.zeros(n_params)

    expz0 = vqc.forward(theta, X)
    prob0 = np.clip((1 + expz0) / 2, 1e-6, 1 - 1e-6)
    # dL/dprob for BCE, averaged over batch
    dL_dprob = (-y / prob0 + (1 - y) / (1 - prob0)) / len(y)
    dprob_dexpz = 0.5

    for k in range(n_params):
        shift = np.zeros(n_params)
        shift[k] = np.pi / 2
        expz_plus = vqc.forward(theta + shift, X)
        expz_minus = vqc.forward(theta - shift, X)
        dexpz_dtheta = 0.5 * (expz_plus - expz_minus)
        grad[k] = np.sum(dL_dprob * dprob_dexpz * dexpz_dtheta)

    return grad


@dataclass
class TrainingHistory:
    rounds: list = field(default_factory=list)
    global_loss: list = field(default_factory=list)
    global_accuracy: list = field(default_factory=list)


def local_train(vqc: VQC, theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                 epochs: int = 6, lr: float = 0.35, maxiter: int = 20) -> np.ndarray:
    """Local client optimization using L-BFGS-B with the exact
    parameter-shift gradient. `epochs`/`lr` are kept as arguments for
    backward compatibility but the optimizer does its own line search;
    `maxiter` controls the local compute budget per federated round.
    """
    from scipy.optimize import minimize

    def loss_and_grad(t):
        prob = vqc.predict_proba(t, X)
        L = bce_loss(y, prob)
        g = _param_shift_gradient(vqc, t, X, y)
        return L, g

    res = minimize(loss_and_grad, theta, jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter})
    return res.x


def evaluate(vqc: VQC, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    prob = vqc.predict_proba(theta, X)
    pred = (prob >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    loss = bce_loss(y, prob)
    return {"accuracy": acc, "loss": loss, "prob": prob, "pred": pred}


def fedavg_train(
    vqc: VQC,
    clients,
    n_rounds: int = 8,
    local_epochs: int = 6,
    lr: float = 0.35,
    seed: int = 42,
    excluded_client: int | None = None,
    init_theta: np.ndarray | None = None,
):
    """Standard FedAvg: local parameter-shift SGD per client, then weight
    averaging on the server (weighted by client dataset size).

    If `excluded_client` is set, that client does not participate — this is
    used to build the "full retrain without client j" oracle baseline.
    """
    theta = init_theta.copy() if init_theta is not None else vqc.initial_weights(seed=seed)
    history = TrainingHistory()

    active_clients = [c for c in clients if c.client_id != excluded_client]

    for r in range(n_rounds):
        local_thetas = []
        weights = []
        for client in active_clients:
            local_theta = local_train(
                vqc, theta, client.X_train, client.y_train,
                maxiter=local_epochs,
            )
            local_thetas.append(local_theta)
            weights.append(len(client.X_train))

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        theta = np.average(np.stack(local_thetas, axis=0), axis=0, weights=weights)

        # Track global progress on the pooled active-client test data
        X_eval = np.concatenate([c.X_test for c in active_clients], axis=0)
        y_eval = np.concatenate([c.y_test for c in active_clients], axis=0)
        metrics = evaluate(vqc, theta, X_eval, y_eval)
        history.rounds.append(r)
        history.global_loss.append(metrics["loss"])
        history.global_accuracy.append(metrics["accuracy"])

    return theta, history
