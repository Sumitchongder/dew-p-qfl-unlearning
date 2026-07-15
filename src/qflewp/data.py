"""
Synthetic supply-chain risk dataset generator for the QFL-EWP paper.

Produces a binary risk-classification task ("Warning" vs "Healthy") split
non-IID across N clients, each with its own systematic bias (region/sector
effect) so that unlearning one client is scientifically meaningful (i.e.
that client's data really does leave an identifiable fingerprint in the
trained model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientDataset:
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _make_client(rng, client_id, n_samples, n_features, bias_vector, non_iid_strength, noise):
    # Each client's features are drawn around a client-specific bias
    # (non-IID covariate shift), plus shared task structure.
    X = rng.normal(loc=bias_vector * non_iid_strength, scale=1.0, size=(n_samples, n_features))

    # Ground-truth risk score is a fixed linear/quadratic function of the
    # features (shared across clients) plus client-specific noise.
    weight = np.array([1.4, -1.1, 0.9, 0.6] + [0.3] * max(0, n_features - 4))[:n_features]
    score = X @ weight + 0.4 * (X[:, 0] * X[:, 1]) + rng.normal(0, noise, n_samples)
    prob = 1 / (1 + np.exp(-score))
    y = (prob >= 0.5).astype(int)

    n_train = int(0.75 * n_samples)
    return ClientDataset(
        client_id=client_id,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
    )


def generate_federated_dataset(
    n_clients: int = 5,
    samples_per_client: int = 160,
    n_features: int = 4,
    non_iid_strength: float = 0.9,
    noise: float = 0.35,
    seed: int = 7,
):
    """Return a list of ClientDataset, one per client, non-IID partitioned."""
    rng = np.random.default_rng(seed)
    clients = []
    for c in range(n_clients):
        bias_vector = rng.normal(0, 1.0, n_features)
        clients.append(
            _make_client(
                rng,
                client_id=c,
                n_samples=samples_per_client,
                n_features=n_features,
                bias_vector=bias_vector,
                non_iid_strength=non_iid_strength,
                noise=noise,
            )
        )
    return clients


def pooled_test_set(clients, exclude_client: int | None = None):
    Xs, ys = [], []
    for c in clients:
        if exclude_client is not None and c.client_id == exclude_client:
            continue
        Xs.append(c.X_test)
        ys.append(c.y_test)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
