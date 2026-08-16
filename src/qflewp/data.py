"""
Synthetic supply-chain risk dataset generator for the QFL-EWP paper.

Implements the generative procedure of Appendix C exactly:

    1. Sample a single shared latent risk direction w ~ N(0, I_d).
    2. For each client i, sample a client-specific offset
       delta_i ~ N(0, beta^2 * I_d), where beta is the non-IID strength.
    3. Draw each client's features as x ~ N(delta_i, I_d).
    4. Draw labels from a logistic model,
       y ~ Bernoulli(sigma(w^T x + eps)),  eps ~ N(0, noise^2).

All draws use a single NumPy Generator seeded with `1000 + run_seed`
(Appendix C / Table 1: "Data generation: Seeded (seed = 1000 + run seed)"),
so the entire dataset -- including the client partition and the
train/test split -- is exactly reproducible from the released code.

(An earlier revision of this module used a hand-specified fixed weight
vector, an added nonlinear interaction term, and a deterministic
prob >= 0.5 threshold instead of a genuine Bernoulli draw. That did not
match Appendix C's stated generative process and has been replaced with
the procedure above.)
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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _make_client(rng, client_id, n_samples, n_features, w, delta_i, noise, train_frac=0.75):
    # Step 3 (Appendix C): x ~ N(delta_i, I_d).
    X = rng.normal(loc=delta_i, scale=1.0, size=(n_samples, n_features))

    # Step 4 (Appendix C): y ~ Bernoulli(sigma(w^T x + eps)), eps ~ N(0, noise^2).
    eps = rng.normal(0.0, noise, size=n_samples)
    prob = _sigmoid(X @ w + eps)
    y = rng.binomial(1, prob).astype(int)

    n_train = int(train_frac * n_samples)
    return ClientDataset(
        client_id=client_id,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
    )


def generate_federated_dataset(
    n_clients: int = 5,
    samples_per_client: int = 180,
    n_features: int = 4,
    non_iid_strength: float = 0.9,
    noise: float = 0.2,
    seed: int = 1000,
):
    """Return a list of ClientDataset, one per client, non-IID partitioned,
    exactly following the generative procedure of Appendix C.

    `seed` is expected to be called as `1000 + run_seed` by the calling
    pipeline code (see src/qflewp/pipeline.py), matching Table 1's
    "Data generation: Seeded (seed = 1000 + run seed)".
    """
    rng = np.random.default_rng(seed)

    # Step 1 (Appendix C): a single shared latent risk direction w.
    w = rng.normal(0.0, 1.0, size=n_features)

    clients = []
    for c in range(n_clients):
        # Step 2 (Appendix C): client-specific offset delta_i ~ N(0, beta^2 I).
        delta_i = rng.normal(0.0, non_iid_strength, size=n_features)
        clients.append(
            _make_client(
                rng,
                client_id=c,
                n_samples=samples_per_client,
                n_features=n_features,
                w=w,
                delta_i=delta_i,
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
