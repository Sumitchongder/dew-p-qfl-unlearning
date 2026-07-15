"""
Fast correctness checks (< 15 s total) for the core qflewp components.
These are unit-level sanity checks, not a substitute for the full
experimental pipeline in scripts/run_full_pipeline.sh.

Run with:
    pytest tests/ -v
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qflewp.circuit import VQC
from src.qflewp.data import generate_federated_dataset
from src.qflewp.qfim import DiagonalQFIM
from src.qflewp.entanglement import EntanglementAnalyzer
from src.qflewp.pruning import ewp_pruning, random_pruning, prune_by_score


def test_circuit_is_unitary():
    """The statevector norm must be exactly 1 after the full circuit."""
    vqc = VQC(n_qubits=4, n_layers=3, n_features=4)
    theta = vqc.initial_weights(seed=0)
    X = np.random.default_rng(0).uniform(-1, 1, size=(10, 4))
    state, _ = vqc.forward_full(theta, X)
    norms = np.sum(np.abs(state) ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_parameter_count_matches_ansatz():
    vqc = VQC(n_qubits=4, n_layers=3, n_features=4)
    assert vqc.num_parameters == 4 * 2 * 3  # RY + RZ per qubit per layer


def test_predict_proba_in_unit_interval():
    vqc = VQC(n_qubits=4, n_layers=3, n_features=4)
    theta = vqc.initial_weights(seed=1)
    X = np.random.default_rng(1).uniform(-1, 1, size=(20, 4))
    p = vqc.predict_proba(theta, X)
    assert np.all((p >= 0) & (p <= 1))


def test_dataset_generation_is_deterministic():
    a = generate_federated_dataset(n_clients=3, samples_per_client=20, seed=7)
    b = generate_federated_dataset(n_clients=3, samples_per_client=20, seed=7)
    assert np.allclose(a[0].X_train, b[0].X_train)
    assert np.array_equal(a[0].y_train, b[0].y_train)


def test_qfim_is_nonnegative_and_client_dependent():
    vqc = VQC(n_qubits=4, n_layers=2, n_features=4)
    theta = vqc.initial_weights(seed=2)
    clients = generate_federated_dataset(n_clients=2, samples_per_client=30, seed=2)
    qfim = DiagonalQFIM(vqc)
    f0 = qfim.estimate(theta, clients[0].X_train)
    f1 = qfim.estimate(theta, clients[1].X_train)
    assert np.all(f0 >= 0) and np.all(f1 >= 0)
    assert not np.allclose(f0, f1), "QFIM must depend on the client's data"


def test_entanglement_weight_bounded_and_differentiated():
    vqc = VQC(n_qubits=4, n_layers=3, n_features=4)
    theta = vqc.initial_weights(seed=3)
    clients = generate_federated_dataset(n_clients=1, samples_per_client=30, seed=3)
    result = EntanglementAnalyzer(vqc).compute(theta, clients[0].X_train)
    assert np.all(result.gate_weights >= 0) and np.all(result.gate_weights <= 1.0001)
    assert result.per_layer_qubit.std() > 1e-4, "weights must not be a single flat scalar"


def test_pruning_zeros_exactly_the_target_fraction():
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, 24)
    scores = np.random.default_rng(4).random(24)
    result = prune_by_score(theta, scores, fraction=0.25, method_name="test")
    n_pruned = int(np.sum(result.pruned_weights == 0.0))
    assert n_pruned == round(0.25 * 24)


def test_ewp_score_is_product_of_qfim_and_entanglement():
    vqc = VQC(n_qubits=4, n_layers=2, n_features=4)
    theta = vqc.initial_weights(seed=5)
    clients = generate_federated_dataset(n_clients=1, samples_per_client=20, seed=5)
    f_kk = DiagonalQFIM(vqc).estimate(theta, clients[0].X_train)
    w_ent = EntanglementAnalyzer(vqc).compute(theta, clients[0].X_train).gate_weights
    result = ewp_pruning(theta, f_kk, w_ent, fraction=0.25)
    assert np.allclose(result.scores, w_ent * f_kk)


def test_random_pruning_is_seed_reproducible():
    theta = np.ones(24)
    r1 = random_pruning(theta, fraction=0.3, seed=42)
    r2 = random_pruning(theta, fraction=0.3, seed=42)
    assert np.array_equal(r1.mask, r2.mask)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
