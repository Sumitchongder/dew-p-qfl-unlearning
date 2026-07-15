"""
Real statevector-based variational quantum circuit (VQC) simulator.

This replaces the placeholder circuit in src/quantum/model.py, which never
embedded data into the circuit and therefore could not produce a data-
dependent Quantum Fisher Information Matrix (QFIM).

Architecture (data re-uploading ansatz):
    for each layer l in [0 .. n_layers-1]:
        RX(x_i)  on every qubit i        <- data encoding (not trainable)
        RY(theta) on every qubit          <- trainable
        ring-CNOT entangling block
        RZ(theta) on every qubit          <- trainable

Qubit index convention: qubit q corresponds to bit q of the flat statevector
index (i.e. qubit 0 is the least-significant bit). This is the convention
used consistently by apply_1q / apply_cnot / reduced_density_matrix below.

Everything is batched over samples so that a full client mini-batch is
evolved in one call, which keeps parameter-shift gradients and parameter-
shift QFIM estimation fast in pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Single-qubit rotation matrices (Qiskit convention)
# ----------------------------------------------------------------------

def _rx(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_1d(theta)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    u = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    u[:, 0, 0] = c
    u[:, 0, 1] = -1j * s
    u[:, 1, 0] = -1j * s
    u[:, 1, 1] = c
    return u


def _ry(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_1d(theta)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    u = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    u[:, 0, 0] = c
    u[:, 0, 1] = -s
    u[:, 1, 0] = s
    u[:, 1, 1] = c
    return u


def _rz(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_1d(theta)
    u = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    u[:, 0, 0] = np.exp(-1j * theta / 2)
    u[:, 1, 1] = np.exp(1j * theta / 2)
    return u


def apply_1q(state: np.ndarray, qubit: int, u: np.ndarray) -> np.ndarray:
    """Apply a (possibly per-sample) 2x2 gate to `qubit`.

    state : (batch, dim) complex
    u     : (batch, 2, 2) or (1, 2, 2) complex
    """
    dim = state.shape[1]
    idx = np.arange(dim)
    idx0 = idx[(idx >> qubit) & 1 == 0]
    idx1 = idx0 + (1 << qubit)

    a = state[:, idx0]
    b = state[:, idx1]

    u00 = u[:, 0, 0][:, None]
    u01 = u[:, 0, 1][:, None]
    u10 = u[:, 1, 0][:, None]
    u11 = u[:, 1, 1][:, None]

    new_a = u00 * a + u01 * b
    new_b = u10 * a + u11 * b

    out = state.copy()
    out[:, idx0] = new_a
    out[:, idx1] = new_b
    return out


def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    dim = state.shape[1]
    idx = np.arange(dim)
    mask = ((idx >> control) & 1) == 1
    perm = idx.copy()
    perm[mask] = idx[mask] ^ (1 << target)
    return state[:, perm]


def reduced_density_matrix(state: np.ndarray, qubit: int) -> np.ndarray:
    """Average (over the batch) single-qubit reduced density matrix."""
    dim = state.shape[1]
    idx = np.arange(dim)
    idx0 = idx[(idx >> qubit) & 1 == 0]
    idx1 = idx0 + (1 << qubit)

    v0 = state[:, idx0]
    v1 = state[:, idx1]

    rho00 = np.mean(np.sum(np.abs(v0) ** 2, axis=1))
    rho11 = np.mean(np.sum(np.abs(v1) ** 2, axis=1))
    rho01 = np.mean(np.sum(v0 * np.conj(v1), axis=1))

    rho = np.array([[rho00, rho01], [np.conj(rho01), rho11]], dtype=complex)
    return rho


def von_neumann_entropy(rho: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals.real, 1e-12, 1.0)
    return float(-np.sum(eigvals * np.log2(eigvals)))


@dataclass
class ParameterInfo:
    index: int
    layer: int
    qubit: int
    gate: str


@dataclass
class VQC:
    """Variational quantum classifier with data re-uploading."""

    n_qubits: int = 4
    n_layers: int = 3
    n_features: int = 4
    # Kept small (< pi) deliberately: large re-uploading scales alias
    # around the Bloch sphere and destroy the encoding's monotonicity,
    # which empirically wrecks trainability. 0.5 rad was tuned on the
    # synthetic supply-chain task.
    feature_scale: float = 0.5

    parameter_map: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.parameter_map = []
        idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                self.parameter_map.append(ParameterInfo(idx, layer, q, "RY"))
                idx += 1
            for q in range(self.n_qubits):
                self.parameter_map.append(ParameterInfo(idx, layer, q, "RZ"))
                idx += 1
        self._n_params = idx

    @property
    def num_parameters(self) -> int:
        return self._n_params

    def initial_weights(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, self.num_parameters)

    # ------------------------------------------------------------------

    def _encode(self, state: np.ndarray, X: np.ndarray) -> np.ndarray:
        for q in range(self.n_qubits):
            col = X[:, q % X.shape[1]] * self.feature_scale
            state = apply_1q(state, q, _rx(col))
        return state

    def forward_full(self, theta: np.ndarray, X: np.ndarray):
        """Run the circuit, returning the final statevector and the
        intermediate statevector captured right after each layer's
        entangling block (used for the entanglement-weight computation).
        """
        batch = X.shape[0]
        dim = 2 ** self.n_qubits
        state = np.zeros((batch, dim), dtype=complex)
        state[:, 0] = 1.0

        post_entangler_states = []
        p = 0
        for layer in range(self.n_layers):
            state = self._encode(state, X)

            for q in range(self.n_qubits):
                state = apply_1q(state, q, _ry(np.full(batch, theta[p])))
                p += 1

            for q in range(self.n_qubits - 1):
                state = apply_cnot(state, q, q + 1)
            state = apply_cnot(state, self.n_qubits - 1, 0)

            post_entangler_states.append(state.copy())

            for q in range(self.n_qubits):
                state = apply_1q(state, q, _rz(np.full(batch, theta[p])))
                p += 1

        return state, post_entangler_states

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return <Z_0> expectation value per sample, shape (batch,)."""
        state, _ = self.forward_full(theta, X)
        dim = state.shape[1]
        idx = np.arange(dim)
        z0 = 1 - 2 * ((idx >> 0) & 1)  # +1 if bit0==0 else -1
        probs = np.abs(state) ** 2
        return probs @ z0

    def predict_proba(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        expz = self.forward(theta, X)
        return np.clip((1 + expz) / 2, 1e-6, 1 - 1e-6)
