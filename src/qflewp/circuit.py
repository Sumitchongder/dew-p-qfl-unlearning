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
index (i.e. qubit 0 is the least-significant bit). This is Qiskit's own
statevector-ordering convention, and is the convention used consistently by
apply_1q / apply_cnot / reduced_density_matrix below.

Two interchangeable backends implement circuit evolution:

  - `backend="qiskit"` (default): builds an actual `qiskit.QuantumCircuit`
    per sample and evolves it with `qiskit.quantum_info.Statevector`, i.e.
    genuine Qiskit state-vector simulation, matching the manuscript's
    statement that the pipeline is implemented in Qiskit.
  - `backend="numpy"`: a hand-rolled batched state-vector simulator that
    applies the same gate sequence directly as dense matrix contractions
    over the whole sample batch at once. This produces numerically
    identical statevectors to the Qiskit backend (see
    `tests/test_qflewp.py::test_qiskit_numpy_backend_equivalence`), but is
    substantially faster because it evolves an entire mini-batch in one
    vectorized pass rather than one `QuantumCircuit` per sample, which
    matters for the many thousands of circuit evaluations used by
    parameter-shift training/QFIM estimation. It exists purely as a
    validated performance backend for large sweeps and is not a different
    method -- it is the same circuit, simulated a different way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QISKIT_AVAILABLE = False


# ----------------------------------------------------------------------
# Single-qubit rotation matrices (Qiskit convention) -- used by the numpy
# backend only; the qiskit backend uses QuantumCircuit.rx/ry/rz directly.
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


def reduced_density_matrix_2q(state: np.ndarray, qubit_a: int, qubit_b: int) -> np.ndarray:
    """Average (over the batch) two-qubit reduced density matrix for the
    pair (qubit_a, qubit_b), obtained by tracing out every other qubit.

    Basis ordering within the returned 4x4 matrix is
    |00>, |01>, |10>, |11> in (qubit_a, qubit_b) order. Indices of the full
    statevector are grouped by their (qubit_a, qubit_b) bit values; because
    those two bits are independent of all remaining ("traced-out") bits,
    filtering the full index array by a fixed (a, b) value preserves the
    relative ordering of the traced-out bits identically across all four
    groups, so the four column blocks below are already aligned term-by-term
    and can be paired directly without extra bookkeeping.
    """
    dim = state.shape[1]
    idx = np.arange(dim)
    bit_a = (idx >> qubit_a) & 1
    bit_b = (idx >> qubit_b) & 1
    ab = bit_a * 2 + bit_b

    cols = [state[:, idx[ab == k]] for k in range(4)]

    rho = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            rho[i, j] = np.mean(np.sum(cols[i] * np.conj(cols[j]), axis=1))
    return rho


def wootters_concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence (Wootters, PRL 80, 2245 (1998)) of a two-qubit
    density matrix `rho` (4x4, basis order |00>,|01>,|10>,|11>).
    """
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ np.conj(rho) @ yy
    r_mat = rho @ rho_tilde

    eigvals = np.linalg.eigvals(r_mat)
    eigvals = np.sqrt(np.clip(eigvals.real, 0.0, None))
    eigvals = np.sort(eigvals)[::-1]

    c = eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]
    return float(max(0.0, c))


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
    # "qiskit" (default): genuine per-sample QuantumCircuit + Statevector
    # simulation. "numpy": validated-equivalent batched simulator, kept as
    # a fast path for large sweeps -- see module docstring.
    backend: str = "qiskit"

    parameter_map: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.backend == "qiskit" and not _QISKIT_AVAILABLE:
            raise ImportError(
                "backend='qiskit' requires the `qiskit` package "
                "(pip install qiskit). Install it, or construct VQC with "
                "backend='numpy' to use the validated-equivalent numpy "
                "simulator instead."
            )
        if self.backend not in ("qiskit", "numpy"):
            raise ValueError(f"backend must be 'qiskit' or 'numpy', got {self.backend!r}")

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

        Dispatches to the Qiskit or numpy backend per `self.backend`; both
        return identical (batch, 2**n_qubits) complex arrays, up to
        floating-point noise (see the backend-equivalence test).
        """
        if self.backend == "qiskit":
            return self._forward_full_qiskit(theta, X)
        return self._forward_full_numpy(theta, X)

    def _build_qiskit_circuit(self, theta: np.ndarray, x_row: np.ndarray, stop_layer: int | None = None):
        """Build the Qiskit circuit for one sample `x_row`, up to and
        including the ring-CNOT block of `stop_layer` (post-entangler
        snapshot) if given, otherwise the full L-layer circuit."""
        qc = QuantumCircuit(self.n_qubits)
        p = 0
        n_layers = self.n_layers if stop_layer is None else stop_layer + 1
        for layer in range(n_layers):
            for q in range(self.n_qubits):
                qc.rx(float(x_row[q % x_row.shape[0]]) * self.feature_scale, q)
            for q in range(self.n_qubits):
                qc.ry(float(theta[p]), q)
                p += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)

            if stop_layer is not None and layer == stop_layer:
                return qc

            for q in range(self.n_qubits):
                qc.rz(float(theta[p]), q)
                p += 1
        return qc

    def _forward_full_qiskit(self, theta: np.ndarray, X: np.ndarray):
        batch = X.shape[0]
        dim = 2 ** self.n_qubits

        final_state = np.zeros((batch, dim), dtype=complex)
        post_entangler_states = [np.zeros((batch, dim), dtype=complex) for _ in range(self.n_layers)]

        for i in range(batch):
            x_row = X[i]
            qc_final = self._build_qiskit_circuit(theta, x_row)
            final_state[i] = Statevector.from_instruction(qc_final).data

            for layer in range(self.n_layers):
                qc_partial = self._build_qiskit_circuit(theta, x_row, stop_layer=layer)
                post_entangler_states[layer][i] = Statevector.from_instruction(qc_partial).data

        return final_state, post_entangler_states

    def _forward_full_numpy(self, theta: np.ndarray, X: np.ndarray):
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
