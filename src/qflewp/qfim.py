"""
Diagonal Quantum Fisher Information Matrix (QFIM) estimation via the
parameter-shift / state-overlap rule (Stokes et al., "Quantum Natural
Gradient", Quantum 4, 269 (2020)):

    F_kk(x) = (1 - |<psi(theta; x) | psi(theta + pi*e_k; x)>|^2) / 4

This is exact (not a finite-difference approximation) for any parameter
whose generator is a Pauli operator with eigenvalues +-1/2, which holds for
every RY/RZ gate in the ansatz.

Crucially, because the circuit encodes data via re-uploading, the resulting
statevector |psi(theta; x)> depends on the input x. Averaging F_kk(x) over
a target client j's samples gives a genuinely client-conditioned diagonal
QFIM entry F_kk^(j), which is what the EWP pruning score requires — the
previous implementation (cos^2(theta)) had no such dependence.
"""

from __future__ import annotations

import numpy as np

from src.qflewp.circuit import VQC


class DiagonalQFIM:
    def __init__(self, vqc: VQC, epsilon: float = 1e-8):
        self.vqc = vqc
        self.epsilon = epsilon

    def estimate(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return F_kk averaged over the samples in X, shape (n_params,)."""
        n_params = self.vqc.num_parameters
        base_state, _ = self.vqc.forward_full(theta, X)  # (batch, dim)

        diag = np.zeros(n_params)
        for k in range(n_params):
            shift = np.zeros(n_params)
            shift[k] = np.pi
            shifted_state, _ = self.vqc.forward_full(theta + shift, X)

            overlap = np.sum(np.conj(base_state) * shifted_state, axis=1)
            fidelity = np.abs(overlap) ** 2
            f_kk_per_sample = (1 - fidelity) / 4
            diag[k] = np.mean(f_kk_per_sample)

        return diag + self.epsilon
