from __future__ import annotations

import numpy as np

from src.qflewp.circuit import VQC


class DiagonalQFIM:
    def __init__(self, vqc: VQC, epsilon: float = 1e-8):
        self.vqc = vqc
        self.epsilon = epsilon

    def estimate(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        
        n_params = self.vqc.num_parameters
        base_state, _ = self.vqc.forward_full(theta, X)  # (batch, dim)

        diag = np.zeros(n_params)
        for k in range(n_params):
            shift = np.zeros(n_params)
            shift[k] = np.pi
            shifted_state, _ = self.vqc.forward_full(theta + shift, X)

            overlap = np.sum(np.conj(base_state) * shifted_state, axis=1)
            fidelity = np.abs(overlap) ** 2
            f_kk_per_sample = 1 - fidelity  # Eq. (3): no /4 factor
            diag[k] = np.mean(f_kk_per_sample)

        return diag + self.epsilon
