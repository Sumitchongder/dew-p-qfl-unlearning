"""
Diagonal Quantum Fisher Information Matrix (QFIM) estimation via the
parameter-shift / state-overlap rule, matching the paper's Eq. (3):

    F_kk(x) = 1 - |<psi(theta; x) | psi(theta + pi*e_k; x)>|^2

(Earlier versions of this module divided by an extra factor of 4, following
the "/4" convention sometimes used for the quantum geometric tensor when
generators are normalized to eigenvalues +-1/2 -- see Stokes et al.,
"Quantum Natural Gradient", Quantum 4, 269 (2020). That convention disagreed
with the paper's stated Eq. (3), which omits the factor of 4. The two
conventions only rescale every F_kk by the same constant, so the pruning
*ranking* -- and therefore every reported accuracy/forgetting/AUROC number,
which depends only on which parameters end up below the fraction-based
threshold -- is unaffected; this change exists purely so the code matches
the manuscript's formula exactly.)

This is exact (not a finite-difference approximation) for any parameter
whose generator is a Pauli operator with eigenvalues +-1, which holds for
every RY/RZ gate in the ansatz.

Crucially, because the circuit encodes data via re-uploading, the resulting
statevector |psi(theta; x)> depends on the input x. Averaging F_kk(x) over
a target client j's samples gives a genuinely client-conditioned diagonal
QFIM entry F_kk^(j), which is what the EWP pruning score requires — the
previous implementation (cos^2(theta)) had no such dependence.

Evaluation cost: `estimate()` computes a single reference-state batch
(the unshifted forward_full(theta, X)) once, then one shifted-state batch
per trainable parameter (forward_full(theta + pi*e_k, X)), reusing the
cached reference state for the overlap rather than re-preparing it per
parameter. For a batch of B inputs and P trainable parameters this is
B*(P+1) total state preparations, i.e. P circuit evaluations per input
plus the one shared reference pass -- not 2P. This matches the paper's
own Table 9 QFIM circuit-evaluation counts (= P, e.g. 24 for P=24 at
n_qubits=4), not 2P.
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
            f_kk_per_sample = 1 - fidelity  # Eq. (3): no /4 factor
            diag[k] = np.mean(f_kk_per_sample)

        return diag + self.epsilon
