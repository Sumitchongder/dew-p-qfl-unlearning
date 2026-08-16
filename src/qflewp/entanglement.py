"""
Per-gate/per-layer entanglement weight w_ent(k).

Primary definition (matches the paper's Eq. (4), Section 4.3): for a
trainable parameter k belonging to layer l, w_ent(k) is the mean pairwise
Wootters concurrence [Wootters, PRL 80, 2245 (1998)] across the CNOT-coupled
qubit pairs of that layer, evaluated on the encoded circuit state
immediately after layer l's ring-CNOT entangling block, averaged over a
representative batch of client data. Because this is a mean over *all*
CNOT-coupled pairs in the layer, it is a single scalar per layer, shared by
every trainable parameter (both RY and RZ, on every qubit) in that layer.

Ablation definition (paper Appendix A.2, "von Neumann entropy ablation";
also the quantity previously shown in Figure 8b): the mean single-qubit
von Neumann entropy of qubit q's reduced state after the same entangling
block, which *is* qubit-specific rather than a single per-layer scalar.
Select it via `EntanglementAnalyzer(vqc, method="von_neumann_entropy")`.

The two definitions are reported as strongly correlated in the paper and
the choice does not qualitatively change the pruning conclusions, but only
one of them can be the *primary* score reported in Eq. (5)/Table 3-9;
`method="concurrence"` is the default so the code matches the paper's
main-text method rather than the entropy variant it originally computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qflewp.circuit import (
    VQC,
    reduced_density_matrix,
    von_neumann_entropy,
    reduced_density_matrix_2q,
    wootters_concurrence,
)

_VALID_METHODS = ("concurrence", "von_neumann_entropy")


@dataclass
class EntanglementResult:
    gate_weights: np.ndarray          # shape (n_params,)
    per_layer_qubit: np.ndarray       # shape (n_layers, n_qubits)
    average_weight: float
    maximum_weight: float
    minimum_weight: float
    method: str                       # "concurrence" (default) or "von_neumann_entropy"


class EntanglementAnalyzer:
    def __init__(self, vqc: VQC, method: str = "concurrence"):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.vqc = vqc
        self.method = method

    def _ring_pairs(self) -> list:
        """CNOT-coupled qubit pairs for one layer's ring-CNOT block,
        matching VQC.forward_full: (0,1), (1,2), ..., (n_qubits-2,
        n_qubits-1), (n_qubits-1, 0). Identical for every layer since the
        ring topology does not change across layers.
        """
        n = self.vqc.n_qubits
        pairs = [(q, q + 1) for q in range(n - 1)]
        pairs.append((n - 1, 0))
        return pairs

    def compute(self, theta: np.ndarray, X: np.ndarray) -> EntanglementResult:
        _, post_entangler_states = self.vqc.forward_full(theta, X)
        n_layers, n_qubits = self.vqc.n_layers, self.vqc.n_qubits
        per_layer_qubit = np.zeros((n_layers, n_qubits))

        if self.method == "concurrence":
            pairs = self._ring_pairs()
            for layer, state in enumerate(post_entangler_states):
                pair_concurrences = []
                for (a, b) in pairs:
                    rho_ab = reduced_density_matrix_2q(state, a, b)
                    pair_concurrences.append(wootters_concurrence(rho_ab))
                # Eq. (4): mean pairwise concurrence over all CNOT-coupled
                # pairs in the layer -- one scalar shared by every parameter
                # in that layer (both RY and RZ, every qubit).
                per_layer_qubit[layer, :] = float(np.mean(pair_concurrences))
        else:  # "von_neumann_entropy" ablation (Appendix A.2)
            for layer, state in enumerate(post_entangler_states):
                for q in range(n_qubits):
                    rho_q = reduced_density_matrix(state, q)
                    per_layer_qubit[layer, q] = von_neumann_entropy(rho_q)

        gate_weights = np.zeros(self.vqc.num_parameters)
        for p in self.vqc.parameter_map:
            gate_weights[p.index] = per_layer_qubit[p.layer, p.qubit]

        return EntanglementResult(
            gate_weights=gate_weights,
            per_layer_qubit=per_layer_qubit,
            average_weight=float(gate_weights.mean()),
            maximum_weight=float(gate_weights.max()),
            minimum_weight=float(gate_weights.min()),
            method=self.method,
        )
