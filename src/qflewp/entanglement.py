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
    method: str                      


class EntanglementAnalyzer:
    def __init__(self, vqc: VQC, method: str = "concurrence"):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.vqc = vqc
        self.method = method

    def _ring_pairs(self) -> list:
   
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
                
                per_layer_qubit[layer, :] = float(np.mean(pair_concurrences))
        else:  
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
