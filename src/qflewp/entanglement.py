"""
Per-gate entanglement weight w_ent(k).

Definition used here (von Neumann entropy variant from the paper's figure
list): for the parameter k that sits on qubit q in layer l, w_ent(k) is the
single-qubit von Neumann entropy of qubit q's reduced state, measured
immediately after layer l's ring-CNOT entangling block (the point in the
circuit where that layer's entanglement has just been generated), averaged
over a representative batch of client data and bounded in [0, 1] since a
single qubit's maximum entropy is 1 bit.

Both the RY parameter (applied before the entangler) and the RZ parameter
(applied after the entangler) belonging to qubit q in layer l are assigned
the same weight, because both gates control the amplitude/phase that feeds
into (or was just shaped by) that layer's entangling operation on qubit q.

This differs from the flat, single-scalar-for-every-parameter placeholder
in the original src/quantum/entanglement.py, which could not distinguish
gate importance and therefore made the "entanglement-weighted" part of EWP
vacuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qflewp.circuit import VQC, reduced_density_matrix, von_neumann_entropy


@dataclass
class EntanglementResult:
    gate_weights: np.ndarray          # shape (n_params,)
    per_layer_qubit: np.ndarray       # shape (n_layers, n_qubits)
    average_weight: float
    maximum_weight: float
    minimum_weight: float


class EntanglementAnalyzer:
    def __init__(self, vqc: VQC):
        self.vqc = vqc

    def compute(self, theta: np.ndarray, X: np.ndarray) -> EntanglementResult:
        _, post_entangler_states = self.vqc.forward_full(theta, X)

        per_layer_qubit = np.zeros((self.vqc.n_layers, self.vqc.n_qubits))
        for layer, state in enumerate(post_entangler_states):
            for q in range(self.vqc.n_qubits):
                rho = reduced_density_matrix(state, q)
                per_layer_qubit[layer, q] = von_neumann_entropy(rho)

        gate_weights = np.zeros(self.vqc.num_parameters)
        for p in self.vqc.parameter_map:
            gate_weights[p.index] = per_layer_qubit[p.layer, p.qubit]

        return EntanglementResult(
            gate_weights=gate_weights,
            per_layer_qubit=per_layer_qubit,
            average_weight=float(gate_weights.mean()),
            maximum_weight=float(gate_weights.max()),
            minimum_weight=float(gate_weights.min()),
        )
