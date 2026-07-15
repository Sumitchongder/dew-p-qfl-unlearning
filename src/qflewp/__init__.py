"""
qflewp -- Dynamic Entanglement-Weighted Pruning for Quantum Federated
Unlearning.

Public API re-exports for convenient importing, e.g.:

    from src.qflewp import VQC, generate_federated_dataset, fedavg_train
"""

from src.qflewp.circuit import VQC
from src.qflewp.data import generate_federated_dataset, ClientDataset, pooled_test_set
from src.qflewp.federated import fedavg_train, local_train, evaluate, bce_loss
from src.qflewp.qfim import DiagonalQFIM
from src.qflewp.entanglement import EntanglementAnalyzer, EntanglementResult
from src.qflewp.pruning import (
    ewp_pruning, fisher_only_pruning, entanglement_only_pruning, random_pruning,
    fine_tune_only, recovery_fine_tune, full_retrain_oracle, PruneResult,
)
from src.qflewp.evaluate import (
    utility_accuracy, utility_auroc, retrain_distance,
    membership_inference_advantage, forgetting_score, evaluate_method,
)

__version__ = "1.0.0"

__all__ = [
    "VQC", "generate_federated_dataset", "ClientDataset", "pooled_test_set",
    "fedavg_train", "local_train", "evaluate", "bce_loss",
    "DiagonalQFIM", "EntanglementAnalyzer", "EntanglementResult",
    "ewp_pruning", "fisher_only_pruning", "entanglement_only_pruning",
    "random_pruning", "fine_tune_only", "recovery_fine_tune", "full_retrain_oracle",
    "PruneResult", "utility_accuracy", "utility_auroc", "retrain_distance",
    "membership_inference_advantage", "forgetting_score", "evaluate_method",
]
