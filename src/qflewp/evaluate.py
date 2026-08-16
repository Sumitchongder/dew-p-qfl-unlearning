"""
Evaluation suite (Table 3 in the paper): utility, forgetting / membership
inference, and retrain distance — computed from the *actual* trained model's
predictions, never from synthetic random numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.qflewp.circuit import VQC


def utility_accuracy(vqc: VQC, theta, X, y) -> float:
    prob = vqc.predict_proba(theta, X)
    pred = (prob >= 0.5).astype(int)
    return float(np.mean(pred == y))


def utility_auroc(vqc: VQC, theta, X, y) -> float:
    prob = vqc.predict_proba(theta, X)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def retrain_distance(theta_unlearned, theta_oracle) -> float:
    """L2 distance in parameter space to the full-retrain oracle -- the
    standard closeness-to-exact-unlearning proxy."""
    return float(np.linalg.norm(theta_unlearned - theta_oracle))


def membership_inference_advantage(vqc: VQC, theta, forgotten_client, held_out_clients, seed=0):
    """Shadow-model membership-inference attack (Shokri et al. 2017 style;
    Appendix D), matching the paper: a logistic-regression attacker is
    trained on the target model's output confidence to distinguish (a) the
    forgotten client's *training* samples (members) from (b) samples from
    clients that were genuinely never trained on (non-members: retained
    clients' test splits).

    The attacker's single input feature is confidence = |p - 0.5| * 2,
    exactly as described in Appendix D. We fit a one-dimensional
    `sklearn.linear_model.LogisticRegression` on this feature and evaluate
    it on a held-out split of the same member/non-member pool, then report
    the AUC and membership advantage = 2*(attack accuracy - 0.5) of that
    fitted attacker, i.e. Appendix D's "logistic-regression shadow-model
    attacker" is instantiated directly rather than approximated by ranking
    the raw confidence score.

    Advantage = 2*(AUC - 0.5): 0 means the attacker cannot distinguish
    members from non-members (good forgetting); 1 means perfect membership
    leakage.
    """
    rng = np.random.default_rng(seed)

    member_X = forgotten_client.X_train
    non_member_X = np.concatenate([c.X_test for c in held_out_clients], axis=0)

    n = min(len(member_X), len(non_member_X))
    if n < 4:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    idx_m = rng.choice(len(member_X), size=n, replace=False)
    idx_n = rng.choice(len(non_member_X), size=n, replace=False)

    member_conf = np.abs(vqc.predict_proba(theta, member_X[idx_m]) - 0.5) * 2
    non_member_conf = np.abs(vqc.predict_proba(theta, non_member_X[idx_n]) - 0.5) * 2

    scores = np.concatenate([member_conf, non_member_conf]).reshape(-1, 1)
    labels = np.concatenate([np.ones(n), np.zeros(n)])  # 1 = member

    if len(np.unique(labels)) < 2:
        return {"advantage": 0.0, "attack_auc": 0.5, "roc": None}

    # Shuffle, then split into an attacker-training half and an
    # attacker-evaluation half (Shokri et al. shadow-model protocol:
    # the attacker is *fit*, then evaluated on held-out members/non-members).
    order = rng.permutation(len(labels))
    scores, labels = scores[order], labels[order]
    split = len(labels) // 2
    fit_X, fit_y = scores[:split], labels[:split]
    eval_X, eval_y = scores[split:], labels[split:]

    if len(np.unique(fit_y)) < 2 or len(np.unique(eval_y)) < 2:
        # Degenerate small-sample split: fall back to fitting and
        # evaluating on the full pool rather than failing the metric.
        fit_X, fit_y, eval_X, eval_y = scores, labels, scores, labels

    attacker = LogisticRegression()
    attacker.fit(fit_X, fit_y)
    attack_scores = attacker.predict_proba(eval_X)[:, 1]

    auc = roc_auc_score(eval_y, attack_scores)
    fpr, tpr, _ = roc_curve(eval_y, attack_scores)
    advantage = float(2 * (auc - 0.5))
    return {"advantage": advantage, "attack_auc": float(auc), "roc": (fpr.tolist(), tpr.tolist())}


def forgetting_score(membership_result: dict) -> float:
    """Forgetting score = 1 - |membership advantage|, rescaled to [0, 1]
    where 1 = perfect forgetting (attacker at chance level).

    This is the metric that actually produced every forgetting-score
    number in results/ (Tables 3-9) and Figure 4b: e.g.
    1 - |-0.322| = 0.678 (QFL-EWP) and 1 - |-0.204| = 0.796 (full retrain),
    matching Table 3 exactly. It is reported here under the key
    `"forgetting_score"` for that reason -- changing this formula would
    silently invalidate the frozen results/ artifacts and every reported
    p-value in Table 4, so it is left as-is.

    Note on the paper's wording: Section 5.4's prose describes the
    forgetting score as an "output-distribution divergence between the
    unlearned model and theta_retrain on the forgotten client's data",
    which is closer to `output_divergence_forgetting_score` below than to
    this formula, while Appendix D's *operational* definition and Figure
    4b's axis label ("Forgetting score (1 - |MI advantage|)") both match
    this formula exactly. `evaluate_method` reports both metrics under
    separate keys so a reader can see the actual relationship rather than
    only one name resolving to two different formulas depending on which
    part of the paper they're reading.
    """
    return float(1 - abs(membership_result["advantage"]))


def output_divergence_forgetting_score(vqc: VQC, theta_unlearned, theta_oracle, X_forgotten) -> float:
    """Section 5.4's literal "output-distribution divergence" forgetting
    metric: the mean absolute difference between the unlearned model's and
    the oracle model's predicted risk probabilities on the forgotten
    client's held-out instances,

        F_forget = mean_i |p_unlearned(x_i) - p_oracle(x_i)|,

    evaluated on `X_forgotten` (the forgotten client's held-out/test
    split). Lower is better here (0 = the unlearned model reproduces the
    oracle's predictions on the forgotten client's data exactly), which is
    the opposite direction from `forgetting_score` above (where higher is
    better) -- do not average or compare the two directly.

    This metric is *not* the one used to produce the results/ tables (see
    `forgetting_score`'s docstring); it is provided so the code also
    directly implements the metric as literally worded in the paper's main
    text, in addition to the one that actually generated the reported
    numbers.
    """
    if len(X_forgotten) == 0:
        return float("nan")
    p_unlearned = vqc.predict_proba(theta_unlearned, X_forgotten)
    p_oracle = vqc.predict_proba(theta_oracle, X_forgotten)
    return float(np.mean(np.abs(p_unlearned - p_oracle)))


def evaluate_method(vqc, theta, theta_oracle, retained_clients, forgotten_client, seed=0):
    X_ret, y_ret = np.concatenate([c.X_test for c in retained_clients]), \
        np.concatenate([c.y_test for c in retained_clients])

    mi = membership_inference_advantage(vqc, theta, forgotten_client, retained_clients, seed=seed)

    return {
        "utility_accuracy": utility_accuracy(vqc, theta, X_ret, y_ret),
        "utility_auroc": utility_auroc(vqc, theta, X_ret, y_ret),
        # Reported forgetting score (Tables 3-9, Figure 4b): 1 - |MI advantage|.
        "forgetting_score": forgetting_score(mi),
        # Section 5.4's literal output-distribution-divergence forgetting
        # metric; not used in results/, provided for completeness and
        # future audits (see docstring above). Lower is better.
        "forgetting_output_divergence": output_divergence_forgetting_score(
            vqc, theta, theta_oracle, forgotten_client.X_test
        ),
        "membership_advantage": mi["advantage"],
        "attack_auc": mi["attack_auc"],
        "retrain_distance": retrain_distance(theta, theta_oracle),
        "roc": mi["roc"],
    }
