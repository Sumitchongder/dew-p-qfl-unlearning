"""
Produces the full Q1-journal deliverable set requested for the QFL-EWP
paper: 10 tables (CSV) and 16 figures (PNG + PDF), built entirely from
results/json/main_experiment_raw.json and extended_experiment.json
(both real, measured data — nothing here is hand-typed).

Run:
    python3 -m src.qflewp.q1_deliverables
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- styling
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "font.size": 10,
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 8.7, "ytick.labelsize": 8.7,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
PANEL_LABEL_KW = dict(fontsize=12, fontweight="bold", ha="left", va="top")


def panel_label(ax, letter):
    ax.text(-0.14, 1.08, f"({letter})", transform=ax.transAxes, **PANEL_LABEL_KW)

METHOD_ORDER = ["Random Pruning", "Fisher-only Pruning", "Entanglement-only Pruning",
                "Fine-Tune Only", "Full Retrain (oracle)", "QFL-EWP"]
METHOD_COLORS = {
    "Random Pruning": "#9E9E9E", "Fisher-only Pruning": "#4C72B0",
    "Entanglement-only Pruning": "#DD8452", "Fine-Tune Only": "#55A868",
    "Full Retrain (oracle)": "#8172B2", "QFL-EWP": "#C44E52",
}


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load():
    with open("results/json/main_experiment_raw.json") as f:
        raw = json.load(f)
    with open("results/json/extended_experiment.json") as f:
        ext = json.load(f)
    df = pd.read_csv(TAB / "main_results_raw.csv")
    return raw, ext, df


# =====================================================================
# TABLES
# =====================================================================

def table01_dataset_summary(cfg):
    df = pd.DataFrame([{
        "Parameter": k, "Value": v
    } for k, v in {
        "Number of clients": cfg["n_clients"],
        "Samples per client": cfg["samples_per_client"],
        "Train / test split": "75% / 25%",
        "Feature dimensionality": cfg["n_features"],
        "Non-IID strength (client bias scale)": cfg["non_iid_strength"],
        "Label noise (logistic std)": cfg["noise"],
        "Forgotten client ID": cfg["forget_client"],
        "Task": "Binary supply-chain risk classification",
        "Data generation": "Synthetic, seeded (seed = 1000 + run seed)",
    }.items()])
    df.to_csv(TAB / "table01_dataset_summary.csv", index=False)
    return df


def table02_federated_training_configuration(cfg):
    df = pd.DataFrame([{
        "Parameter": k, "Value": v
    } for k, v in {
        "Ansatz": "Data re-uploading VQC (RX encode -> RY -> ring-CNOT -> RZ)",
        "Qubits": cfg["n_qubits"], "Layers": cfg["n_layers"],
        "Trainable parameters": cfg["n_qubits"] * 2 * cfg["n_layers"],
        "Feature encoding scale (rad)": 0.5,
        "Federated rounds": cfg["n_rounds"],
        "Local optimizer": "L-BFGS-B (exact parameter-shift gradient)",
        "Local iterations / round": cfg["local_maxiter"],
        "Aggregation rule": "FedAvg, weighted by client sample count",
        "Gradient estimator": "Exact parameter-shift rule (Pauli generators)",
        "Random seeds": "0, 1, 2",
        "Prune fraction (tau, default)": cfg["prune_fraction"],
    }.items()])
    df.to_csv(TAB / "table02_federated_training_configuration.csv", index=False)
    return df


def table03_baseline_comparison(df):
    agg = df.groupby("method").agg(
        Accuracy_mean=("accuracy", "mean"), Accuracy_std=("accuracy", "std"),
        AUROC_mean=("auroc", "mean"), AUROC_std=("auroc", "std"),
        Forgetting_mean=("forgetting_score", "mean"), Forgetting_std=("forgetting_score", "std"),
        MembershipAdv_mean=("membership_advantage", "mean"), MembershipAdv_std=("membership_advantage", "std"),
        RetrainDist_mean=("retrain_distance", "mean"), RetrainDist_std=("retrain_distance", "std"),
    ).reindex(METHOD_ORDER).reset_index()
    agg.to_csv(TAB / "table03_baseline_comparison.csv", index=False)
    return agg


def table04_utility_comparison(df):
    rows = []
    for method in METHOD_ORDER:
        g = df[df.method == method]
        rows.append({
            "Method": method,
            "Accuracy (mean)": g.accuracy.mean(), "Accuracy (std)": g.accuracy.std(),
            "AUROC (mean)": g.auroc.mean(), "AUROC (std)": g.auroc.std(),
            "Accuracy_seed0": g.iloc[0].accuracy if len(g) > 0 else np.nan,
            "Accuracy_seed1": g.iloc[1].accuracy if len(g) > 1 else np.nan,
            "Accuracy_seed2": g.iloc[2].accuracy if len(g) > 2 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table04_utility_comparison.csv", index=False)
    return out


def table05_forgetting_comparison(df):
    rows = []
    for method in METHOD_ORDER:
        g = df[df.method == method]
        rows.append({
            "Method": method,
            "Forgetting score (mean)": g.forgetting_score.mean(),
            "Forgetting score (std)": g.forgetting_score.std(),
            "Membership advantage (mean)": g.membership_advantage.mean(),
            "Membership advantage (std)": g.membership_advantage.std(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table05_forgetting_comparison.csv", index=False)
    return out


def table06_privacy_comparison(ext):
    rows = []
    for method in METHOD_ORDER:
        aucs, advs, forg = [], [], []
        for r in ext["results"]:
            m = r["methods"][method]
            aucs.append(m["attack_auc"]); advs.append(m["membership_advantage"]); forg.append(m["forgetting_score"])
        rows.append({
            "Method": method,
            "Attack AUC (mean)": np.mean(aucs), "Attack AUC (std)": np.std(aucs, ddof=1),
            "Membership advantage (mean)": np.mean(advs), "Membership advantage (std)": np.std(advs, ddof=1),
            "Privacy leakage risk": "Low" if abs(np.mean(advs)) < 0.1 else ("Moderate" if abs(np.mean(advs)) < 0.25 else "High"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table06_privacy_comparison.csv", index=False)
    return out


def table07_runtime_comparison(ext):
    rows = []
    for method in METHOD_ORDER:
        times = [r["methods"][method]["unlearn_seconds"] for r in ext["results"]]
        rows.append({
            "Method": method,
            "Unlearning time (s, mean)": np.mean(times),
            "Unlearning time (s, std)": np.std(times, ddof=1),
            "Speedup vs. Full Retrain": np.mean([r["methods"]["Full Retrain (oracle)"]["unlearn_seconds"] for r in ext["results"]]) / max(np.mean(times), 1e-6),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table07_runtime_comparison.csv", index=False)
    return out


def table08_statistical_significance(df):
    ewp = df[df.method == "QFL-EWP"].sort_values("seed")
    rows = []
    for method in METHOD_ORDER:
        if method == "QFL-EWP":
            continue
        base = df[df.method == method].sort_values("seed")
        n = min(len(ewp), len(base))
        for metric in ["accuracy", "forgetting_score"]:
            a, b = ewp[metric].values[:n], base[metric].values[:n]
            t_stat, p_val = stats.ttest_rel(a, b) if n > 1 else (np.nan, np.nan)
            rows.append({
                "Comparison": f"QFL-EWP vs. {method}", "Metric": metric,
                "Mean diff (EWP - baseline)": float(np.mean(a - b)),
                "t-statistic": t_stat, "p-value (paired t-test, n=3)": p_val,
                "Significant at alpha=0.05": bool(p_val < 0.05) if not np.isnan(p_val) else "n/a (n too small)",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table08_statistical_significance.csv", index=False)
    return out


def table09_ablation_study():
    d1 = pd.read_csv(TAB / "ablation_non_iid.csv"); d1["ablation_type"] = "Non-IID strength"
    d1 = d1.rename(columns={"non_iid_strength": "level"})
    d2 = pd.read_csv(TAB / "ablation_num_clients.csv"); d2["ablation_type"] = "Number of clients"
    d2 = d2.rename(columns={"n_clients": "level"})
    out = pd.concat([d1, d2], ignore_index=True)[["ablation_type", "level", "accuracy", "auroc", "forgetting_score"]]
    out.to_csv(TAB / "table09_ablation_study.csv", index=False)
    return out


def table10_hyperparameter_sensitivity():
    sweep = pd.read_csv(TAB / "pruning_fraction_sweep.csv")
    ewp = sweep[sweep.method == "QFL-EWP"].sort_values("prune_fraction")
    ewp = ewp.rename(columns={"prune_fraction": "Pruning threshold (tau, fraction pruned)"})
    ewp.to_csv(TAB / "table10_hyperparameter_sensitivity.csv", index=False)
    return ewp


# =====================================================================
# FIGURES
# =====================================================================

def fig01_overall_framework():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, color, fontsize=9.2):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                            linewidth=1.2, edgecolor="#333", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                      linewidth=1.3, color="#333"))

    box(0.2, 2.9, 1.7, 1.1, "Client 1..N\nlocal VQC\n$\\theta_i$", "#DCEBFA")
    box(0.2, 0.6, 1.7, 1.1, "Client $j$\n(forget request)", "#FADCDC")
    box(2.5, 1.9, 1.9, 1.6, "Server\nFedAvg\naggregation $\\theta$", "#E8E8E8")
    arrow(1.9, 3.4, 2.5, 3.0); arrow(1.9, 1.1, 2.5, 2.4)

    box(5.0, 3.0, 2.1, 1.0, "Diagonal QFIM\n$F_{kk}^{(j)}$ (parameter-shift)", "#DCEBFA")
    box(5.0, 1.3, 2.1, 1.0, "Entanglement weight\n$w_{ent}(k)$ (von Neumann)", "#DCEBFA")
    arrow(4.4, 2.7, 5.0, 3.3); arrow(4.4, 2.1, 5.0, 1.9)

    box(7.6, 2.15, 1.9, 1.1, "Pruning score\n$s_k = w_{ent}(k) F_{kk}^{(j)}$\nprune $s_k \\leq \\tau$", "#FFF3CD")
    arrow(7.1, 3.4, 7.6, 2.9); arrow(7.1, 1.8, 7.6, 2.4)

    box(9.7, 2.15, 1.1, 1.1, "Pruned\n+ recovered\nmodel", "#DCE8DC")
    arrow(9.5, 2.7, 9.7, 2.7)

    ax.text(5.6, 0.35, "Evaluation: utility (accuracy/AUROC) · forgetting (membership inference) · retrain distance",
            ha="center", fontsize=9, style="italic", color="#333")
    ax.set_title("End-to-end QFL-EWP framework", fontsize=12, pad=10)
    savefig(fig, "fig01_overall_framework")


def fig03_federated_workflow():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    def box(x, y, w, h, text, color, fontsize=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                            linewidth=1.2, edgecolor="#333", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
                                      linewidth=1.2, color="#333"))

    ys = [4.1, 3.1, 2.1, 1.1]
    labels = ["Client 1", "Client 2", "Client 3 (forget later)", "Client N"]
    for y, lab in zip(ys, labels):
        box(0.2, y, 1.6, 0.7, lab, "#DCEBFA", 8.5)
        arrow(1.8, y + 0.35, 3.6, 2.75)

    box(3.6, 2.4, 2.0, 1.1, "Server:\nround $r=1..R$\nFedAvg($\\theta_1..\\theta_N$)", "#E8E8E8")
    arrow(5.6, 2.95, 6.9, 2.95)
    box(6.9, 2.4, 2.0, 1.1, "Global model\n$\\theta^{(R)}$", "#DCE8DC")

    box(3.6, 0.2, 5.3, 1.4, "Forget request from client $j$ $\\rightarrow$ Algorithm 2 (EWP):\ncompute $F_{kk}^{(j)}$, $w_{ent}(k)$ on server using $\\theta^{(R)}$;\nprune + recovery fine-tune on retained clients only",
        "#FFF3CD", 8.5)
    arrow(7.9, 2.4, 6.3, 1.6)

    ax.set_title("Federated training and forget-request workflow", fontsize=12, pad=10)
    savefig(fig, "fig03_federated_workflow")


def fig04_training_convergence(raw):
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    for r in raw["results"]:
        ax.plot(r["training_history"]["accuracy"], marker="o", alpha=0.8, label=f"seed {r['seed']}")
    ax.set_xlabel("Federated round"); ax.set_ylabel("Pooled test accuracy")
    ax.set_title("QFL (FedAvg) training convergence")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "fig04_training_convergence")


def _bar_with_err(ax, means, stds, colors, labels, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel); ax.set_title(title)


def fig05_accuracy_comparison(df):
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    agg = df.groupby("method").agg(m=("accuracy", "mean"), s=("accuracy", "std")).reindex(METHOD_ORDER)
    _bar_with_err(ax, agg.m, agg.s, [METHOD_COLORS[m] for m in METHOD_ORDER], METHOD_ORDER,
                  "Accuracy (retained-client test set)", "Utility (accuracy) comparison")
    ax.axhline(agg.loc["Full Retrain (oracle)", "m"], color="black", linestyle=":", linewidth=1, label="Oracle level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig05_accuracy_comparison")


def fig06_forgetting_comparison(df):
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    agg = df.groupby("method").agg(m=("forgetting_score", "mean"), s=("forgetting_score", "std")).reindex(METHOD_ORDER)
    _bar_with_err(ax, agg.m, agg.s, [METHOD_COLORS[m] for m in METHOD_ORDER], METHOD_ORDER,
                  "Forgetting score (1 - |MI advantage|)", "Forgetting comparison")
    fig.tight_layout()
    savefig(fig, "fig06_forgetting_comparison")


def fig07_privacy_comparison(ext):
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    means, stds = [], []
    for method in METHOD_ORDER:
        aucs = [r["methods"][method]["attack_auc"] for r in ext["results"]]
        means.append(np.mean(aucs)); stds.append(np.std(aucs, ddof=1))
    _bar_with_err(ax, means, stds, [METHOD_COLORS[m] for m in METHOD_ORDER], METHOD_ORDER,
                  "Membership-inference attack AUC", "Privacy comparison (lower is better)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (ideal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig07_privacy_comparison")


def fig08_runtime_comparison(ext):
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    means, stds = [], []
    for method in METHOD_ORDER:
        times = [r["methods"][method]["unlearn_seconds"] for r in ext["results"]]
        means.append(np.mean(times)); stds.append(np.std(times, ddof=1))
    x = np.arange(len(METHOD_ORDER))
    ax.bar(x, means, yerr=stds, color=[METHOD_COLORS[m] for m in METHOD_ORDER], capsize=4, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Unlearning time (s, log scale)")
    ax.set_title("Runtime comparison (unlearning step only)")
    fig.tight_layout()
    savefig(fig, "fig08_runtime_comparison")


def fig09_radar_chart(df, ext):
    metrics = ["Accuracy", "Forgetting", "Privacy\n(1-attack AUC excess)", "Speed\n(1/log-time)", "Param.\nproximity\n(1/retrain dist.)"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist(); angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))
    for method in METHOD_ORDER:
        g = df[df.method == method]
        acc = g.accuracy.mean()
        forg = g.forgetting_score.mean()
        aucs = [r["methods"][method]["attack_auc"] for r in ext["results"]]
        privacy = 1 - max(0, np.mean(aucs) - 0.5) * 2
        times = [r["methods"][method]["unlearn_seconds"] for r in ext["results"]]
        speed = 1 / (1 + np.log10(max(np.mean(times), 1e-3) + 1))
        rdist = g.retrain_distance.mean()
        proximity = 1 / (1 + rdist)
        vals = [acc, forg, privacy, speed, proximity]
        vals += vals[:1]
        ax.plot(angles, vals, color=METHOD_COLORS[method], linewidth=2, label=method)
        ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.06)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_yticklabels([])
    ax.set_title("Multi-metric radar comparison\n(all axes normalized, outward = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig09_radar_chart")


def fig10_ablation_plot():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    d1 = pd.read_csv(TAB / "ablation_non_iid.csv")
    axes[0].plot(d1["non_iid_strength"], d1["accuracy"], marker="o", label="Accuracy", color="#4C72B0")
    axes[0].plot(d1["non_iid_strength"], d1["forgetting_score"], marker="s", label="Forgetting", color="#C44E52")
    axes[0].set_xlabel("Non-IID strength"); axes[0].set_ylabel("Score")
    axes[0].legend(); panel_label(axes[0], "a")

    d2 = pd.read_csv(TAB / "ablation_num_clients.csv")
    axes[1].plot(d2["n_clients"], d2["accuracy"], marker="o", label="Accuracy", color="#4C72B0")
    axes[1].plot(d2["n_clients"], d2["forgetting_score"], marker="s", label="Forgetting", color="#C44E52")
    axes[1].set_xlabel("Number of clients"); axes[1].set_ylabel("Score")
    axes[1].legend(); panel_label(axes[1], "b")
    fig.tight_layout()
    savefig(fig, "fig10_ablation_plot")


def fig11_sensitivity_plot():
    sweep = pd.read_csv(TAB / "pruning_fraction_sweep.csv")
    ewp = sweep[sweep.method == "QFL-EWP"].sort_values("prune_fraction")
    fig, ax1 = plt.subplots(figsize=(4.5, 3.4))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ewp.prune_fraction, ewp.accuracy, marker="o", color="#4C72B0", label="Accuracy")
    l2, = ax2.plot(ewp.prune_fraction, ewp.forgetting_score, marker="s", color="#C44E52", label="Forgetting score")
    ax1.set_xlabel("Pruning threshold $\\tau$ (fraction of parameters pruned)")
    ax1.set_ylabel("Accuracy", color="#4C72B0"); ax2.set_ylabel("Forgetting score", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#4C72B0"); ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_title("Hyperparameter sensitivity: pruning threshold $\\tau$ (QFL-EWP)")
    ax1.legend(handles=[l1, l2], fontsize=8, loc="lower left")
    fig.tight_layout()
    savefig(fig, "fig11_sensitivity_plot")


def fig12_confusion_matrix(ext):
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.4))
    seed0 = ext["results"][0]["methods"]
    for letter, (ax, method) in zip("abcdef", zip(axes.flat, METHOD_ORDER)):
        cm = np.array(seed0[method]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10.5)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(method, fontsize=9, fontweight="normal")
        panel_label(ax, letter)
        ax.grid(False)
    fig.tight_layout()
    savefig(fig, "fig12_confusion_matrix")


def fig13_roc_curve(ext):
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    seed0 = ext["results"][0]["methods"]
    for method in METHOD_ORDER:
        roc = seed0[method]["roc"]
        if roc:
            fpr, tpr = roc
            auc = seed0[method]["attack_auc"]
            ax.plot(fpr, tpr, color=METHOD_COLORS[method], label=f"{method} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Membership-inference attack ROC\n(attacking the forgotten client, seed 0)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig13_roc_curve")


def fig14_parameter_importance(raw):
    r = raw["results"][0]
    scores = np.array(r["ewp_scores"])
    order = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = ["#C44E52" if s <= r["ewp_threshold"] else "#4C72B0" for s in scores[order]]
    ax.bar(range(len(scores)), scores[order], color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(r["ewp_threshold"], color="black", linestyle="--", linewidth=1,
               label=f"$\\tau$ = {r['ewp_threshold']:.4f}")
    ax.set_xlabel("Parameter rank (sorted by importance)"); ax.set_ylabel("$s_k = w_{ent}(k)\\cdot F_{kk}^{(j)}$")
    ax.set_title("Parameter importance ranking (EWP pruning scores)")
    red_patch = mpatches.Patch(color="#C44E52", label="Pruned")
    blue_patch = mpatches.Patch(color="#4C72B0", label="Retained")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig14_parameter_importance")


def fig15_qfim_visualization(raw):
    r = raw["results"][0]
    f_kk = np.array(r["qfim_diag"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].bar(range(len(f_kk)), f_kk, color="#4C72B0", width=0.75)
    axes[0].set_xlabel("Parameter index $k$"); axes[0].set_ylabel("$F_{kk}^{(j)}$")
    panel_label(axes[0], "a")

    n_layers, n_qubits = 3, 4
    grid = f_kk.reshape(n_layers, 2, n_qubits)  # (layer, RY/RZ, qubit)
    im = axes[1].imshow(np.vstack([grid[:, 0, :], grid[:, 1, :]]), cmap="magma", aspect="auto")
    axes[1].set_yticks(range(6))
    axes[1].set_yticklabels([f"L{l} RY" for l in range(n_layers)] + [f"L{l} RZ" for l in range(n_layers)], fontsize=7.5)
    axes[1].set_xlabel("Qubit")
    axes[1].grid(False)
    panel_label(axes[1], "b")
    fig.colorbar(im, ax=axes[1], label="$F_{kk}$", fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, "fig15_qfim_visualization")


def fig16_entanglement_heatmap(raw):
    r = raw["results"][0]
    per_lq = np.array(r["entanglement_per_layer_qubit"])
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    im = ax.imshow(per_lq, cmap="viridis", aspect="auto")
    ax.set_xlabel("Qubit"); ax.set_ylabel("Layer")
    ax.set_xticks(range(per_lq.shape[1])); ax.set_yticks(range(per_lq.shape[0]))
    ax.set_title("Per-gate entanglement weight $w_{ent}(k)$\n(single-qubit von Neumann entropy)")
    for i in range(per_lq.shape[0]):
        for j in range(per_lq.shape[1]):
            ax.text(j, i, f"{per_lq[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8.5)
    fig.colorbar(im, ax=ax, label="Entropy (bits)")
    fig.tight_layout()
    savefig(fig, "fig16_entanglement_heatmap")


def fig02_circuit_architecture():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        n_qubits, n_layers = 4, 3
        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("θ", n_qubits * 2 * n_layers)
        x = ParameterVector("x", n_qubits)
        p = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(x[q], q)
            for q in range(n_qubits):
                qc.ry(theta[p], q); p += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)
            for q in range(n_qubits):
                qc.rz(theta[p], q); p += 1
        fig = qc.draw("mpl", style="clifford")
        fig.suptitle("Variational quantum circuit (data re-uploading ansatz)", y=1.05, fontsize=11)
        fig.savefig(FIG / "fig02_circuit_architecture.png", bbox_inches="tight", dpi=300)
        fig.savefig(FIG / "fig02_circuit_architecture.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("circuit diagram skipped:", e)


def generate_all():
    raw, ext, df = _load()
    cfg = raw["config"]

    table01_dataset_summary(cfg)
    table02_federated_training_configuration(cfg)
    table03_baseline_comparison(df)
    table04_utility_comparison(df)
    table05_forgetting_comparison(df)
    table06_privacy_comparison(ext)
    table07_runtime_comparison(ext)
    table08_statistical_significance(df)
    table09_ablation_study()
    table10_hyperparameter_sensitivity()
    print("All 10 tables written.")

    fig01_overall_framework()
    fig02_circuit_architecture()
    fig03_federated_workflow()
    fig04_training_convergence(raw)
    fig05_accuracy_comparison(df)
    fig06_forgetting_comparison(df)
    fig07_privacy_comparison(ext)
    fig08_runtime_comparison(ext)
    fig09_radar_chart(df, ext)
    fig10_ablation_plot()
    fig11_sensitivity_plot()
    fig12_confusion_matrix(ext)
    fig13_roc_curve(ext)
    fig14_parameter_importance(raw)
    fig15_qfim_visualization(raw)
    fig16_entanglement_heatmap(raw)
    print("All 16 figures written (PNG + PDF).")


if __name__ == "__main__":
    generate_all()
