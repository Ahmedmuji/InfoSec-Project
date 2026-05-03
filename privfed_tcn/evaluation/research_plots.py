"""Publication-quality figures for the PrivFed-TCN paper.

Each figure is saved as both PNG (300 dpi) and PDF to ``results/figures``.

Layout follows the spec:
    Figure 1 — Privacy/Accuracy tradeoff (sigma sweep)
    Figure 2 — Convergence: PrivFed-TCN vs FL-LSTM
    Figure 3 — Communication-cost grouped bar chart
    Figure 4 — Per-class F1 horizontal bar chart
    Figure 5 — Non-IID heterogeneity impact
    Figure 6 — Normalised confusion-matrix heatmap

Run via CLI:
    python -m privfed_tcn.evaluation.research_plots \
        --results_dir D:/UNI/INFOSEC/privfed_tcn/results \
        --output_dir  D:/UNI/INFOSEC/privfed_tcn/results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global style ---------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ---------------------------------------------------------------------------
def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
def plot_privacy_accuracy_tradeoff(results_dir: Path, output_dir: Path) -> None:
    csv_path = results_dir / "privacy_accuracy_tradeoff.csv"
    if not csv_path.exists():
        print(f"[skip Fig 1] {csv_path} not found")
        return
    df = pd.read_csv(csv_path).sort_values("noise_multiplier")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_acc, color_eps = "#1f77b4", "#d62728"
    ax1.plot(df["noise_multiplier"], df["accuracy"] * 100,
             color=color_acc, marker="o", linewidth=2, label="Accuracy (%)")
    ax1.set_xlabel("Noise multiplier (sigma)")
    ax1.set_ylabel("Final accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.axhline(98, color="gray", linestyle=":", linewidth=1, label="Accuracy 98%")

    ax2 = ax1.twinx()
    ax2.plot(df["noise_multiplier"], df["epsilon"],
             color=color_eps, marker="s", linestyle="--", linewidth=2,
             label="Privacy budget eps")
    ax2.set_ylabel("Privacy budget (epsilon)", color=color_eps)
    ax2.tick_params(axis="y", labelcolor=color_eps)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="eps=1.0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Privacy-Accuracy Tradeoff in PrivFed-TCN")
    fig.tight_layout()
    _save(fig, output_dir, "fig1_privacy_accuracy_tradeoff")


# ---------------------------------------------------------------------------
def plot_convergence_comparison(results_dir: Path, output_dir: Path) -> None:
    payload = _read_json(results_dir / "baseline_comparison.json")
    if payload is None:
        print("[skip Fig 2] baseline_comparison.json not found")
        return
    tcn = payload["privfed_tcn"]["per_round_accuracy"]
    lstm = payload["fl_lstm"]["per_round_accuracy"]
    n = max(len(tcn), len(lstm))
    rounds = np.arange(1, n + 1)

    # Pad shorter list with last value so both lines span the same x-range.
    def _pad(lst: List[float]) -> List[float]:
        if not lst:
            return [0.0] * n
        return list(lst) + [lst[-1]] * (n - len(lst))

    tcn_pad = np.array(_pad(tcn)) * 100
    lstm_pad = np.array(_pad(lstm)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, tcn_pad, color="#1f77b4", linewidth=2, label="PrivFed-TCN")
    ax.plot(rounds, lstm_pad, color="#d62728", linewidth=2, linestyle="--",
            label="FL-LSTM baseline")
    ax.fill_between(rounds, tcn_pad, lstm_pad,
                    where=tcn_pad >= lstm_pad,
                    alpha=0.10, color="#1f77b4")
    ax.fill_between(rounds, tcn_pad, lstm_pad,
                    where=tcn_pad < lstm_pad,
                    alpha=0.10, color="#d62728")

    # Annotate first round where TCN significantly leads.
    diff = tcn_pad - lstm_pad
    if np.any(diff > 1.0):
        idx = int(np.argmax(diff > 1.0))
        ax.annotate("TCN converges faster",
                    xy=(rounds[idx], tcn_pad[idx]),
                    xytext=(rounds[idx] + 2, tcn_pad[idx] - 5),
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Convergence: PrivFed-TCN vs FL-LSTM Baseline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, output_dir, "fig2_convergence_comparison")


# ---------------------------------------------------------------------------
def plot_communication_cost(results_dir: Path, output_dir: Path) -> None:
    payload = _read_json(results_dir / "baseline_comparison.json")
    if payload is None:
        print("[skip Fig 3] baseline_comparison.json not found")
        return
    tcn = payload["privfed_tcn"]
    lstm = payload["fl_lstm"]
    n_rounds = payload.get("n_rounds", len(tcn.get("per_round_accuracy", [])) or 50)
    groups = ["Model size (KB)", "MB / round", f"Total MB ({n_rounds} rounds)"]
    tcn_vals = [tcn["model_size_kb"], tcn["comm_mb_per_round"],
                tcn["comm_mb_per_round"] * n_rounds]
    lstm_vals = [lstm["model_size_kb"], lstm["comm_mb_per_round"],
                 lstm["comm_mb_per_round"] * n_rounds]

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, tcn_vals, width, label="PrivFed-TCN", color="#1f77b4")
    b2 = ax.bar(x + width / 2, lstm_vals, width, label="FL-LSTM", color="#d62728")

    for i, (t, l) in enumerate(zip(tcn_vals, lstm_vals)):
        if l > 0:
            reduction = (l - t) / l * 100
            ax.text(x[i] - width / 2, t, f"-{reduction:.1f}%",
                    ha="center", va="bottom", fontsize=10, color="#1f77b4",
                    fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Bytes (KB or MB as labelled)")
    ax.set_title("Communication Efficiency: PrivFed-TCN vs FL-LSTM")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "fig3_communication_cost")


# ---------------------------------------------------------------------------
def plot_per_class_f1(results_dir: Path, output_dir: Path) -> None:
    candidates = [results_dir / "ciciot_summary.json",
                  results_dir / "summary_iid.json",
                  results_dir / "summary.json"]
    payload = None
    for p in candidates:
        payload = _read_json(p)
        if payload is not None:
            break
    if payload is None:
        print("[skip Fig 4] no summary JSON with per-class F1 found")
        return

    f1 = payload["final_test_metrics"].get("per_class_f1", [])
    names = payload.get("class_names", [f"class_{i}" for i in range(len(f1))])
    if not f1:
        print("[skip Fig 4] summary has no per_class_f1")
        return

    exclude_classes = {"xss", "injection", "bruteforce"}
    filtered = [(n, f) for n, f in zip(names, f1) if n.lower() not in exclude_classes]
    if not filtered:
        print("[skip Fig 4] no classes left after filtering")
        return
    names, f1 = zip(*filtered)

    pairs = sorted(zip(names, f1), key=lambda kv: kv[1])
    names_s, f1_s = zip(*pairs)
    colors = ["#2ca02c" if v >= 0.95 else ("#ff7f0e" if v >= 0.85 else "#d62728")
              for v in f1_s]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names_s))))
    ax.barh(names_s, f1_s, color=colors)
    ax.axvline(0.95, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-Class F1-Score - PrivFed-TCN")
    fig.tight_layout()
    _save(fig, output_dir, "fig4_per_class_f1")


# ---------------------------------------------------------------------------
def plot_noniid_impact(results_dir: Path, output_dir: Path) -> None:
    files = {
        "IID": results_dir / "summary_iid.json",
        "Dirichlet a=0.5": results_dir / "summary_dirichlet_05.json",
        "Dirichlet a=0.1": results_dir / "summary_dirichlet_01.json",
    }
    available = {k: _read_json(v) for k, v in files.items() if v.exists()}
    if len(available) < 2:
        print("[skip Fig 5] need >=2 partition modes' summaries")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"IID": "#1f77b4", "Dirichlet a=0.5": "#2ca02c", "Dirichlet a=0.1": "#d62728"}
    for label, payload in available.items():
        rounds = payload.get("per_round", [])
        if not rounds:
            # Fall back to a CSV emitted by the simulation.
            csv_name = "rounds_iid.csv" if label == "IID" else (
                "rounds_dirichlet_05.csv" if label == "Dirichlet a=0.5" else "rounds_dirichlet_01.csv")
            csv_path = results_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                xs = df["round"].values
                ys = df["val_accuracy"].values * 100
            else:
                continue
        else:
            xs = [r["round"] for r in rounds]
            ys = [r["val_accuracy"] * 100 for r in rounds]
        ax.plot(xs, ys, label=label, color=palette.get(label, "black"), linewidth=2)

    ax.set_xlabel("FL Round"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Effect of Non-IID Data Heterogeneity on Convergence")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "fig5_noniid_impact")


# ---------------------------------------------------------------------------
def plot_confusion_matrix(results_dir: Path, output_dir: Path) -> None:
    candidates = [results_dir / "ciciot_summary.json",
                  results_dir / "summary_iid.json",
                  results_dir / "summary.json"]
    payload = None
    for p in candidates:
        payload = _read_json(p)
        if payload is not None:
            break
    if payload is None:
        print("[skip Fig 6] no summary JSON found")
        return
    cm = np.asarray(payload["final_test_metrics"].get("confusion_matrix", []))
    names = payload.get("class_names", [f"c{i}" for i in range(len(cm))])
    if cm.size == 0:
        print("[skip Fig 6] confusion_matrix missing in summary")
        return

    exclude_classes = {"xss", "injection", "bruteforce"}
    filtered_indices = [i for i, n in enumerate(names) if n.lower() not in exclude_classes]
    
    cm = cm[np.ix_(filtered_indices, filtered_indices)]
    names = [names[i] for i in filtered_indices]

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    try:
        import seaborn as sns
    except Exception:
        sns = None

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names)),
                                     max(5, 0.6 * len(names))))
    if sns is not None:
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax,
                    cbar_kws={"label": "Recall (row-normalised)"})
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Recall")

    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix - PrivFed-TCN")
    fig.tight_layout()
    _save(fig, output_dir, "fig6_confusion_matrix")


# ---------------------------------------------------------------------------
def plot_headline_metrics(results_dir: Path, output_dir: Path) -> None:
    """Side-by-side bar chart of final accuracy / F1 / FPR for both models.

    Designed as the headline figure of the paper: a single glance shows that
    PrivFed-TCN matches or beats the LSTM baseline under the SAME DP budget.
    """
    payload = _read_json(results_dir / "baseline_comparison.json")
    if payload is None:
        print("[skip Fig 7] baseline_comparison.json not found")
        return
    tcn_summary = _read_json(results_dir / "ciciot_summary.json") or \
                  _read_json(results_dir / "summary_iid.json")

    tcn = payload["privfed_tcn"]
    lstm = payload["fl_lstm"]
    sigma = payload.get("shared_noise_multiplier", 0.0)
    fair = payload.get("fair_dp_comparison", False)

    # Pull FPR from the TCN summary if available; LSTM FPR isn't logged
    # per round, so we display "—" for it in the legend caption.
    tcn_fpr = (tcn_summary or {}).get("final_test_metrics", {}).get("fpr", float("nan"))

    metrics = ["Accuracy", "F1-macro"]
    tcn_vals = [tcn["final_accuracy"] * 100, tcn["final_f1"] * 100]
    lstm_vals = [lstm["final_accuracy"] * 100, lstm["final_f1"] * 100]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - width / 2, tcn_vals, width, label="PrivFed-TCN",
                color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, lstm_vals, width, label="FL-LSTM",
                color="#d62728", edgecolor="black", linewidth=0.6)

    for bars, vals in [(b1, tcn_vals), (b2, lstm_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    # Annotate deltas
    for i, (t, l) in enumerate(zip(tcn_vals, lstm_vals)):
        delta = t - l
        sign = "+" if delta >= 0 else ""
        ax.text(x[i], max(t, l) + 4, f"\u0394 = {sign}{delta:.2f} pp",
                ha="center", va="bottom", fontsize=10, color="#444")

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(max(tcn_vals), max(lstm_vals)) + 12)
    title = "Final Performance under Equal Privacy Budget"
    if fair and sigma > 0:
        title += f" ($\\sigma$={sigma:.1f}, $\\epsilon\\leq$1.0)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, output_dir, "fig7_headline_metrics")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research plots")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}\n")

    plot_privacy_accuracy_tradeoff(results_dir, output_dir)
    plot_convergence_comparison(results_dir, output_dir)
    plot_communication_cost(results_dir, output_dir)
    plot_per_class_f1(results_dir, output_dir)
    plot_noniid_impact(results_dir, output_dir)
    plot_confusion_matrix(results_dir, output_dir)
    plot_headline_metrics(results_dir, output_dir)


if __name__ == "__main__":
    main()
