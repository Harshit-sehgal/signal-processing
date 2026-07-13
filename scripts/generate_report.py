"""Generate a Markdown evaluation report and figures (Segment 7).

Reads ``outputs/evaluation_results.json`` (produced by
``scripts/evaluate_dataset.py --mat-dir``) and emits:

* a human-readable Markdown table of per-model detection metrics at three
  grouped-evaluation levels (leave-one-recording-out, leave-one-stickout-out,
  leave-one-rpm-out — Goal 6.4),
* a ROC curve, a confusion matrix for the strongest model, and a cross-condition
  balanced-accuracy bar chart (Goal 7 figures),
* an explicit limitations section stating the aspirational targets and the
  real measured values.

Degrades gracefully when no results are present.
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_KEYS = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
GROUP_LEVELS = [
    ("leave_one_recording_out", "Leave-one-recording-out"),
    ("leave_one_stickout_out", "Leave-one-stickout-out"),
    ("leave_one_rpm_out", "Leave-one-rpm-out"),
]
ASPIRATIONAL = {
    "balanced_accuracy": 0.95,
    "precision": 0.95,
    "recall": 0.95,
    "f1": 0.95,
    "roc_auc": 0.98,
}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _metrics_table(data, level_key, title, lines):
    lines.append(f"## {title}")
    lines.append("")
    header = "| Model | " + " | ".join(METRIC_KEYS) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(METRIC_KEYS) + 1)) + "|")
    for model, metrics in data.get(level_key, {}).items():
        if not metrics:
            continue
        row = "| " + model + " | " + " | ".join(
            _fmt(metrics.get(mk)) for mk in METRIC_KEYS
        ) + " |"
        lines.append(row)
    lines.append("")


def _ablation_section(data, lines):
    abl = data.get("feature_ablations", {})
    if not abl:
        return
    lines.append("## Feature ablations")
    lines.append("")
    lines.append(
        "Same leakage-proof leave-one-recording-out CV as above, but with a "
        "feature subset dropped, to isolate the contribution of the PG-AMCD "
        "frequency/IMF-derived features."
    )
    lines.append("")
    header = "| Ablation | models | RF balanced_acc | RF roc_auc |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * 4) + "|")
    full_rf = (data.get("leave_one_recording_out", {})
               .get("random_forest", {}) or {})
    lines.append(
        "| full feature set | 4 | "
        f"{_fmt(full_rf.get('balanced_accuracy'))} | {_fmt(full_rf.get('roc_auc'))} |"
    )
    for name, levels in abl.items():
        rf = (levels.get("leave_one_recording_out", {}) or {}).get("random_forest", {})
        n_models = len([m for m, v in levels.get("leave_one_recording_out", {}).items() if v])
        lines.append(
            f"| {name} | {n_models} | "
            f"{_fmt(rf.get('balanced_accuracy'))} | {_fmt(rf.get('roc_auc'))} |"
        )
    lines.append("")
def _roc_figure(data, fig_dir):
    oof = data.get("oof", {})
    y_true = np.asarray(oof.get("y_true", []), dtype=int)
    proba = oof.get("proba", {})
    if len(y_true) == 0 or not proba:
        return None
    plt.figure(figsize=(6, 5))
    for name, p in proba.items():
        fpr, tpr, _ = roc_curve(y_true, np.asarray(p, dtype=float))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (leave-one-recording-out)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "roc_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _confusion_figure(data, fig_dir):
    oof = data.get("oof", {})
    y_true = np.asarray(oof.get("y_true", []), dtype=int)
    proba = oof.get("proba", {})
    best = data.get("calibration", {}).get("best_model")
    if len(y_true) == 0 or best not in proba:
        return None
    pred = (np.asarray(proba[best], dtype=float) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix ({best})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["stable", "chatter"])
    plt.yticks([0, 1], ["stable", "chatter"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _bar_figure(data, fig_dir):
    models = set()
    for level_key, _ in GROUP_LEVELS:
        models.update(m for m, v in data.get(level_key, {}).items() if v)
    models = sorted(models)
    if not models:
        return None
    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, (level_key, title) in enumerate(GROUP_LEVELS):
        vals = [
            (data.get(level_key, {}).get(m, {}) or {}).get("balanced_accuracy", np.nan)
            for m in models
        ]
        plt.bar(x + i * width, vals, width, label=title.replace("leave_one_", "LO-"))
    plt.axhline(ASPIRATIONAL["balanced_accuracy"], color="red", ls="--",
                label=f"target {ASPIRATIONAL['balanced_accuracy']:.2f}")
    plt.xlabel("Model")
    plt.ylabel("Balanced accuracy")
    plt.title("Cross-condition balanced accuracy")
    plt.xticks(x + width, models, rotation=20, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "cross_condition_balanced_acc.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD report generator")
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "outputs", "evaluation_results.json"),
    )
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "outputs", "evaluation_report.md")
    )
    parser.add_argument(
        "--figure-dir", default=os.path.join(ROOT, "outputs", "figures")
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(
            f"No evaluation results at {args.input}.\n"
            "Run `make evaluate` (or scripts/evaluate_dataset.py --mat-dir) on a "
            "labelled dataset first."
        )
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    fig_dir = args.figure_dir
    os.makedirs(fig_dir, exist_ok=True)

    lines = ["# PG-AMCD Chatter Detection Evaluation Report", ""]
    lines.append(
        "Metrics use leakage-proof grouped cross-validation (Goal 6.4): windows "
        "from one recording never span train/test."
    )
    lines.append("")
    lc = data.get("label_counts", {})
    lines.append(
        f"**Dataset:** {data.get('n_recordings', '?')} recordings, "
        f"{data.get('n_windows', '?')} windows "
        f"(chatter={lc.get('chatter', '?')}, stable={lc.get('stable', '?')}). "
        f"Features: {len(data.get('feature_keys', []))} window-level "
        f"(Goal 6.2)."
    )
    lines.append("")

    for level_key, title in GROUP_LEVELS:
        _metrics_table(data, level_key, title, lines)

    # Calibration
    cal = data.get("calibration", {})
    lines.append("## Probability calibration")
    lines.append("")
    lines.append(f"Best model (by ROC-AUC): **{cal.get('best_model')}** "
                 f"(method: {cal.get('method')}).")
    cm = cal.get("calibrated_metrics")
    if cm:
        lines.append("")
        lines.append("| Metric | Calibrated |")
        lines.append("| --- | --- |")
        for mk in METRIC_KEYS:
            lines.append(f"| {mk} | {_fmt(cm.get(mk))} |")
    lines.append("")


    _ablation_section(data, lines)
    # Figures
    roc = _roc_figure(data, fig_dir)
    conf = _confusion_figure(data, fig_dir)
    bar = _bar_figure(data, fig_dir)
    lines.append("## Figures")
    lines.append("")
    for name, path in (("ROC curve", roc), ("Confusion matrix", conf),
                       ("Cross-condition balanced accuracy", bar)):
        if path:
            lines.append(f"* {name}: `{os.path.relpath(path, ROOT)}`")
    lines.append("")

    # Limitations / aspirational targets
    lines.append("## Limitations and aspirational targets")
    lines.append("")
    lines.append(
        "The roadmap sets aspirational targets (balanced accuracy >= 0.95, "
        "precision >= 0.95, recall >= 0.95, F1 >= 0.95, ROC-AUC >= 0.98). The "
        "real measured values above are reported honestly even where they fall "
        "short. Improving them is future work: larger/balanced labelled corpora, "
        "per-recording multi-window temporal smoothing (Goal 6.5), and additional "
        "feature/architecture tuning."
    )
    lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Wrote report to {args.output}")
    for name, path in (("ROC", roc), ("confusion", conf), ("bar", bar)):
        if path:
            print(f"Wrote {name} figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
