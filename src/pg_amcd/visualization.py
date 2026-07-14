"""Reusable chart-generation functions for PG-AMCD progress reporting.

All functions accept file paths so that figures are generated from saved
JSON/CSV metrics rather than from in-memory state.  This keeps the dashboard
source-of-truth in versioned artifacts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_or_return(fig, output_path: Optional[str] = None) -> Optional[str]:
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return None


# --------------------------------------------------------------------------- #
# Project-level charts
# --------------------------------------------------------------------------- #
def plot_project_scorecard(
    scorecard_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[str]:
    """Bar chart of the eight segment scores."""
    data = _load_json(scorecard_path)
    if data is None or "scorecard" not in data:
        return None

    scores = data["scorecard"]
    labels = [
        "Architecture",
        "Correctness",
        "Input Validation",
        "Reproducibility",
        "Mathematical Validation",
        "Chatter Detection",
        "Research Readiness",
        "Visualisation",
    ]
    values = [
        scores.get(k, 0.0)
        for k in [
            "architecture",
            "correctness",
            "input_validation",
            "reproducibility",
            "mathematical_validation",
            "chatter_detection",
            "research_readiness",
            "visualisation",
        ]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color="steelblue")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title("PG-AMCD Project Scorecard")
    ax.axvline(100, color="green", linestyle="--", alpha=0.5, label="target")
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_score_history(
    history_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[str]:
    """Line chart of each segment score over commits/time."""
    data = _load_json(history_path)
    if not data:
        return None

    labels = [
        "architecture",
        "correctness",
        "input_validation",
        "reproducibility",
        "mathematical_validation",
        "chatter_detection",
        "research_readiness",
        "visualisation",
    ]
    x = list(range(len(data)))
    fig, ax = plt.subplots(figsize=figsize)
    for label in labels:
        y = [entry["scorecard"].get(label, 0.0) for entry in data]
        ax.plot(x, y, marker="o", label=label.replace("_", " ").title())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Run index")
    ax.set_ylabel("Score")
    ax.set_title("Project Score History")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return _save_or_return(fig, output_path)


# --------------------------------------------------------------------------- #
# Validation charts
# --------------------------------------------------------------------------- #
def plot_validation_summary(
    validation_report_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional[str]:
    """Bar chart of validation outcomes (valid, invalid, metadata issues)."""
    report = _load_json(validation_report_path)
    if report is None:
        return None

    meta = report.get("metadata", {})
    labels = [
        "Valid",
        "Invalid",
        "Missing meta",
        "Duplicate meta",
        "Missing labels",
        "Invalid RPM",
        "Invalid tooth",
        "Unmatched rows",
    ]
    values = [
        report.get("n_valid", 0),
        report.get("n_invalid", 0),
        meta.get("missing_metadata", 0),
        meta.get("duplicate_metadata_entries", 0),
        meta.get("missing_chatter_label", 0),
        meta.get("invalid_rpm_values", 0),
        meta.get("invalid_tooth_values", 0),
        meta.get("metadata_row_no_file", 0),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = ["green" if label == "Valid" else "steelblue" for label in labels]
    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Count")
    ax.set_title("Dataset Validation Summary")
    ax.tick_params(axis="x", rotation=30)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_validation_error_distribution(
    validation_report_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> Optional[str]:
    """Pie chart of validation-error categories."""
    report = _load_json(validation_report_path)
    if report is None:
        return None

    meta = report.get("metadata", {})
    labels = []
    sizes = []
    for k, label in [
        ("sampling_rate_mismatches", "Sampling rate"),
        ("missing_metadata", "Missing metadata"),
        ("missing_chatter_label", "Missing label"),
        ("invalid_rpm_values", "Invalid RPM"),
        ("invalid_tooth_values", "Invalid tooth"),
        ("duplicate_metadata_entries", "Duplicate meta"),
        ("metadata_row_no_file", "Unmatched row"),
    ]:
        v = meta.get(k, 0)
        if v:
            labels.append(label)
            sizes.append(v)

    if not sizes:
        labels = ["No errors"]
        sizes = [1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Validation Error Distribution")
    plt.tight_layout()
    return _save_or_return(fig, output_path)


# --------------------------------------------------------------------------- #
# Decomposition / pipeline charts
# --------------------------------------------------------------------------- #
def plot_decomposition_metrics(
    provenance_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[str]:
    """Grouped bar chart of per-file decomposition metrics."""
    data = _load_json(provenance_path)
    if data is None:
        return None

    files = [f for f in data.get("files_processed", []) if "validation" in f]
    if not files:
        return None

    names = [os.path.basename(f["path"]) for f in files]
    nrmse = [f["validation"].get("nrmse", 0.0) for f in files]
    mmi = [f["validation"].get("mode_mixing_index", 0.0) for f in files]
    foi = [f["validation"].get("frequency_ordering_index", 0.0) for f in files]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, nrmse, width, label="NRMSE")
    ax.bar(x, mmi, width, label="MMI")
    ax.bar(x + width, foi, width, label="Frequency ordering")
    ax.set_xticks(x, labels=names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Metric value")
    ax.set_title("Decomposition Metrics per Recording")
    ax.legend()
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_imf_gates(
    gate_values: Sequence[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    labels: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Bar chart of IMF gate values."""
    values = list(gate_values)
    if labels is None:
        labels = [f"IMF {i + 1}" for i in range(len(values))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("Gate value")
    ax.set_title("IMF Gate Values")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_signal_stages(
    time: Sequence[float],
    raw: Sequence[float],
    preprocessed: Sequence[float],
    gated: Sequence[float],
    denoised: Sequence[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> Optional[str]:
    """Stacked plot of raw, preprocessed, gated, and denoised signals."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for ax, sig, title in zip(
        axes,
        [raw, preprocessed, gated, denoised],
        ["Raw", "Preprocessed", "Physics-gated", "Wavelet-denoised"],
    ):
        ax.plot(time, sig, color="steelblue")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_psd_comparison(
    freqs: Sequence[Sequence[float]],
    psds: Sequence[Sequence[float]],
    labels: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[str]:
    """Overlay PSDs for raw, preprocessed, gated, and denoised signals."""
    fig, ax = plt.subplots(figsize=figsize)
    for f, p, label in zip(freqs, psds, labels):
        ax.semilogy(f, p, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density Comparison")
    ax.legend()
    plt.tight_layout()
    return _save_or_return(fig, output_path)


# --------------------------------------------------------------------------- #
# Detection charts
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
    class_names: Sequence[str] = ("stable", "chatter"),
) -> Optional[str]:
    """Plot a confusion matrix from a 2x2 array."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_roc_curve(
    fpr_tpr_pairs: Sequence[Tuple[Sequence[float], Sequence[float], str]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> Optional[str]:
    """Plot one or more ROC curves."""
    fig, ax = plt.subplots(figsize=figsize)
    for fpr, tpr, label in fpr_tpr_pairs:
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_precision_recall_curve(
    precision_recall_pairs: Sequence[Tuple[Sequence[float], Sequence[float], str]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> Optional[str]:
    """Plot one or more precision-recall curves."""
    fig, ax = plt.subplots(figsize=figsize)
    for prec, rec, label in precision_recall_pairs:
        ax.plot(rec, prec, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_calibration_curve(
    prob_true_pairs: Sequence[Tuple[Sequence[float], Sequence[float], str]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> Optional[str]:
    """Plot calibration curves (predicted vs observed probability)."""
    fig, ax = plt.subplots(figsize=figsize)
    for prob, freq, label in prob_true_pairs:
        ax.plot(prob, freq, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_detection_timeline(
    time: Sequence[float],
    signal: Sequence[float],
    probabilities: Sequence[float],
    threshold: float = 0.5,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> Optional[str]:
    """Plot signal, chatter probability, and decision threshold over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, signal, color="gray", alpha=0.6, label="signal")
    ax.plot(time, probabilities, color="red", label="chatter probability")
    ax.axhline(threshold, color="blue", linestyle="--", label="threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / Probability")
    ax.set_title("Detection Timeline")
    ax.legend()
    plt.tight_layout()
    return _save_or_return(fig, output_path)


# --------------------------------------------------------------------------- #
# Research / ablation charts
# --------------------------------------------------------------------------- #
def plot_ablation_results(
    ablation_data: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    metric: str = "f1",
) -> Optional[str]:
    """Bar chart of ablation F1 scores relative to the full pipeline."""
    names = list(ablation_data.keys())
    values = [ablation_data[n].get(metric, 0.0) for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, values, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Ablation Results")
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cross_condition_heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Optional[str]:
    """Plot a cross-condition generalisation heatmap."""
    mat = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title("Cross-Condition Performance")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = mat[i, j]
            # Choose a text colour that contrasts with the viridis colormap.
            text_color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cutoff_search(
    cutoff_metrics: Sequence[Dict[str, Any]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[str]:
    """Plot cutoff search objective components vs cutoff frequency."""
    if not cutoff_metrics:
        return None
    cutoffs = [m["cutoff"] for m in cutoff_metrics]
    fig, ax = plt.subplots(figsize=figsize)
    for key in ["spectral_overlap", "max_adjacent_correlation", "absolute_oi", "seed_instability"]:
        if key in cutoff_metrics[0]:
            ax.plot(cutoffs, [m[key] for m in cutoff_metrics], marker="o", label=key)
    ax.set_xlabel("Cutoff frequency (Hz)")
    ax.set_ylabel("Objective component")
    ax.set_title("Cutoff Search Objectives")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_return(fig, output_path)
