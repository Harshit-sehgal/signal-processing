"""Reusable chart-generation functions for PG-AMCD progress reporting.

All functions accept file paths so that figures are generated from saved
JSON/CSV metrics rather than from in-memory state.  This keeps the dashboard
source-of-truth in versioned artifacts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_or_return(fig, output_path: Optional[str] = None) -> Optional[plt.Figure]:
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# --------------------------------------------------------------------------- #
# Project-level charts
# --------------------------------------------------------------------------- #
def plot_project_scorecard(
    scorecard_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
) -> Optional[plt.Figure]:
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
    selected_cutoff: Optional[float] = None,
) -> Optional[plt.Figure]:
    """Plot cutoff search objective value with selected cutoff and second-best gap."""
    if not cutoff_metrics:
        return None

    def _cutoff_key(row: Any) -> float:
        row = cast(Dict[str, Any], row)
        return float(row.get("cutoff", row.get("cutoff_hz", 0.0)))

    def _objective_key(row: Any) -> float:
        row = cast(Dict[str, Any], row)
        return float(row.get("final_score", row.get("objective", 0.0)))

    sorted_metrics = sorted(cutoff_metrics, key=_objective_key)
    cutoffs = [_cutoff_key(m) for m in sorted_metrics]
    objectives = [_objective_key(m) for m in sorted_metrics]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cutoffs, objectives, marker="o", color="#2878B5", label="Objective")

    if selected_cutoff is None and cutoffs:
        selected_cutoff = cutoffs[0]

    if selected_cutoff is not None:
        ax.axvline(selected_cutoff, color="crimson", linestyle="--", label="Selected")

    if len(objectives) >= 2:
        best, second = objectives[0], objectives[1]
        gap = second - best
        ax.annotate(
            f"gap to 2nd = {gap:.4g}",
            xy=(cutoffs[0], best),
            xytext=(cutoffs[0], best + 0.05 * (max(objectives) - min(objectives) or 1.0)),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            fontsize=8,
        )

    for key in ["spectral_overlap", "max_adjacent_correlation", "absolute_oi", "seed_instability"]:
        if key in sorted_metrics[0]:
            ax.plot(
                cutoffs,
                [m[key] for m in sorted_metrics],
                marker=".",
                alpha=0.6,
                label=key.replace("_", " "),
            )

    ax.set_xlabel("Cutoff frequency (Hz)")
    ax.set_ylabel("Objective / component value")
    ax.set_title("Controlled cutoff-objective search")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_ceemdan_convergence(
    trial_counts: Sequence[int],
    metrics_map: Dict[str, Sequence[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> Optional[plt.Figure]:
    """Plot CEEMDAN metrics versus trial count for convergence analysis.

    ``metrics_map`` keys are subplot titles (e.g. IMF count, spectral overlap,
    orthogonality, matched IMF correlation, selected cutoff, runtime).
    """
    n = len(metrics_map)
    if n == 0 or not trial_counts:
        return None

    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, metrics_map.items()):
        ax.plot(trial_counts, values, marker="o", color="#2878B5")
        ax.set_xlabel("CEEMDAN trial count")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs CEEMDAN trial count")
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_seed_stability_per_imf(
    imf_labels: Sequence[str],
    centre_frequencies: Sequence[Sequence[float]],
    energy_percentages: Sequence[Sequence[float]],
    matched_correlations: Sequence[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> Optional[plt.Figure]:
    """Plot per-IMF seed stability with confidence intervals.

    ``centre_frequencies`` and ``energy_percentages`` have shape
    (n_imfs, n_seeds).  ``matched_correlations`` is a vector of per-IMF
    correlations after structural matching.
    """
    labels = list(imf_labels)
    n = len(labels)
    if n == 0:
        return None

    cf = np.asarray(centre_frequencies)
    ep = np.asarray(energy_percentages)
    mc = np.asarray(matched_correlations)

    if cf.shape[0] != n or ep.shape[0] != n or mc.shape[0] != n:
        raise ValueError(
            "centre_frequencies, energy_percentages and matched_correlations "
            f"must have {n} rows/entries (one per IMF)"
        )
    if cf.ndim != 2 or ep.ndim != 2 or cf.shape[1] != ep.shape[1]:
        raise ValueError(
            "centre_frequencies and energy_percentages must be 2-D arrays "
            "with the same shape (n_imfs, n_seeds)"
        )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = cast(NDArray[Any], axes)

    ax = axes[0, 0]
    positions = np.arange(n)
    if cf.size:
        cf_mean = np.mean(cf, axis=1)
        cf_std = np.std(cf, axis=1)
        ax.errorbar(positions, cf_mean, yerr=cf_std, fmt="o", capsize=4, color="#2878B5")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Centre frequency (Hz)")
    ax.set_title("Centre frequency across seeds")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    if ep.size:
        ep_mean = np.mean(ep, axis=1)
        ep_std = np.std(ep, axis=1)
        ax.errorbar(positions, ep_mean, yerr=ep_std, fmt="o", capsize=4, color="#59A14F")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Energy percentage (%)")
    ax.set_title("Energy percentage across seeds")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.bar(positions, matched_correlations, color="#F28E2B")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Correlation")
    ax.set_title("Matched-IMF correlation across seeds")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    if cf.size:
        ax.boxplot(cf.T)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Centre frequency (Hz)")
        ax.set_title("Centre-frequency distribution across seeds")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_gate_stability_matched(
    labels: Sequence[str],
    mean_gates: Sequence[float],
    std_gates: Sequence[float],
    matched_ids: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Plot gate stability using matched-IMF identities rather than raw indices."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(x, mean_gates, yerr=std_gates, fmt="o", capsize=4, color="#2878B5")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Gate value")
    ax.set_title(f"Gate stability across seeds (matched modes: {', '.join(matched_ids)})")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cumulative_retention(
    stages: Sequence[str],
    retention_values: Sequence[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional[plt.Figure]:
    """Bar chart of cumulative energy retention across pipeline stages."""
    if not stages or not retention_values:
        return None
    if len(stages) != len(retention_values):
        raise ValueError("stages and retention_values must have the same length")
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(stages, retention_values, color=["#2878B5", "#59A14F", "#F28E2B", "#E15759"])
    ax.set_ylabel("Retention / ratio")
    ax.set_title("Cumulative chatter-band retention")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, retention_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_engineering_scientific_scorecard(
    engineering_scores: Sequence[float],
    scientific_scores: Sequence[float],
    stage_labels: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Side-by-side engineering vs scientific scores per stage."""
    x = np.arange(len(stage_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, engineering_scores, width, label="Engineering / Completeness", color="#2878B5")
    ax.bar(x + width / 2, scientific_scores, width, label="Scientific / Performance", color="#E15759")
    ax.set_xticks(x, stage_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Engineering vs Scientific scores by stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_adjacent_overlap_diagnostics(
    imf_labels: Sequence[str],
    centre_frequencies: Sequence[float],
    bandwidths: Sequence[float],
    overlaps: Sequence[float],
    correlations: Sequence[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Optional[plt.Figure]:
    """Stage 1 diagnostic: adjacent IMF centre frequencies, bandwidths and overlap."""
    n = len(imf_labels)
    if n == 0:
        return None
    if len(centre_frequencies) != n or len(bandwidths) != n:
        raise ValueError(
            "centre_frequencies and bandwidths must have the same length as imf_labels"
        )
    if len(overlaps) != n - 1 or len(correlations) != n - 1:
        raise ValueError(
            "overlaps and correlations must have length len(imf_labels) - 1"
        )

    labels = [f"{imf_labels[i]}-{imf_labels[i + 1]}" for i in range(n - 1)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = cast(NDArray[Any], axes)

    ax = axes[0, 0]
    ax.bar(x, centre_frequencies[:-1], alpha=0.7, label="lower IMF")
    ax.bar(x, centre_frequencies[1:], alpha=0.7, label="upper IMF")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Centre frequency (Hz)")
    ax.set_title("Adjacent IMF centre frequencies")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.bar(x, bandwidths[:-1], alpha=0.7, label="lower IMF")
    ax.bar(x, bandwidths[1:], alpha=0.7, label="upper IMF")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Bandwidth (Hz)")
    ax.set_title("Adjacent IMF bandwidths")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.bar(x, overlaps, color="#E15759")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Spectral overlap")
    ax.set_ylim(0, 1)
    ax.set_title("Adjacent IMF spectral overlap")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, correlations, color="#59A14F")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(-1, 1)
    ax.set_title("Adjacent IMF time-domain correlation")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_harmonic_overlap_diagnostics(
    spindle_only: Sequence[float],
    tooth_only: Sequence[float],
    overlap: Sequence[float],
    union: Sequence[float],
    labels: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Stage 2 diagnostic: spindle-only, tooth-only, overlap and union energy per IMF."""
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - 1.5 * width, spindle_only, width, label="Spindle-only")
    ax.bar(x - 0.5 * width, tooth_only, width, label="Tooth-only")
    ax.bar(x + 0.5 * width, overlap, width, label="Overlap")
    ax.bar(x + 1.5 * width, union, width, label="Union")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Energy ratio")
    ax.set_title("Spindle/tooth harmonic energy overlap per IMF")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_wavelet_level_comparison(
    wavelet_names: Sequence[str],
    levels: Sequence[int],
    metric_matrix: Sequence[Sequence[float]],
    metric_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Optional[plt.Figure]:
    """Stage 3 heatmap comparing wavelet families and decomposition levels."""
    mat = np.asarray(metric_matrix)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(levels)), labels=[str(level) for level in levels])
    ax.set_yticks(np.arange(len(wavelet_names)), labels=wavelet_names)
    ax.set_xlabel("Decomposition level")
    ax.set_ylabel("Wavelet")
    ax.set_title(f"{metric_name} by wavelet and level")
    for i in range(len(wavelet_names)):
        for j in range(len(levels)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label=metric_name)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_thresholding_comparison(
    modes: Sequence[str],
    metric_values: Dict[str, Sequence[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Stage 3 grouped bar chart of soft/hard/garrote/firm/SURE thresholding."""
    x = np.arange(len(modes))
    width = 0.8 / max(1, len(metric_values))
    fig, ax = plt.subplots(figsize=figsize)
    for idx, (metric, values) in enumerate(metric_values.items()):
        ax.bar(x + idx * width, values, width, label=metric)
    ax.set_xticks(x + width * (len(metric_values) - 1) / 2, modes, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Denoising thresholding-mode comparison")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_transient_preservation(
    true_events: Sequence[float],
    recovered_events: Sequence[float],
    labels: Sequence[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Stage 3 transient event timing error or energy retention."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, true_events, label="Ground truth", s=80, marker="x")
    ax.scatter(x, recovered_events, label="Recovered", s=80, marker="o")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Amplitude / time / energy")
    ax.set_title("Transient preservation per event")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_ablation_matrix(
    experiments: Sequence[str],
    stages: Sequence[str],
    matrix: Sequence[Sequence[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Optional[plt.Figure]:
    """Cross-stage ablation matrix heatmap."""
    mat = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(stages)), labels=stages)
    ax.set_yticks(np.arange(len(experiments)), labels=experiments)
    ax.set_title("End-to-end ablation matrix")
    for i in range(len(experiments)):
        for j in range(len(stages)):
            text_color = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_confidence_interval_bars(
    labels: Sequence[str],
    means: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Optional[plt.Figure]:
    """Generic dataset-level bar chart with error bars showing confidence intervals."""
    x = np.arange(len(labels))
    means_arr = np.asarray(means)
    yerr = np.asarray([means_arr - np.asarray(lower), np.asarray(upper) - means_arr])
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, means_arr, yerr=yerr, capsize=4)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save_or_return(fig, output_path)
