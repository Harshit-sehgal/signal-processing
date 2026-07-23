"""Unit tests for the visualization module."""
import json
import os

import matplotlib.pyplot as plt
import pytest

from pg_amcd import visualization as viz


def _write_scorecard(tmp_path):
    path = os.path.join(tmp_path, "project_scorecard.json")
    data = {
        "scorecard": {
            "architecture": 85.0,
            "correctness": 72.0,
            "input_validation": 90.0,
            "reproducibility": 66.0,
            "mathematical_validation": 61.0,
            "chatter_detection": 30.0,
            "research_readiness": 59.0,
            "visualisation": 20.0,
        }
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _write_validation_report(tmp_path):
    path = os.path.join(tmp_path, "validation_report.json")
    data = {
        "n_files": 10,
        "n_valid": 8,
        "n_invalid": 2,
        "metadata": {
            "missing_metadata": 1,
            "duplicate_metadata_entries": 0,
            "missing_chatter_label": 0,
            "invalid_rpm_values": 1,
            "invalid_tooth_values": 0,
            "metadata_row_no_file": 0,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _write_provenance(tmp_path):
    path = os.path.join(tmp_path, "provenance.json")
    data = {
        "files_processed": [
            {
                "path": "stickout/test.mat",
                "validation": {
                    "nrmse": 0.01,
                    "mode_mixing_index": 0.1,
                    "frequency_ordering_index": 0.9,
                },
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def test_plot_project_scorecard(tmp_path):
    scorecard_path = _write_scorecard(tmp_path)
    output_path = os.path.join(tmp_path, "scorecard.png")
    result = viz.plot_project_scorecard(scorecard_path, output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_validation_summary(tmp_path):
    report_path = _write_validation_report(tmp_path)
    output_path = os.path.join(tmp_path, "validation.png")
    result = viz.plot_validation_summary(report_path, output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_decomposition_metrics(tmp_path):
    prov_path = _write_provenance(tmp_path)
    output_path = os.path.join(tmp_path, "decomposition.png")
    result = viz.plot_decomposition_metrics(prov_path, output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_imf_gates(tmp_path):
    output_path = os.path.join(tmp_path, "gates.png")
    result = viz.plot_imf_gates([0.1, 0.3, 0.8], output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_confusion_matrix(tmp_path):
    output_path = os.path.join(tmp_path, "confusion.png")
    result = viz.plot_confusion_matrix([[9, 1], [2, 8]], output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_roc_curve(tmp_path):
    output_path = os.path.join(tmp_path, "roc.png")
    fpr = [0.0, 0.2, 0.5, 1.0]
    tpr = [0.0, 0.6, 0.8, 1.0]
    result = viz.plot_roc_curve([(fpr, tpr, "model")], output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_missing_input_returns_none():
    assert viz.plot_project_scorecard("/nonexistent/path.json") is None


def test_plot_cutoff_search(tmp_path):
    output_path = os.path.join(tmp_path, "cutoff_search.png")
    metrics = [
        {"cutoff_hz": 50.0, "final_score": 0.3, "spectral_overlap": 0.2},
        {"cutoff_hz": 100.0, "final_score": 0.1, "spectral_overlap": 0.15},
        {"cutoff_hz": 150.0, "final_score": 0.25, "spectral_overlap": 0.18},
    ]
    result = viz.plot_cutoff_search(metrics, output_path, selected_cutoff=100.0)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_ceemdan_convergence(tmp_path):
    output_path = os.path.join(tmp_path, "convergence.png")
    trial_counts = [10, 20, 50, 100]
    metrics_map = {
        "IMF count": [8, 8, 9, 9],
        "Spectral overlap": [0.3, 0.25, 0.22, 0.21],
        "Orthogonality": [0.05, 0.04, 0.03, 0.03],
    }
    result = viz.plot_ceemdan_convergence(trial_counts, metrics_map, output_path)
    assert result is None
    assert os.path.exists(output_path)


def test_plot_seed_stability_per_imf(tmp_path):
    output_path = os.path.join(tmp_path, "seed_stability.png")
    labels = ["IMF 1", "IMF 2", "IMF 3"]
    centre_frequencies = [[125.0, 126.0, 124.0], [40.0, 41.0, 39.0], [260.0, 259.0, 261.0]]
    energy_percentages = [[30.0, 31.0, 29.0], [20.0, 20.0, 20.0], [10.0, 10.0, 10.0]]
    matched_correlations = [0.99, 0.95, 0.88]
    result = viz.plot_seed_stability_per_imf(
        labels,
        centre_frequencies,
        energy_percentages,
        matched_correlations,
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_seed_stability_per_imf_returns_figure_without_output_path():
    labels = ["IMF 1", "IMF 2"]
    centre_frequencies = [[125.0, 124.0], [40.0, 41.0]]
    energy_percentages = [[30.0, 31.0], [20.0, 20.0]]
    matched_correlations = [0.99, 0.95]
    result = viz.plot_seed_stability_per_imf(
        labels,
        centre_frequencies,
        energy_percentages,
        matched_correlations,
    )
    assert result is not None
    assert isinstance(result, plt.Figure)
    plt.close(result)


def test_plot_seed_stability_per_imf_empty_labels_returns_none():
    result = viz.plot_seed_stability_per_imf([], [], [], [])
    assert result is None


def test_plot_seed_stability_per_imf_mismatched_lengths_raise(tmp_path):
    with pytest.raises(ValueError):
        viz.plot_seed_stability_per_imf(
            ["IMF 1", "IMF 2", "IMF 3"],
            [[125.0], [40.0]],
            [[30.0], [20.0], [10.0]],
            [0.99, 0.95, 0.88],
            os.path.join(tmp_path, "seed_stability.png"),
        )


def test_plot_gate_stability_matched(tmp_path):
    output_path = os.path.join(tmp_path, "gate_stability.png")
    result = viz.plot_gate_stability_matched(
        ["M1", "M2", "M3"],
        [0.9, 0.5, 0.2],
        [0.05, 0.08, 0.03],
        ["IMF 1", "IMF 2", "IMF 3"],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_cumulative_retention(tmp_path):
    output_path = os.path.join(tmp_path, "retention.png")
    result = viz.plot_cumulative_retention(
        ["Raw", "Stage 1", "Stage 2", "Stage 3"],
        [1.0, 0.85, 0.42, 0.38],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_engineering_scientific_scorecard(tmp_path):
    output_path = os.path.join(tmp_path, "eng_sci_scorecard.png")
    result = viz.plot_engineering_scientific_scorecard(
        [95.0, 90.0, 85.0, 80.0],
        [70.0, 75.0, 80.0, 85.0],
        ["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_adjacent_overlap_diagnostics(tmp_path):
    output_path = os.path.join(tmp_path, "adjacent_overlap.png")
    result = viz.plot_adjacent_overlap_diagnostics(
        ["IMF 1", "IMF 2", "IMF 3"],
        [125.0, 40.0, 260.0],
        [15.0, 10.0, 20.0],
        [0.2, 0.3],
        [0.1, -0.05],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_adjacent_overlap_diagnostics_returns_figure_without_output_path():
    result = viz.plot_adjacent_overlap_diagnostics(
        ["IMF 1", "IMF 2", "IMF 3"],
        [125.0, 40.0, 260.0],
        [15.0, 10.0, 20.0],
        [0.2, 0.3],
        [0.1, -0.05],
    )
    assert result is not None
    assert isinstance(result, plt.Figure)
    plt.close(result)


def test_plot_adjacent_overlap_diagnostics_empty_labels_returns_none():
    result = viz.plot_adjacent_overlap_diagnostics([], [], [], [], [])
    assert result is None


def test_plot_adjacent_overlap_diagnostics_mismatched_lengths_raise(tmp_path):
    with pytest.raises(ValueError):
        viz.plot_adjacent_overlap_diagnostics(
            ["IMF 1", "IMF 2", "IMF 3"],
            [125.0, 40.0],
            [15.0, 10.0, 20.0],
            [0.2],
            [0.1, -0.05],
            os.path.join(tmp_path, "adjacent_overlap.png"),
        )


def test_plot_harmonic_overlap_diagnostics(tmp_path):
    output_path = os.path.join(tmp_path, "harmonic_overlap.png")
    result = viz.plot_harmonic_overlap_diagnostics(
        [0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3],
        [0.05, 0.1, 0.15],
        [0.25, 0.4, 0.55],
        ["IMF 1", "IMF 2", "IMF 3"],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_wavelet_level_comparison(tmp_path):
    output_path = os.path.join(tmp_path, "wavelet_level.png")
    result = viz.plot_wavelet_level_comparison(
        ["db4", "db6", "db8"],
        [2, 3, 4],
        [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]],
        "SNR",
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_thresholding_comparison(tmp_path):
    output_path = os.path.join(tmp_path, "thresholding.png")
    result = viz.plot_thresholding_comparison(
        ["soft", "hard", "garrote"],
        {"SNR": [12.0, 10.0, 11.0], "Chatter retention": [0.8, 0.7, 0.75]},
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_transient_preservation(tmp_path):
    output_path = os.path.join(tmp_path, "transient.png")
    result = viz.plot_transient_preservation(
        [1.0, 2.0, 3.0],
        [0.95, 1.9, 2.85],
        ["Event 1", "Event 2", "Event 3"],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_ablation_matrix(tmp_path):
    output_path = os.path.join(tmp_path, "ablation.png")
    result = viz.plot_ablation_matrix(
        ["Baseline", "No gating", "Full method"],
        ["Stage 1", "Stage 2", "Stage 3"],
        [[0.6, 0.6, 0.6], [0.65, 0.55, 0.6], [0.8, 0.85, 0.9]],
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_plot_confidence_interval_bars(tmp_path):
    output_path = os.path.join(tmp_path, "ci_bars.png")
    result = viz.plot_confidence_interval_bars(
        ["A", "B", "C"],
        [0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6],
        [0.6, 0.7, 0.8],
        "Metric",
        "Dataset-level metric with 95% CI",
        output_path,
    )
    assert result is None
    assert os.path.exists(output_path)


def test_empty_input_returns_none(tmp_path):
    assert viz.plot_ceemdan_convergence([], {}, os.path.join(tmp_path, "empty.png")) is None
    assert viz.plot_cumulative_retention([], [], os.path.join(tmp_path, "empty.png")) is None
