"""Unit tests for the visualization module."""

import json
import os


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
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_validation_summary(tmp_path):
    report_path = _write_validation_report(tmp_path)
    output_path = os.path.join(tmp_path, "validation.png")
    result = viz.plot_validation_summary(report_path, output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_decomposition_metrics(tmp_path):
    prov_path = _write_provenance(tmp_path)
    output_path = os.path.join(tmp_path, "decomposition.png")
    result = viz.plot_decomposition_metrics(prov_path, output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_imf_gates(tmp_path):
    output_path = os.path.join(tmp_path, "gates.png")
    result = viz.plot_imf_gates([0.1, 0.3, 0.8], output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_confusion_matrix(tmp_path):
    output_path = os.path.join(tmp_path, "confusion.png")
    result = viz.plot_confusion_matrix([[9, 1], [2, 8]], output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_roc_curve(tmp_path):
    output_path = os.path.join(tmp_path, "roc.png")
    fpr = [0.0, 0.2, 0.5, 1.0]
    tpr = [0.0, 0.6, 0.8, 1.0]
    result = viz.plot_roc_curve([(fpr, tpr, "model")], output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_plot_missing_input_returns_none():
    assert viz.plot_project_scorecard("/nonexistent/path.json") is None
