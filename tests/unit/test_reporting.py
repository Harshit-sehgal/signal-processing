"""Unit tests for the reporting module."""

import json
import os

from pg_amcd.reporting import (
    generate_html_dashboard,
    generate_markdown_report,
    generate_csv_summaries,
    generate_run_reports,
    compare_runs,
)


def _write_run_artifacts(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    provenance = {
        "run_id": "abc123",
        "success_count": 2,
        "failure_count": 0,
        "files_processed": [
            {
                "path": "stickout/test.mat",
                "validation": {
                    "nrmse": 0.01,
                    "mode_mixing_index": 0.1,
                    "frequency_ordering_index": 0.9,
                },
                "diagnostics": {"nrmse": 0.01, "mmi": 0.1},
            }
        ],
    }
    with open(os.path.join(run_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh)

    scorecard = {
        "commit": "abc123",
        "timestamp": "2024-01-01T00:00:00",
        "scorecard": {
            "architecture": 85.0,
            "correctness": 72.0,
            "input_validation": 90.0,
            "reproducibility": 66.0,
            "mathematical_validation": 61.0,
            "chatter_detection": 30.0,
            "research_readiness": 59.0,
            "visualisation": 20.0,
        },
        "overall": 61.625,
        "details": {},
    }
    with open(os.path.join(run_dir, "project_scorecard.json"), "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh)

    validation = {
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
    with open(os.path.join(run_dir, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(validation, fh)


def test_generate_html_dashboard(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    _write_run_artifacts(run_dir)
    path = generate_html_dashboard(
        run_dir,
        scorecard_path=os.path.join(run_dir, "project_scorecard.json"),
        validation_report_path=os.path.join(run_dir, "validation_report.json"),
        provenance_path=os.path.join(run_dir, "provenance.json"),
    )
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "PG-AMCD Run Report" in html
    assert "Overall Score" in html


def test_generate_markdown_report(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    _write_run_artifacts(run_dir)
    path = generate_markdown_report(
        run_dir,
        scorecard_path=os.path.join(run_dir, "project_scorecard.json"),
        validation_report_path=os.path.join(run_dir, "validation_report.json"),
        provenance_path=os.path.join(run_dir, "provenance.json"),
    )
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        md = fh.read()
    assert "# PG-AMCD Run Report" in md
    assert "Project Scorecard" in md


def test_generate_csv_summaries(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    _write_run_artifacts(run_dir)
    paths = generate_csv_summaries(
        run_dir,
        output_dir=os.path.join(run_dir, "metrics"),
        provenance_path=os.path.join(run_dir, "provenance.json"),
    )
    assert "validation_metrics" in paths
    assert "feature_summary" in paths
    assert os.path.exists(paths["validation_metrics"])
    assert os.path.exists(paths["feature_summary"])


def test_generate_run_reports(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    _write_run_artifacts(run_dir)
    paths = generate_run_reports(
        run_dir,
        validation_report_path=os.path.join(run_dir, "validation_report.json"),
    )
    assert "scorecard" in paths
    assert "html" in paths
    assert "markdown" in paths
    assert os.path.exists(paths["scorecard"])
    assert os.path.exists(paths["html"])
    assert os.path.exists(paths["markdown"])
    assert os.path.exists(os.path.join(run_dir, "project_scorecard.png"))
    assert os.path.exists(os.path.join(run_dir, "validation_summary.png"))


def test_compare_runs(tmp_path):
    base_dir = os.path.join(str(tmp_path), "base")
    cand_dir = os.path.join(str(tmp_path), "candidate")
    _write_run_artifacts(base_dir)
    _write_run_artifacts(cand_dir)
    path = compare_runs(base_dir, cand_dir)
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "Run Comparison" in html
