"""Focused tests for the Stage 1--4 combined report."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from pg_amcd.stage_reporting import SECTION_TITLES, generate_pipeline_report


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
) + b"\x00" * 128


def _write_minimal_run(run_dir: Path) -> None:
    run_dir.mkdir()
    manifest = {
        "run_id": "report-run",
        "git_commit": "a" * 40,
        "git_dirty": True,
        "status": "completed",
        "start_timestamp": "2026-07-14T00:00:00+00:00",
        "end_timestamp": "2026-07-14T00:01:00+00:00",
        "cli_command": "pg-amcd run --through-stage 4",
        "python_version": "3.12",
        "operating_system": "Linux",
        "dependency_versions": {"numpy": "2.4.2"},
        "resolved_config": {"sampling_rate": 1000},
        "input_files": [{"path": "sample.mat", "sha256": "b" * 64}],
        "metadata_checksum": "c" * 64,
        "pipeline_version": "4.0.0",
        "feature_schema_version": "1.0.0",
        "input_validation": {"n_files": 1, "n_valid": 1, "n_invalid": 0},
        "per_stage_runtime": {"Stage_1": 1.0},
        "per_recording_runtime": {"sample": 1.0},
        "output_checksums": {"Stage_1/sample/stage_1_metrics.json": "d" * 64},
        "warnings": ["synthetic-only evidence"],
        "failures": [],
        "limitations": ["real-data Stage 1 run not yet recorded"],
        "stages": {"Stage_1": {"status": "completed", "runtime_seconds": 1.0}},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = run_dir / "Stage_1" / "sample"
    record.mkdir(parents=True)
    (record / "stage_1_metrics.json").write_text(
        json.dumps({"selected_cutoff_hz": 321.5, "reconstruction_nrmse": 0.002}),
        encoding="utf-8",
    )
    (record / "imf_metrics.csv").write_text(
        "imf_index,energy_percentage\n1,100\n", encoding="utf-8"
    )
    (record / "01_raw_signal.png").write_bytes(_PNG)


def test_report_has_ten_sections_and_reads_run_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    paths = generate_pipeline_report(run_dir)
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    report_html = Path(paths["html"]).read_text(encoding="utf-8")

    for index, title in enumerate(SECTION_TITLES, start=1):
        assert f"## {index}. {title}" in markdown
        assert f"{index}. {title}" in report_html
    assert "321.5" in markdown
    assert "synthetic-only evidence" in markdown
    assert "real-data Stage 1 run not yet recorded" in markdown
    assert "figures/Stage_1__sample__01_raw_signal.png" in markdown
    assert (Path(paths["figures_dir"]) / "Stage_1__sample__01_raw_signal.png").is_file()


def test_report_requires_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    try:
        generate_pipeline_report(run_dir)
    except FileNotFoundError as exc:
        assert "run_manifest.json" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("report generation accepted a run without a manifest")

