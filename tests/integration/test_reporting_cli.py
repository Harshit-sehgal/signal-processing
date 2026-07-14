"""Integration checks for the manifest-driven Stage 1--4 report command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    return environment


def _run_report(run_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pg_amcd.cli",
            "report",
            "--run-dir",
            str(run_dir),
            *extra,
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _write_manifest_driven_run(run_dir: Path) -> None:
    run_dir.mkdir()
    for stage_number in range(1, 5):
        (run_dir / f"Stage_{stage_number}").mkdir()
    (run_dir / "report" / "figures").mkdir(parents=True)

    manifest = {
        "run_id": "abc123",
        "git_commit": "a" * 40,
        "git_dirty": True,
        "status": "completed",
        "start_timestamp": "2026-07-14T00:00:00+00:00",
        "end_timestamp": "2026-07-14T00:00:01+00:00",
        "cli_command": "pg-amcd run --through-stage 4",
        "python_version": "3.12",
        "operating_system": "Linux",
        "dependency_versions": {"numpy": "2.0.0"},
        "resolved_config": {"through_stage": 4, "sampling_rate": 1000.0},
        "input_files": [{"relative_path": "sample.mat", "sha256": "b" * 64}],
        "metadata_checksum": "c" * 64,
        "pipeline_version": "4.0.0",
        "feature_schema_version": "1.0.0",
        "input_validation": {"n_files": 1, "n_valid": 1, "n_invalid": 0},
        "per_stage_runtime": {f"Stage_{number}": 0.1 for number in range(1, 5)},
        "per_recording_runtime": {"sample": 0.4},
        "output_checksums": {"Stage_1/sample/stage_1_metrics.json": "d" * 64},
        "warnings": ["fixture warning retained from manifest"],
        "failures": [],
        "limitations": ["fixture limitation retained from manifest"],
        "stage_evidence": {
            f"Stage_{number}": {
                "status": "completed",
                "tests": {"unit": True, "synthetic": True, "integration": True},
                "input_validation_passed": True,
            }
            for number in range(1, 5)
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    stage_1 = run_dir / "Stage_1" / "sample"
    stage_1.mkdir()
    (stage_1 / "stage_1_metrics.json").write_text(
        json.dumps(
            {
                "selected_cutoff_hz": 123.456,
                "reconstruction_nrmse": 0.012345,
            }
        ),
        encoding="utf-8",
    )
    (stage_1 / "imf_metrics.csv").write_text(
        "imf_index,energy_percentage\n1,100\n", encoding="utf-8"
    )


def test_cli_report_regenerates_scorecard_and_combined_reports(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_manifest_driven_run(run_dir)

    process = _run_report(run_dir)

    assert process.returncode == 0, process.stderr
    for filename in ("stage_scorecard.json", "stage_scorecard.png", "stage_progress.png"):
        assert (run_dir / filename).is_file()
        assert (run_dir / filename).stat().st_size > 0
    markdown_path = run_dir / "report" / "pipeline_report.md"
    html_path = run_dir / "report" / "pipeline_report.html"
    assert markdown_path.is_file()
    assert html_path.is_file()
    assert (run_dir / "report" / "figures").is_dir()

    markdown = markdown_path.read_text(encoding="utf-8")
    report_html = html_path.read_text(encoding="utf-8")
    assert "123.456" in markdown
    assert "0.012345" in markdown
    assert "fixture warning retained from manifest" in markdown
    assert "fixture limitation retained from manifest" in markdown
    for title in (
        "Run overview",
        "Input validation",
        "Preprocessing summary",
        "Stage 1 decomposition",
        "Stage 2 IMF gating",
        "Stage 3 wavelet denoising",
        "Stage 4 feature extraction",
        "Stage scorecard",
        "Warnings and failures",
        "Limitations",
    ):
        assert title in markdown
        assert title in report_html


def test_cli_report_fails_when_required_manifest_is_missing(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    process = _run_report(run_dir)

    assert process.returncode != 0
    assert "run_manifest.json" in process.stderr
    assert not (run_dir / "report" / "pipeline_report.md").exists()


def test_cli_report_rejects_retired_cross_run_comparison_option(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_manifest_driven_run(run_dir)

    process = _run_report(run_dir, "--baseline-dir", str(tmp_path / "baseline"))

    assert process.returncode == 2
    assert "unrecognized arguments: --baseline-dir" in process.stderr
