"""End-to-end checks for the canonical Stage 1--4 command-line contract."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

PER_RECORDING_ARTIFACTS = {
    "Stage_1": (
        "preprocessed_physical.npz",
        "preprocessed_scaled.npz",
        "decomposition.npz",
        "imf_metrics.csv",
        "cutoff_search.csv",
        "stage_1_metrics.json",
        "stage_1_summary.md",
        "stage_1_config.json",
    ),
    "Stage_2": (
        "imf_indicators.csv",
        "imf_gates.csv",
        "weighted_reconstruction_scaled.npz",
        "weighted_reconstruction_physical.npz",
        "stage_2_metrics.json",
        "stage_2_summary.md",
        "stage_2_config.json",
    ),
    "Stage_3": (
        "wavelet_coefficients.npz",
        "wavelet_thresholds.csv",
        "denoised_scaled.npz",
        "denoised_physical.npz",
        "stage_3_metrics.json",
        "stage_3_summary.md",
        "stage_3_config.json",
    ),
    "Stage_4": (
        "window_features.csv",
        "window_features.json",
        "feature_schema.json",
        "feature_quality.json",
        "stage_4_metrics.json",
        "stage_4_summary.md",
        "stage_4_config.json",
    ),
}
STAGE_4_AGGREGATE_ARTIFACTS = (
    "all_recording_features.csv",
    "feature_summary.csv",
    "feature_missingness.csv",
    "feature_correlations.csv",
    "feature_schema.json",
)

# This deliberately resolves all scientific bands inside the 1 kHz fixture's
# Nyquist range.  The production loader deep-merges unspecified defaults.
TEST_CONFIG = {
    "through_stage": 4,
    "sampling_rate": 1000.0,
    "segment_points": 512,
    "use_physics_gating": False,
    "ceemdan": {
        "trials": 1,
        "search_trials": 1,
        "epsilon": 0.02,
        "noise_seed": 42,
        "sifting_iterations": 2,
        "search_sifting_iterations": 2,
        "gate_stability_sifting_iterations": 2,
        "search_cutoffs": [20.0],
        "search_seeds": 1,
        "stability_seeds": [42],
        "parallel": False,
        "max_imf": 3,
    },
    "maiw": {
        "chatter_band_center": 320.0,
        "chatter_band_spread": 60.0,
    },
    "wavelet": {"wavelet_name": "db2", "level": 2},
    "features": {
        "window_seconds": 0.25,
        "overlap_ratio": 0.5,
        "band_energy_ranges_hz": [
            [0.0, 50.0],
            [50.0, 150.0],
            [150.0, 300.0],
            [300.0, 490.0],
        ],
    },
    "output": {"png_dpi": 72, "write_svg": False},
}


def _config(*, use_physics: bool) -> dict:
    value = json.loads(json.dumps(TEST_CONFIG))
    value["use_physics_gating"] = use_physics
    return value


def _make_synthetic_mat(path: Path, fs: float = 1000.0, n_samples: int = 1000) -> None:
    """Write a finite two-column ``tsDS`` MAT recording."""

    rng = np.random.default_rng(1234)
    time = np.arange(n_samples) / fs
    signal = (
        0.6 * np.sin(2 * np.pi * 40.0 * time)
        + 0.4 * np.sin(2 * np.pi * 320.0 * time)
        + rng.normal(0.0, 0.15, n_samples)
    )
    scipy.io.savemat(path, {"tsDS": np.column_stack((time, signal))})


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    return environment


def _run_cli(*arguments: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pg_amcd.cli", *arguments],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _write_config(path: Path, *, use_physics: bool) -> None:
    path.write_text(json.dumps(_config(use_physics=use_physics)), encoding="utf-8")


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["relative_path", "recording_id", "rpm", "tooth_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def completed_cli_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("stage_1_4_cli")
    input_dir = root / "input"
    output_dir = root / "outputs"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "sample.mat")

    config_path = root / "config.json"
    _write_config(config_path, use_physics=True)
    metadata_path = root / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {
                "relative_path": "sample.mat",
                "recording_id": "sample_recording",
                "rpm": 600,
                "tooth_count": 2,
            }
        ],
    )
    arguments = (
        "run",
        "--input-dir",
        str(input_dir),
        "--metadata",
        str(metadata_path),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--through-stage",
        "4",
    )
    process = _run_cli(*arguments)
    run_dirs = (
        sorted(path for path in output_dir.iterdir() if path.is_dir())
        if output_dir.exists()
        else []
    )
    return {
        "process": process,
        "arguments": arguments,
        "run_dirs": run_dirs,
        "output_dir": output_dir,
    }


def test_cli_run_materialises_complete_stage_1_4_contract(completed_cli_run):
    process = completed_cli_run["process"]
    assert isinstance(process, subprocess.CompletedProcess)
    assert process.returncode == 0, f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"

    run_dirs = completed_cli_run["run_dirs"]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert len(run_dir.name) == 64
    int(run_dir.name, 16)

    top_level_directories = {path.name for path in run_dir.iterdir() if path.is_dir()}
    assert top_level_directories == {"Stage_1", "Stage_2", "Stage_3", "Stage_4", "report"}
    for filename in (
        "run_manifest.json",
        "stage_scorecard.json",
        "stage_scorecard.png",
        "stage_progress.png",
    ):
        assert (run_dir / filename).is_file(), filename
    assert (run_dir / "report" / "pipeline_report.md").is_file()
    assert (run_dir / "report" / "pipeline_report.html").is_file()
    assert (run_dir / "report" / "figures").is_dir()
    assert not any((run_dir / f"Stage_{stage}").exists() for stage in range(5, 8))

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["through_stage"] == 4
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 0
    assert manifest["failures"] == []
    assert manifest["metadata_checksum"]
    assert manifest["input_files"][0]["relative_path"] == "sample.mat"
    assert len(manifest["input_files"][0]["sha256"]) == 64
    assert manifest["stage_evidence"]["Stage_4"]["tests"]["integration"] is True
    assert manifest["selected_processing_parameters"]["sample_recording"]
    assert manifest["output_checksums"]

    for stage, filenames in PER_RECORDING_ARTIFACTS.items():
        recording_dir = run_dir / stage / "sample_recording"
        assert recording_dir.is_dir()
        for filename in filenames:
            path = recording_dir / filename
            assert path.is_file() and path.stat().st_size > 0, str(path)
        assert any(recording_dir.glob("*.png")), stage

    aggregate = run_dir / "Stage_4" / "aggregate"
    for filename in STAGE_4_AGGREGATE_ARTIFACTS:
        assert (aggregate / filename).is_file(), filename

    with np.load(run_dir / "Stage_1" / "sample_recording" / "decomposition.npz") as values:
        for key in (
            "time_segment",
            "scaled_source_segment",
            "physical_source_segment",
            "imfs",
            "residual_scaled",
            "residual_physical",
        ):
            assert key in values
            assert np.all(np.isfinite(values[key]))
    with np.load(
        run_dir / "Stage_2" / "sample_recording" / "weighted_reconstruction_physical.npz"
    ) as values:
        assert np.all(np.isfinite(values["signal"]))
    with np.load(run_dir / "Stage_3" / "sample_recording" / "denoised_physical.npz") as values:
        assert np.all(np.isfinite(values["signal"]))

    scorecard = json.loads((run_dir / "stage_scorecard.json").read_text(encoding="utf-8"))
    assert set(scorecard) >= {"Stage_1", "Stage_2", "Stage_3", "Stage_4"}
    assert all(scorecard[f"Stage_{number}"]["score"] >= 90 for number in range(1, 5))


def test_cli_reuses_only_identity_matched_completed_run(completed_cli_run):
    run_dir = completed_cli_run["run_dirs"][0]
    before = (run_dir / "run_manifest.json").read_bytes()
    process = _run_cli(*completed_cli_run["arguments"])

    assert process.returncode == 0, process.stderr
    assert "Reusing identity-matched completed run" in process.stdout
    assert (run_dir / "run_manifest.json").read_bytes() == before


def test_cli_public_surface_ends_at_stage_4():
    help_process = _run_cli("--help", timeout=120)
    assert help_process.returncode == 0
    assert "{run,validate,report}" in help_process.stdout
    assert "evaluate" not in help_process.stdout

    rejected = _run_cli(
        "run",
        "--input-dir",
        ".",
        "--through-stage",
        "5",
        timeout=120,
    )
    assert rejected.returncode != 0
    assert "outside the current project scope" in rejected.stderr


def test_cli_validate_accepts_a_finite_nonphysics_recording(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "sample.mat")
    config_path = tmp_path / "config.json"
    _write_config(config_path, use_physics=False)
    report_path = tmp_path / "validation_report.json"

    process = _run_cli(
        "validate",
        "--input-dir",
        str(input_dir),
        "--config",
        str(config_path),
        "--output",
        str(report_path),
        timeout=120,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "valid"
    assert report["n_files"] == report["n_valid"] == report["n_signal_valid"] == 1
    assert report["n_invalid"] == 0
    assert report["files"][0]["valid"] is True
    assert report["files"][0]["estimated_sampling_rate_hz"] == pytest.approx(1000.0, rel=0.05)
    assert report["warnings"] == [
        "No metadata supplied; metadata-dependent Stage 4 features will be undefined"
    ]


def test_cli_validate_rejects_sampling_rate_mismatch(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "bad.mat", fs=2000.0, n_samples=2000)
    config_path = tmp_path / "config.json"
    _write_config(config_path, use_physics=False)
    report_path = tmp_path / "validation_report.json"

    process = _run_cli(
        "validate",
        "--input-dir",
        str(input_dir),
        "--config",
        str(config_path),
        "--output",
        str(report_path),
        timeout=120,
    )

    assert process.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_files"] == 1
    assert report["n_valid"] == 0
    assert report["n_invalid"] == 1
    assert report["files"][0]["valid"] is False
    assert any(
        "differs from configured rate" in error.lower() for error in report["files"][0]["errors"]
    )


def test_cli_validate_physics_metadata_is_strict_and_label_optional(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "sample.mat")
    _make_synthetic_mat(input_dir / "orphan.mat")
    config_path = tmp_path / "config.json"
    _write_config(config_path, use_physics=True)
    metadata_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "validation_report.json"

    _write_metadata(
        metadata_path,
        [
            {
                "relative_path": "sample.mat",
                "recording_id": "sample",
                "rpm": 1200,
                "tooth_count": "",
            }
        ],
    )
    invalid = _run_cli(
        "validate",
        "--input-dir",
        str(input_dir),
        "--metadata",
        str(metadata_path),
        "--config",
        str(config_path),
        "--output",
        str(report_path),
        timeout=120,
    )
    assert invalid.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_files"] == 2
    assert report["n_valid"] == 0
    assert report["metadata"]["missing_recordings"] == ["orphan.mat"]
    assert report["metadata"]["missing_tooth_count"] == ["sample.mat"]
    errors_by_path = {entry["path"]: entry["errors"] for entry in report["files"]}
    assert any("tooth_count" in error for error in errors_by_path["sample.mat"])
    assert any("No metadata row" in error for error in errors_by_path["orphan.mat"])

    _write_metadata(
        metadata_path,
        [
            {
                "relative_path": relative,
                "recording_id": Path(relative).stem,
                "rpm": 1200,
                "tooth_count": 4,
            }
            for relative in ("sample.mat", "orphan.mat")
        ],
    )
    valid = _run_cli(
        "validate",
        "--input-dir",
        str(input_dir),
        "--metadata",
        str(metadata_path),
        "--config",
        str(config_path),
        "--output",
        str(report_path),
        timeout=120,
    )
    assert valid.returncode == 0, valid.stderr
    valid_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert valid_report["n_valid"] == 2
    assert valid_report["metadata"]["missing_recordings"] == []
    assert valid_report["metadata"]["missing_tooth_count"] == []
    assert all(entry["metadata"]["label"] is None for entry in valid_report["files"])


def test_cli_run_physics_mode_refuses_missing_metadata_before_processing(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "sample.mat")
    config_path = tmp_path / "config.json"
    _write_config(config_path, use_physics=True)

    process = _run_cli(
        "run",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--config",
        str(config_path),
        timeout=120,
    )

    assert process.returncode == 2
    assert "requires --metadata" in process.stdout
    assert "tooth_count" in process.stdout
    assert not (tmp_path / "outputs").exists()


def test_cli_rejects_ambiguous_duplicate_metadata_before_processing(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(input_dir / "sample.mat")
    config_path = tmp_path / "config.json"
    _write_config(config_path, use_physics=True)
    metadata_path = tmp_path / "metadata.csv"
    duplicate_rows = [
        {
            "relative_path": "sample.mat",
            "recording_id": recording_id,
            "rpm": rpm,
            "tooth_count": 2,
        }
        for recording_id, rpm in (("first", 600), ("second", 900))
    ]
    _write_metadata(metadata_path, duplicate_rows)
    validation_report = tmp_path / "validation.json"

    validation = _run_cli(
        "validate",
        "--input-dir",
        str(input_dir),
        "--metadata",
        str(metadata_path),
        "--config",
        str(config_path),
        "--output",
        str(validation_report),
    )
    assert validation.returncode == 1
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert report["metadata"]["ambiguous_rows"] == 1
    assert any("ambiguous" in failure.lower() for failure in report["failures"])

    output_dir = tmp_path / "outputs"
    run = _run_cli(
        "run",
        "--input-dir",
        str(input_dir),
        "--metadata",
        str(metadata_path),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
    )
    assert run.returncode == 2
    assert "Metadata is ambiguous" in run.stdout
    assert not output_dir.exists()
