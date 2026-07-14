"""Focused tests for traceable Stage 1--4 scoring."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from pg_amcd import stage_scoring


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
) + b"\x00" * 128


def _manifest() -> dict:
    return {
        "run_id": "run-123",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "status": "completed",
        "start_timestamp": "2026-07-14T00:00:00+00:00",
        "end_timestamp": "2026-07-14T00:01:00+00:00",
        "cli_command": "pg-amcd run --through-stage 4",
        "python_version": "3.12.3",
        "operating_system": "Linux",
        "dependency_versions": {"numpy": "2.4.2"},
        "resolved_config": {"sampling_rate": 1000.0},
        "input_files": [{"path": "sample.mat", "sha256": "b" * 64}],
        "metadata_checksum": "c" * 64,
        "pipeline_version": "4.0.0",
        "feature_schema_version": "1.0.0",
        "input_validation": {"n_invalid": 0},
        "per_stage_runtime": {stage: 1.0 for stage in stage_scoring.STAGES},
        "per_recording_runtime": {"sample": 4.0},
        "output_checksums": {"Stage_1/sample/decomposition.npz": "d" * 64},
        "warnings": [],
        "failures": [],
        "stages": {stage: {"status": "completed", "runtime_seconds": 1.0}
                   for stage in stage_scoring.STAGES},
        "stage_evidence": {
            stage: {
                "tests": {"unit": True, "synthetic": True, "integration": True},
                "input_validation_passed": True,
            }
            for stage in stage_scoring.STAGES
        },
    }


def _write_stage_1_complete(run_dir: Path) -> None:
    record = run_dir / "Stage_1" / "sample"
    record.mkdir(parents=True)
    signal = np.linspace(-1.0, 1.0, 64)
    imfs = np.vstack((signal * 0.6, signal * 0.4))
    np.savez_compressed(record / "preprocessed_physical.npz", signal=signal)
    np.savez_compressed(record / "preprocessed_scaled.npz", signal=signal / 2.0)
    np.savez_compressed(
        record / "decomposition.npz",
        imfs=imfs,
        physical_source_signal=signal,
        scaled_source_signal=signal / 2.0,
        scale_factor=np.asarray(2.0),
        residual=np.zeros_like(signal),
    )
    (record / "imf_metrics.csv").write_text(
        "imf_index,energy_percentage,centre_frequency,bandwidth,spectral_entropy\n"
        "1,60,100,10,0.2\n2,40,50,8,0.3\n",
        encoding="utf-8",
    )
    (record / "cutoff_search.csv").write_text(
        "cutoff,cutoff_search,objective\n20,20,0.1\n", encoding="utf-8"
    )
    metrics = {
        "number_of_imfs": 2,
        "reconstruction_nrmse": 0.001,
        "absolute_orthogonality_index": 0.02,
        "signed_orthogonality_index": 0.01,
        "mean_adjacent_imf_correlation": 0.1,
        "maximum_adjacent_imf_correlation": 0.1,
        "spectral_overlap": 0.1,
        "frequency_ordering_score": 1.0,
        "energy_percentages": [60.0, 40.0],
        "centre_frequencies": [100.0, 50.0],
        "bandwidths": [10.0, 8.0],
        "spectral_entropies": [0.2, 0.3],
        "seed_stability": {
            "seed_centre_frequency_stability": 0.99,
            "seed_energy_distribution_stability": 0.99,
            "matched_imf_correlation": 0.99,
            "spectral_overlap_variation": 0.01,
            "imf_count_variation": 0.0,
        },
        "cutoff_search": {"selected_cutoff": 20.0},
        "ceemdan_runtime": 0.5,
        "residual_present": True,
    }
    (record / "stage_1_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (record / "stage_1_config.json").write_text(
        json.dumps({"algorithm": "CEEMDAN", "trials": 2, "noise_seed": 42}),
        encoding="utf-8",
    )
    (record / "stage_1_summary.md").write_text(
        "# Stage 1 summary\n\nCEEMDAN was used with explicit residual handling, "
        "physical amplitude preservation, controlled cutoff search, and deterministic seeds.\n",
        encoding="utf-8",
    )
    for _visual_id, aliases in stage_scoring.VISUAL_REQUIREMENTS["Stage_1"]:
        (record / f"{aliases[0]}.png").write_bytes(_PNG)


def test_complete_stage_1_scores_100_with_traceable_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    _write_stage_1_complete(run_dir)

    payload = stage_scoring.calculate_stage_scorecard(run_dir)

    assert payload["Stage_1"]["score"] == 100.0
    assert payload["Stage_1"]["failed"] == []
    assert sum(item["possible"] for item in payload["Stage_1"]["categories"].values()) == 100.0
    assert all(item["checks"] for item in payload["Stage_1"]["categories"].values())


def test_missing_artifacts_and_figures_apply_cap_and_generate_charts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _manifest()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "Stage_1" / "sample").mkdir(parents=True)

    paths = stage_scoring.generate_stage_scorecard(run_dir)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert payload["Stage_1"]["score"] <= 89
    assert "missing required output files" in payload["Stage_1"]["cap_reasons"]
    assert "missing required visualisations" in payload["Stage_1"]["cap_reasons"]
    assert Path(paths["scorecard_png"]).stat().st_size > 1_000
    assert Path(paths["progress_png"]).stat().st_size > 1_000


def test_unverified_automated_tests_apply_integrity_cap(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _manifest()
    manifest["stage_evidence"]["Stage_1"]["tests"]["integration"] = None
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_stage_1_complete(run_dir)

    payload = stage_scoring.calculate_stage_scorecard(run_dir)

    assert payload["Stage_1"]["raw_score"] > 89
    assert payload["Stage_1"]["score"] == 89
    assert "automated tests not verified or failing" in payload["Stage_1"]["cap_reasons"]


def test_dynamic_feature_names_and_current_writer_aliases_are_recognised() -> None:
    assert stage_scoring._key_matches(
        {"features_imf_1_energy_ratio"}, ("imf_energy_ratio",)
    )
    assert stage_scoring._key_matches(
        {"diagnostics_input_energy"}, ("input_energy",)
    )
