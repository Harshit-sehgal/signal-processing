"""Malformed-artifact and integrity-cap contracts for Stage 1--4 scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pg_amcd import stage_scoring


def _record(tmp_path: Path, stage: str = "Stage_1") -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    record = run_dir / stage / "sample"
    record.mkdir(parents=True)
    return run_dir, record


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _documented_schema() -> dict[str, object]:
    return {
        "feature_schema_version": "1.0.0",
        "features": {
            "rms": {
                "family": "time_domain",
                "description": "Root mean square",
                "unit": "signal unit",
                "required_source_stage": "Stage_3",
                "metadata_required": False,
                "dimensionless": False,
                "invalid_handling": "null with reason",
            }
        },
    }


def test_safe_json_recording_discovery_and_evidence_coercion(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert stage_scoring._safe_json(missing) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert stage_scoring._safe_json(malformed) is None
    sequence = tmp_path / "sequence.json"
    _write_json(sequence, [1, 2])
    assert stage_scoring._safe_json(sequence) is None
    mapping = tmp_path / "mapping.json"
    _write_json(mapping, {"ok": True})
    assert stage_scoring._safe_json(mapping) == {"ok": True}

    assert stage_scoring._recording_dirs(tmp_path / "absent") == []
    stage_dir = tmp_path / "Stage_1"
    (stage_dir / "sample").mkdir(parents=True)
    for excluded in ("aggregate", "figures", "report"):
        (stage_dir / excluded).mkdir()
    assert stage_scoring._recording_dirs(stage_dir) == [stage_dir / "sample"]

    assert (
        stage_scoring._evidence_bool(
            {"stage_evidence": {"Stage_1": {"tests": {"unit": 1}}}},
            "Stage_1",
            ("unit",),
        )
        is True
    )
    assert (
        stage_scoring._evidence_bool(
            {"quality_gates": {"Stage_1": {"integration": 0}}},
            "Stage_1",
            ("integration",),
        )
        is False
    )
    assert (
        stage_scoring._evidence_bool(
            {"stages": {"Stage_1": {"status_check": "passed"}}},
            "Stage_1",
            ("status_check",),
        )
        is True
    )
    assert (
        stage_scoring._evidence_bool(
            {"stage_status": {"Stage_1": {"status_check": "error"}}},
            "Stage_1",
            ("status_check",),
        )
        is False
    )
    assert (
        stage_scoring._evidence_bool(
            {"stage_evidence": {"Stage_1": {"unit": "unknown"}}},
            "Stage_1",
            ("unit",),
        )
        is None
    )


def test_weighting_failure_and_global_completion_fallbacks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no scoring checks"):
        stage_scoring._weighted_checks("automated_tests", [])
    assert stage_scoring._stage_has_failure({"failures": "not-a-list"}, "Stage_1") is False

    _, record = _record(tmp_path)
    passed, evidence = stage_scoring._stage_completed({}, "Stage_1", [record])
    assert passed is False
    assert "global status" in evidence
    failed, _ = stage_scoring._stage_completed(
        {"status": "completed", "failures": [{"stage": "Stage_1"}]},
        "Stage_1",
        [record],
    )
    assert failed is False


def test_machine_output_parsers_reject_malformed_files(tmp_path: Path) -> None:
    json_path = tmp_path / "value.json"
    csv_path = tmp_path / "value.csv"
    npz_path = tmp_path / "value.npz"
    text_path = tmp_path / "value.txt"
    _write_json(json_path, {"value": 1})
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    np.savez_compressed(npz_path, signal=np.arange(4))
    text_path.write_text("content", encoding="utf-8")
    assert all(
        stage_scoring._parseable(path) for path in (json_path, csv_path, npz_path, text_path)
    )

    json_path.write_text("{", encoding="utf-8")
    csv_path.unlink()
    npz_path.write_bytes(b"not an archive")
    text_path.write_text("", encoding="utf-8")
    assert not stage_scoring._parseable(json_path)
    assert not stage_scoring._parseable(csv_path)
    assert not stage_scoring._parseable(npz_path)
    assert not stage_scoring._parseable(text_path)

    _, record = _record(tmp_path / "machine")
    passed, evidence = stage_scoring._machine_outputs_parseable("Stage_1", [record])
    assert passed is False
    assert "unreadable" in evidence
    passed, _ = stage_scoring._machine_outputs_parseable("Stage_4", [])
    assert passed is False


def test_finite_or_explained_distinguishes_corruption_from_stage4_nulls(
    tmp_path: Path,
) -> None:
    _, record = _record(tmp_path / "nonfinite", "Stage_1")
    _write_json(record / "metrics.json", {"metric": float("nan")})
    np.savez_compressed(record / "signal.npz", signal=np.asarray([1.0, np.inf]))
    (record / "values.csv").write_text("value\nnan\n", encoding="utf-8")
    passed, evidence = stage_scoring._finite_or_explained("Stage_1", [record])
    assert passed is False
    assert "metrics.json:metric" in evidence
    assert "signal.npz:signal" in evidence
    assert "values.csv" in evidence

    _, stage4 = _record(tmp_path / "explained", "Stage_4")
    (stage4 / "window_features.csv").write_text("feature\nnan\n", encoding="utf-8")
    _write_json(
        stage4 / "feature_quality.json",
        {"undefined_features": {"feature": "missing metadata reason"}},
    )
    assert stage_scoring._finite_or_explained("Stage_4", [stage4])[0] is True

    (stage4 / "broken.json").write_text("{", encoding="utf-8")
    passed, evidence = stage_scoring._finite_or_explained("Stage_4", [stage4])
    assert passed is False
    assert "unreadable" in evidence


def test_metric_key_and_stage4_feature_fallbacks(tmp_path: Path) -> None:
    _, record = _record(tmp_path / "metrics", "Stage_1")
    _write_json(record / "stage_1_metrics.json", {"number_of_imfs": 2})
    broken_csv = record / "broken.csv"
    broken_csv.symlink_to(record / "missing-target.csv")
    assert "number_of_imfs" in stage_scoring._metric_keys(record, "Stage_1")

    _, stage4 = _record(tmp_path / "stage4", "Stage_4")
    _write_json(stage4 / "feature_schema.json", {"features": []})
    outcomes = stage_scoring._metric_outcomes("Stage_4", [stage4])
    assert outcomes
    assert all(outcome[2] is False for outcome in outcomes)


def test_core_scientific_sanity_rejects_invalid_stage1_and_stage2_artifacts(
    tmp_path: Path,
) -> None:
    run_dir, stage1 = _record(tmp_path / "stage1", "Stage_1")
    del run_dir
    _write_json(
        stage1 / "stage_1_metrics.json",
        {"number_of_imfs": 0, "reconstruction_nrmse": -1, "frequency_ordering_score": 0.5},
    )
    assert "invalid IMF count" in stage_scoring._core_scientific_sanity("Stage_1", [stage1])[1]
    _write_json(
        stage1 / "stage_1_metrics.json",
        {"number_of_imfs": 2, "reconstruction_nrmse": 0, "frequency_ordering_score": 2},
    )
    assert "frequency-ordering" in stage_scoring._core_scientific_sanity("Stage_1", [stage1])[1]

    run_dir, stage2 = _record(tmp_path / "stage2", "Stage_2")
    passed, evidence = stage_scoring._core_scientific_sanity("Stage_2", [stage2])
    assert passed is False and "gate table missing" in evidence
    (stage2 / "imf_gates.csv").write_text("index\n1\n", encoding="utf-8")
    assert "gate values missing" in stage_scoring._core_scientific_sanity("Stage_2", [stage2])[1]
    (stage2 / "imf_gates.csv").write_text("gate\nnot-a-number\n", encoding="utf-8")
    assert "invalid gate values" in stage_scoring._core_scientific_sanity("Stage_2", [stage2])[1]
    (stage2 / "imf_gates.csv").write_text("gate\n1.5\n", encoding="utf-8")
    assert "out-of-bounds" in stage_scoring._core_scientific_sanity("Stage_2", [stage2])[1]
    (stage2 / "imf_gates.csv").write_text("gate\n0.5\n", encoding="utf-8")
    assert "traceability missing" in stage_scoring._core_scientific_sanity("Stage_2", [stage2])[1]
    stage1_trace = run_dir / "Stage_1" / "sample"
    stage1_trace.mkdir(parents=True)
    np.savez_compressed(stage1_trace / "decomposition.npz", imfs=np.ones((2, 8)))
    assert "count mismatch" in stage_scoring._core_scientific_sanity("Stage_2", [stage2])[1]


def test_stage3_and_stage4_sanity_require_lengths_thresholds_and_versioned_rows(
    tmp_path: Path,
) -> None:
    run_dir, stage3 = _record(tmp_path / "stage3", "Stage_3")
    stage2 = run_dir / "Stage_2" / "sample"
    stage2.mkdir(parents=True)
    np.savez_compressed(stage3 / "denoised_scaled.npz", signal=np.ones(7))
    np.savez_compressed(stage2 / "weighted_reconstruction_scaled.npz", signal=np.ones(8))
    assert "length mismatch" in stage_scoring._core_scientific_sanity("Stage_3", [stage3])[1]
    np.savez_compressed(stage3 / "denoised_scaled.npz", signal=np.ones(8))
    assert "no stored thresholds" in stage_scoring._core_scientific_sanity("Stage_3", [stage3])[1]

    _, stage4 = _record(tmp_path / "stage4", "Stage_4")
    _write_json(stage4 / "feature_schema.json", {"version": "unversioned"})
    assert "versioned schema" in stage_scoring._core_scientific_sanity("Stage_4", [stage4])[1]
    _write_json(stage4 / "feature_schema.json", {"feature_schema_version": "1.0.0"})
    (stage4 / "window_features.csv").write_text("time_rms\n1.0\n", encoding="utf-8")
    _write_json(stage4 / "stage_4_metrics.json", {})
    assert "repeated-extraction" in stage_scoring._core_scientific_sanity("Stage_4", [stage4])[1]
    _write_json(
        stage4 / "stage_4_metrics.json",
        {"repeat_extraction_stability": {"deterministic": True}},
    )
    assert stage_scoring._core_scientific_sanity("Stage_4", [stage4])[0] is True


def test_schema_traceability_metadata_modes_and_stage_input_validation(
    tmp_path: Path,
) -> None:
    mapping_schema = {
        "schema": {
            "rms": {
                "family": "time_domain",
                "formula": "sqrt(mean(x^2))",
                "units": "signal unit",
                "source_stage": "Stage_3",
                "requires_metadata": False,
                "is_dimensionless": False,
                "undefined_handling": "null with reason",
            }
        }
    }
    entries = stage_scoring._schema_entries(mapping_schema)
    assert entries[0]["name"] == "rms"
    assert stage_scoring._schema_entries({"features": [1, {"name": "rms"}]}) == [{"name": "rms"}]
    assert stage_scoring._schema_entries({}) == []

    _, stage4 = _record(tmp_path / "schema", "Stage_4")
    _write_json(stage4 / "feature_schema.json", mapping_schema)
    assert stage_scoring._schema_documented(stage4 / "feature_schema.json") is True
    assert stage_scoring._physical_traceability("Stage_4", [stage4])[0] is True

    _, stage2 = _record(tmp_path / "metadata", "Stage_2")
    _write_json(stage2 / "stage_2_config.json", {"use_physics_gating": False})
    assert stage_scoring._metadata_validated([stage2]) is True
    _write_json(stage2 / "stage_2_config.json", {"use_physics_gating": True})
    assert stage_scoring._metadata_validated([stage2]) is False
    _write_json(
        stage2 / "stage_2_config.json",
        {"use_physics_gating": True, "rpm": 600.0, "tooth_count": 2},
    )
    assert stage_scoring._metadata_validated([stage2]) is True

    assert (
        stage_scoring._stage_input_validated("Stage_1", {"input_validation": {"n_invalid": 0}}, [])[
            0
        ]
        is True
    )
    assert (
        stage_scoring._stage_input_validated("Stage_1", {"input_validation": {"n_invalid": 1}}, [])[
            0
        ]
        is False
    )
    _, stage3 = _record(tmp_path / "stage3_config", "Stage_3")
    _write_json(stage3 / "stage_3_config.json", {"wavelet_name": "db4"})
    assert stage_scoring._stage_input_validated("Stage_3", {}, [stage3])[0] is False


def test_shape_traceability_visual_and_scope_failures_are_traceable(tmp_path: Path) -> None:
    run_dir, stage1 = _record(tmp_path / "shape", "Stage_1")
    (stage1 / "broken.npz").write_bytes(b"bad archive")
    assert "unreadable NPZ" in stage_scoring._shape_validation("Stage_1", [stage1])[1]
    (stage1 / "broken.npz").unlink()
    np.savez_compressed(stage1 / "empty.npz", signal=np.asarray([]))
    assert "empty array" in stage_scoring._shape_validation("Stage_1", [stage1])[1]
    (stage1 / "empty.npz").unlink()
    assert "no arrays" in stage_scoring._shape_validation("Stage_1", [stage1])[1]
    assert stage_scoring._shape_validation("Stage_4", [stage1])[0] is True

    np.savez_compressed(stage1 / "decomposition.npz", signal=np.ones(8))
    assert stage_scoring._physical_traceability("Stage_1", [stage1])[0] is False
    _, stage2 = _record(tmp_path / "pair", "Stage_2")
    np.savez_compressed(stage2 / "weighted_reconstruction_scaled.npz", signal=np.ones(8))
    np.savez_compressed(stage2 / "weighted_reconstruction_physical.npz", signal=np.ones(7))
    assert stage_scoring._physical_traceability("Stage_2", [stage2])[0] is False

    _write_json(stage1 / "stage_1_metrics.json", {"residual_present": False})
    visual_outcomes = stage_scoring._visual_outcomes("Stage_1", [stage1], {})
    assert all(outcome[0] != "visual_residual" for outcome in visual_outcomes)
    label_outcomes = stage_scoring._visual_outcomes(
        "Stage_4",
        [stage1],
        {"stage_evidence": {"Stage_4": {"labels_available": True}}},
    )
    assert any(outcome[0] == "visual_aggregate_label_groups" for outcome in label_outcomes)

    (run_dir / "Stage_5").mkdir()
    (run_dir / "chatter_probability.csv").write_text("value\n0.5\n", encoding="utf-8")
    scoped, evidence = stage_scoring._no_out_of_scope_outputs(run_dir)
    assert scoped is False
    assert "Stage_5" in evidence
    assert "chatter_probability.csv" in evidence


def test_explicit_integrity_caps_missing_run_and_plotting_branches(tmp_path: Path) -> None:
    manifest = {
        "stage_evidence": {
            "Stage_1": {
                "known_p0_issue": True,
                "fabricated_metrics": True,
                "multiple_active_implementations": True,
                "tests": {"integration": False},
            }
        }
    }
    assert stage_scoring._explicit_cap_flags(manifest, "Stage_1") == [
        "known P0 correctness issue",
        "fabricated metrics",
        "multiple active implementations",
        "failing integration test",
    ]
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        stage_scoring.calculate_stage_scorecard(tmp_path / "missing")

    payload = {
        stage: {
            "score": score,
            "passed": ["ok"],
            "failed": [f"failure_{index}" for index in range(failed_count)],
        }
        for stage, score, failed_count in zip(
            stage_scoring.STAGES,
            (95.0, 80.0, 50.0, 0.0),
            (0, 3, 9, 1),
        )
    }
    scorecard = tmp_path / "scorecard.png"
    progress = tmp_path / "progress.png"
    stage_scoring._plot_scorecard(payload, scorecard)
    stage_scoring._plot_progress(payload, progress)
    assert scorecard.stat().st_size > 1_000
    assert progress.stat().st_size > 1_000
