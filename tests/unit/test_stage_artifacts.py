"""Typed Stage 2--4 artifact-writer integration and contract tests."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pg_amcd.denoising import wavelet_denoise_with_diagnostics
from pg_amcd.features import extract_window_feature_result
from pg_amcd.models import (
    PipelineResult,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
)
from pg_amcd.stage_artifacts import write_aggregate_stage_4, write_recording_artifacts
from pg_amcd.stage_scoring import (
    PER_RECORDING_ARTIFACTS,
    STAGE_4_AGGREGATE_ARTIFACTS,
    STAGE_4_AGGREGATE_VISUALS,
    VISUAL_REQUIREMENTS,
    _visual_present,
)
from pg_amcd.weighting import (
    analyze_physics_guided_weighting,
    restore_physical_units,
    summarize_gate_stability,
)


PHYSICS_CONFIG = {
    "maiw": {"chatter_band_center": 125.0, "chatter_band_spread": 30.0},
    "physics_gating": {
        "chatter_energy_weight": 4.0,
        "correlation_weight": 2.0,
        "kurtosis_weight": 1.0,
        "frequency_proximity_weight": 1.0,
        "harmonic_penalty": 5.0,
        "offset": 1.5,
        "harmonic_tolerance_hz": 3.0,
        "harmonic_count": 5,
        "kurtosis_scale": 10.0,
    },
}


def _pipeline_result(recording_id: str = "typed-sample") -> PipelineResult:
    fs = 1_000.0
    sample_count = 512
    time = np.arange(sample_count, dtype=float) / fs
    scaled_imfs = np.vstack(
        (
            np.sin(2.0 * np.pi * 125.0 * time),
            0.65 * np.sin(2.0 * np.pi * 40.0 * time),
            0.12 * np.sin(2.0 * np.pi * 260.0 * time),
        )
    )
    scaled_residual = np.zeros(sample_count, dtype=float)
    scaled_source = np.sum(scaled_imfs, axis=0)
    scale_factor = 2.5
    physical_imfs = scaled_imfs * scale_factor
    physical_source = scaled_source * scale_factor

    stage_1 = Stage1Output(
        time=time,
        raw_signal=physical_source,
        preprocessed_physical=physical_source,
        preprocessed_scaled=scaled_source,
        segment_time=time,
        segment_raw=physical_source,
        segment_physical=physical_source,
        segment_scaled=scaled_source,
        imfs_scaled=scaled_imfs,
        residual_scaled=scaled_residual,
        imfs_physical=physical_imfs,
        residual_physical=scaled_residual * scale_factor,
        start_index=0,
        end_index=sample_count,
        sampling_rate=fs,
        scale_factor=scale_factor,
        selected_cutoff=10.0,
        random_seed=42,
        ceemdan_parameters={"trials": 4, "noise_width": 0.02},
        cutoff_search=[
            {"cutoff_hz": 5.0, "final_score": 0.2},
            {"cutoff_hz": 10.0, "final_score": 0.1},
        ],
        imf_metrics=[
            {
                "imf_index": index + 1,
                "energy_percentage": float(
                    100.0
                    * np.sum(np.square(imf))
                    / np.sum(np.square(scaled_imfs))
                ),
                "centre_frequency_hz": centre,
                "bandwidth_hz": 3.0,
                "spectral_entropy": 0.1 + 0.05 * index,
            }
            for index, (imf, centre) in enumerate(
                zip(scaled_imfs, (125.0, 40.0, 260.0))
            )
        ],
        seed_stability={
            "imf_count_mismatch_fraction": 0.0,
            "centre_frequency_instability": 0.01,
            "energy_distribution_l1": 0.02,
            "spectral_overlap_standard_deviation": 0.01,
            "matched_imf_correlation_mean": 0.99,
        },
        metrics={
            "number_of_imfs": 3,
            "reconstruction_nrmse": 0.0,
            "absolute_orthogonality_index": 0.01,
            "signed_orthogonality_index": 0.0,
            "mean_adjacent_imf_correlation": 0.0,
            "maximum_adjacent_imf_correlation": 0.0,
            "spectral_overlap": 0.01,
            "frequency_ordering_score": 1.0,
            "runtime_seconds": 0.01,
        },
        runtime_seconds=0.01,
    )

    typed_stage_2 = analyze_physics_guided_weighting(
        np.vstack((scaled_imfs, scaled_residual)),
        scaled_source,
        fs,
        {
            "rpm": 600.0,
            "tooth_count": 4,
            "stickout": 2.5,
            "depth_of_cut": 0.5,
            "feed_rate": 120.0,
            "tool_identifier": "T-1",
        },
        PHYSICS_CONFIG,
    )
    stability = summarize_gate_stability(
        [
            typed_stage_2.gates,
            np.clip(typed_stage_2.gates + np.array([0.01, -0.01, 0.005]), 0.0, 1.0),
        ]
    )
    stage_2_metrics = typed_stage_2.metrics.to_dict()
    stage_2_metrics["gate_vector_stability"] = stability
    stage_2_metrics["selected_imf_consistency"] = stability[
        "selected_imf_consistency"
    ]
    stage_2 = Stage2Output(
        indicators=[indicator.to_dict() for indicator in typed_stage_2.indicators],
        gates=typed_stage_2.gates,
        weighted_scaled=typed_stage_2.reconstructed_scaled,
        weighted_physical=restore_physical_units(
            typed_stage_2.reconstructed_scaled, scale_factor
        ),
        metadata=typed_stage_2.metadata.to_dict(),
        metrics=stage_2_metrics,
        config={
            **typed_stage_2.coefficients,
            "selection_threshold": 0.5,
            "include_residual": False,
            "chatter_band_center": 125.0,
            "chatter_band_spread": 30.0,
        },
        runtime_seconds=typed_stage_2.metrics.reconstruction_runtime_seconds,
    )

    typed_stage_3 = wavelet_denoise_with_diagnostics(
        stage_2.weighted_scaled,
        wavelet_name="db4",
        level=3,
        fs=fs,
        chatter_center=125.0,
        chatter_spread=30.0,
        chatter_threshold_scale=0.5,
        noise_threshold_scale=1.5,
        minimum_noise_sigma=1e-8,
        clean_reference=stage_2.weighted_scaled,
    )
    coefficient_by_name = {
        f"cA_{typed_stage_3.applied_level}": typed_stage_3.approximation_coefficients,
        **typed_stage_3.detail_coefficients,
    }
    stage_3_metrics = typed_stage_3.metrics.to_dict()
    stage_3_metrics["resolved_level"] = typed_stage_3.applied_level
    stage_3 = Stage3Output(
        coefficients=[
            coefficient_by_name[row.coefficient_name]
            for row in typed_stage_3.level_diagnostics
        ],
        threshold_rows=[row.to_dict() for row in typed_stage_3.level_diagnostics],
        denoised_scaled=typed_stage_3.denoised_signal,
        denoised_physical=restore_physical_units(
            typed_stage_3.denoised_signal, scale_factor
        ),
        metrics=stage_3_metrics,
        config={
            "wavelet_name": typed_stage_3.wavelet_name,
            "level": typed_stage_3.applied_level,
            "threshold_mode": typed_stage_3.threshold_mode,
            "chatter_threshold_scale": 0.5,
            "noise_threshold_scale": 1.5,
            "minimum_noise_sigma": 1e-8,
            "band_aware": typed_stage_3.band_aware,
            "chatter_center": 125.0,
            "chatter_spread": 30.0,
        },
        runtime_seconds=typed_stage_3.metrics.runtime_seconds,
        synthetic_signals={
            "clean": stage_2.weighted_scaled,
            "noisy": stage_2.weighted_scaled,
            "recovered": typed_stage_3.denoised_signal,
        },
    )

    all_imfs_physical = np.vstack((physical_imfs, np.zeros(sample_count)))
    valid_features = extract_window_feature_result(
        physical_source,
        physical_source,
        stage_3.denoised_physical,
        all_imfs_physical,
        fs,
        600.0,
        4,
        chatter_center=125.0,
        chatter_spread=30.0,
        imf_gates=stage_2.gates,
        wavelet_name="db4",
        wavelet_level=3,
    )
    undefined_features = extract_window_feature_result(
        physical_source,
        physical_source,
        stage_3.denoised_physical,
        all_imfs_physical,
        fs,
        None,
        None,
        chatter_center=125.0,
        chatter_spread=30.0,
        imf_gates=stage_2.gates,
        wavelet_name="db4",
        wavelet_level=3,
    )
    typed_features = (valid_features, undefined_features)
    feature_rows = [
        {
            "window_index": index,
            "window_start_seconds": float(index),
            "window_end_seconds": float(index + 1),
            **feature_result.values,
        }
        for index, feature_result in enumerate(typed_features)
    ]
    feature_records = [
        {
            "window_index": index,
            "window_start_seconds": float(index),
            "window_end_seconds": float(index + 1),
            **feature_result.to_dict(),
        }
        for index, feature_result in enumerate(typed_features)
    ]
    undefined_reasons = {
        str(index): feature_result.undefined_reasons
        for index, feature_result in enumerate(typed_features)
        if feature_result.undefined_reasons
    }
    stage_4 = Stage4Output(
        feature_rows=feature_rows,
        feature_records=feature_records,
        feature_schema=valid_features.schema_dict(),
        feature_quality={
            "windows": [feature_result.quality for feature_result in typed_features],
            "undefined_reasons_by_window": undefined_reasons,
            "undefined_feature_values": sum(
                len(feature_result.undefined_reasons)
                for feature_result in typed_features
            ),
        },
        metrics={
            "window_count": len(feature_rows),
            "defined_feature_values": sum(
                feature_result.quality["defined_feature_count"]
                for feature_result in typed_features
            ),
            "undefined_feature_values": sum(
                feature_result.quality["undefined_feature_count"]
                for feature_result in typed_features
            ),
            "runtime_seconds": 0.01,
        },
        config={"window_seconds": 1.0, "window_overlap": 0.0},
        runtime_seconds=0.01,
    )

    return PipelineResult(
        raw_signal=physical_source,
        physical_preprocessed_signal=physical_source,
        scaled_preprocessed_signal=scaled_source,
        window_results=[],
        sampling_rate=fs,
        scale_factors={"physical": scale_factor},
        selected_parameters={},
        warnings=[],
        recording_id=recording_id,
        input_path=f"data/{recording_id}.mat",
        metadata={
            "rpm": 600.0,
            "tooth_count": 4,
            "stickout": 2.5,
            "depth_of_cut": 0.5,
            "feed_rate": 120.0,
            "label": "stable",
        },
        stage_1=stage_1,
        stage_2=stage_2,
        stage_3=stage_3,
        stage_4=stage_4,
    )


@pytest.fixture(scope="module")
def artifact_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, PipelineResult]:
    run_dir = tmp_path_factory.mktemp("typed-artifacts")
    result = _pipeline_result()
    write_recording_artifacts(
        run_dir,
        result,
        {"output": {"write_svg": True, "png_dpi": 80}},
    )
    second = replace(
        result,
        recording_id="typed-sample-2",
        input_path="data/typed-sample-2.mat",
        metadata={**result.metadata, "rpm": 720.0, "label": "chatter"},
    )
    write_aggregate_stage_4(
        run_dir,
        [result, second],
        {"output": {"write_svg": True, "png_dpi": 80}},
    )
    return run_dir, result


def _assert_json_finite_or_null(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_json_finite_or_null(child)
    elif isinstance(value, list):
        for child in value:
            _assert_json_finite_or_null(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_required_stage_2_to_4_artifacts_and_visuals_are_nonempty(
    artifact_run: tuple[Path, PipelineResult],
) -> None:
    run_dir, result = artifact_run
    for stage_name in ("Stage_2", "Stage_3", "Stage_4"):
        recording_dir = run_dir / stage_name / result.recording_id
        for filename in PER_RECORDING_ARTIFACTS[stage_name]:
            path = recording_dir / filename
            assert path.is_file(), path
            assert path.stat().st_size > 0, path
        for _visual_id, aliases in VISUAL_REQUIREMENTS[stage_name]:
            assert _visual_present(recording_dir, aliases), aliases
        pngs = sorted(recording_dir.glob("*.png"))
        assert len(pngs) == len(VISUAL_REQUIREMENTS[stage_name])
        assert all(path.stat().st_size > 1_000 for path in pngs)
        assert all(path.with_suffix(".svg").stat().st_size > 1_000 for path in pngs)


def test_harmonic_overlap_diagnostics_visual_is_written(
    artifact_run: tuple[Path, PipelineResult],
) -> None:
    run_dir, result = artifact_run
    stage_2_dir = run_dir / "Stage_2" / result.recording_id
    assert _visual_present(stage_2_dir, ("harmonic_overlap_diagnostics",))
    png = stage_2_dir / "11b_harmonic_overlap_diagnostics.png"
    svg = stage_2_dir / "11b_harmonic_overlap_diagnostics.svg"
    assert png.is_file() and png.stat().st_size > 1_000
    assert svg.is_file() and svg.stat().st_size > 1_000


def test_typed_indicator_gate_and_wavelet_diagnostics_are_mapped_losslessly(
    artifact_run: tuple[Path, PipelineResult],
) -> None:
    run_dir, result = artifact_run
    stage_2_dir = run_dir / "Stage_2" / result.recording_id
    indicators = pd.read_csv(stage_2_dir / "imf_indicators.csv")
    assert np.allclose(indicators["kurtosis_normalized"], indicators["kurtosis_score"])
    assert np.allclose(indicators["source_correlation"], indicators["correlation"])
    stage_2_config = json.loads(
        (stage_2_dir / "stage_2_config.json").read_text(encoding="utf-8")
    )
    assert stage_2_config["use_physics_gating"] is True
    assert stage_2_config["physics_metadata"]["rpm"] == 600.0

    threshold_rows = pd.read_csv(
        run_dir / "Stage_3" / result.recording_id / "wavelet_thresholds.csv"
    )
    approximation = threshold_rows.loc[threshold_rows["is_approximation"]]
    details = threshold_rows.loc[~threshold_rows["is_approximation"]]
    assert len(approximation) == 1
    assert not bool(approximation.iloc[0]["is_detail"])
    assert details["is_detail"].all()
    assert np.allclose(threshold_rows["coefficient_energy"], threshold_rows["input_energy"])
    assert np.allclose(
        threshold_rows["chatter_band_overlap_fraction"],
        threshold_rows["chatter_overlap_fraction"],
    )
    assert threshold_rows["energy_ratio"].sum() == pytest.approx(1.0)


def test_json_and_npz_outputs_are_finite_with_explicit_null_reasons(
    artifact_run: tuple[Path, PipelineResult],
) -> None:
    run_dir, result = artifact_run
    for stage_name in ("Stage_2", "Stage_3", "Stage_4"):
        recording_dir = run_dir / stage_name / result.recording_id
        for path in recording_dir.glob("*.json"):
            raw = path.read_text(encoding="utf-8")
            assert "NaN" not in raw and "Infinity" not in raw
            _assert_json_finite_or_null(json.loads(raw))
        for path in recording_dir.glob("*.npz"):
            with np.load(path, allow_pickle=False) as payload:
                for key in payload.files:
                    array = np.asarray(payload[key])
                    assert array.size > 0, (path, key)
                    if np.issubdtype(array.dtype, np.number):
                        assert np.all(np.isfinite(array)), (path, key)

    stage_4_dir = run_dir / "Stage_4" / result.recording_id
    records = json.loads((stage_4_dir / "window_features.json").read_text())
    assert any(
        feature_value is None
        for record in records
        for feature_value in record["features"].values()
    )
    quality = json.loads((stage_4_dir / "feature_quality.json").read_text())
    assert quality["undefined_feature_values"] > 0
    assert quality["undefined_reasons_by_window"]


def test_aggregate_contract_names_and_figures_match_the_scorer(
    artifact_run: tuple[Path, PipelineResult],
) -> None:
    run_dir, _ = artifact_run
    aggregate = run_dir / "Stage_4" / "aggregate"
    for filename in STAGE_4_AGGREGATE_ARTIFACTS:
        path = aggregate / filename
        assert path.is_file(), path
        assert path.stat().st_size > 0, path
    for _visual_id, aliases in STAGE_4_AGGREGATE_VISUALS:
        assert _visual_present(aggregate, aliases), aliases
    label_figure = aggregate / "aggregate_features_grouped_by_label.png"
    assert label_figure.stat().st_size > 1_000
    assert label_figure.with_suffix(".svg").stat().st_size > 1_000


_SEED_STABILITY_FIGURE = "13b_seed_stability_per_imf.png"
_ADJACENT_OVERLAP_FIGURE = "13c_adjacent_overlap_diagnostics.png"


def test_stage_1_seed_stability_and_adjacent_overlap_figures_are_emitted(
    tmp_path: Path,
) -> None:
    """Seed-stability and adjacent-overlap diagnostics render in Stage 1 artifacts."""

    result = _pipeline_result("stability-sample")
    # Inject per-IMF seed-stability diagnostics.
    per_seed_stability = {
        "imf_count_mismatch_fraction": 0.0,
        "centre_frequency_instability": 0.01,
        "energy_distribution_l1": 0.02,
        "spectral_overlap_standard_deviation": 0.01,
        "matched_imf_correlation_mean": 0.99,
        "per_imf_centre_frequency": [
            [125.0, 126.0],
            [40.0, 41.0],
            [260.0, 259.0],
        ],
        "per_imf_energy_percentage": [
            [30.0, 31.0],
            [20.0, 20.0],
            [10.0, 10.0],
        ],
        "per_imf_matched_correlation": [0.99, 0.95, 0.88],
    }
    stage_1 = replace(result.stage_1, seed_stability=per_seed_stability)
    # Inject adjacent-overlap values for the first (count - 1) IMFs.
    imf_metrics = [dict(row) for row in stage_1.imf_metrics]
    for index, row in enumerate(imf_metrics[:-1]):
        row["adjacent_spectral_overlap"] = 0.05 + 0.01 * index
    stage_1 = replace(stage_1, imf_metrics=imf_metrics)
    result = replace(result, stage_1=stage_1)

    run_dir = tmp_path / "stability-run"
    write_recording_artifacts(
        run_dir,
        result,
        {"output": {"write_svg": True, "png_dpi": 80}},
    )

    stage_1_dir = run_dir / "Stage_1" / result.recording_id
    expected_figures = (_SEED_STABILITY_FIGURE, _ADJACENT_OVERLAP_FIGURE)
    for filename in expected_figures:
        png_path = stage_1_dir / filename
        assert png_path.is_file(), png_path
        assert png_path.stat().st_size > 1_000, png_path
        svg_path = png_path.with_suffix(".svg")
        assert svg_path.is_file(), svg_path
        assert svg_path.stat().st_size > 1_000, svg_path

    # The old direct-save filename should not exist; figures route through the standard list.
    for old_suffix in (".png", ".svg"):
        old_overlap_filename = f"13b_adjacent_overlap_diagnostics{old_suffix}"
        assert not (stage_1_dir / old_overlap_filename).exists()
