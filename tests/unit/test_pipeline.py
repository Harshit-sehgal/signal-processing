"""Focused tests for the canonical, decision-free Stage 1--4 integration."""

import numpy as np
import pytest

from pg_amcd.models import PipelineResult
from pg_amcd.pipeline import process_recording
from pg_amcd.weighting import reconstruct_gated_signal


def _minimal_config(cutoffs, fs=1000.0, *, use_physics=False):
    return {
        "sampling_rate": fs,
        "segment_points": 1000,
        "use_physics_gating": use_physics,
        "preprocessing": {
            "filter_order": 3,
            "low_pass_cutoff_hz": None,
            "scale_percentile": 99.5,
        },
        "ceemdan": {
            "trials": 2,
            "search_trials": 1,
            "epsilon": 0.02,
            "noise_seed": 42,
            "sifting_iterations": 2,
            "search_cutoffs": cutoffs,
            "search_seeds": 2,
            "stability_seeds": [42, 43],
            "parallel": False,
        },
        "maiw": {
            "alpha": 0.25,
            "beta": 0.25,
            "gamma": 0.25,
            "delta": 0.25,
            "chatter_band_center": 125.0,
            "chatter_band_spread": 50.0,
        },
        "physics_gating": {
            "chatter_energy_weight": 4.0,
            "correlation_weight": 2.0,
            "kurtosis_weight": 1.0,
            "frequency_proximity_weight": 1.0,
            "harmonic_penalty": 5.0,
            "offset": 1.5,
            "harmonic_tolerance_hz": 15.0,
            "harmonic_count": 5,
            "kurtosis_scale": 10.0,
            "selection_threshold": 0.5,
            "include_residual": False,
        },
        "wavelet": {
            "wavelet_name": "db4",
            "level": 3,
            "threshold_mode": "soft",
            "minimum_noise_sigma": 1e-8,
        },
        "features": {"window_seconds": 1.0, "overlap_ratio": 0.75},
    }


def _make_signal(fs=1000.0, duration=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    time = np.arange(n) / fs
    signal = (
        np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 120 * time) + rng.normal(0, 0.1, n)
    )
    return time, signal


@pytest.fixture(scope="module")
def legacy_result():
    time, signal = _make_signal()
    return process_recording(
        time,
        signal,
        _minimal_config([50.0, 150.0, 250.0], use_physics=False),
        mode="exploratory",
    )


def test_process_recording_populates_canonical_stage_outputs(legacy_result):
    result = legacy_result
    assert isinstance(result, PipelineResult)
    assert result.stage_1 is not None
    assert result.stage_2 is not None
    assert result.stage_3 is not None
    assert result.stage_4 is not None

    stage_1 = result.stage_1
    stage_2 = result.stage_2
    stage_3 = result.stage_3
    stage_4 = result.stage_4
    assert stage_1.selected_cutoff in [50.0, 150.0, 250.0]
    assert len(stage_1.cutoff_search) == 3
    assert stage_1.metrics["residual_verified"] is True
    assert stage_1.ceemdan_parameters["algorithm"] == "CEEMDAN"
    assert stage_1.seed_stability["n_seeds"] == 2
    assert "reconstruction_nrmse_standard_deviation" not in stage_1.seed_stability
    assert stage_2.config["method"] == "legacy_maiw_baseline"
    assert stage_2.config["include_residual"] is False
    assert stage_2.metrics["residual_excluded"] is True
    assert stage_3.metrics["denoising_scope"] == "reconstruction_level"
    assert stage_3.metrics["synthetic_self_check"]["known_clean_reference_used"] is True
    assert stage_4.metrics["stage_scope"] == "feature_extraction_only"
    assert stage_4.metrics["probabilities_generated"] is False
    assert stage_4.metrics["decisions_generated"] is False


def test_single_scale_factor_and_explicit_residual_are_preserved(legacy_result):
    stage_1 = legacy_result.stage_1
    stage_2 = legacy_result.stage_2
    stage_3 = legacy_result.stage_3
    assert stage_1 is not None and stage_2 is not None and stage_3 is not None

    np.testing.assert_allclose(
        stage_1.segment_scaled * stage_1.scale_factor,
        stage_1.segment_physical,
    )
    np.testing.assert_allclose(
        stage_1.imfs_scaled * stage_1.scale_factor,
        stage_1.imfs_physical,
    )
    np.testing.assert_allclose(
        stage_1.residual_scaled * stage_1.scale_factor,
        stage_1.residual_physical,
    )
    np.testing.assert_allclose(
        np.sum(stage_1.imfs_scaled, axis=0) + stage_1.residual_scaled,
        stage_1.segment_scaled,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        reconstruct_gated_signal(
            np.vstack((stage_1.imfs_scaled, stage_1.residual_scaled)),
            stage_2.gates,
        ),
        stage_2.weighted_scaled,
    )
    assert stage_2.gates.size == stage_1.imfs_scaled.shape[0]
    np.testing.assert_allclose(
        stage_2.weighted_scaled * stage_1.scale_factor,
        stage_2.weighted_physical,
    )
    np.testing.assert_allclose(
        stage_3.denoised_scaled * stage_1.scale_factor,
        stage_3.denoised_physical,
    )


def test_stage4_keeps_nulls_and_compatibility_windows_make_no_decisions(legacy_result):
    stage_4 = legacy_result.stage_4
    assert stage_4 is not None
    assert stage_4.feature_records
    canonical = stage_4.feature_records[0]
    assert canonical["features"]["physics_spindle_frequency_hz"] is None
    assert "physics_spindle_frequency_hz" in canonical["undefined_features"]
    assert stage_4.feature_rows[0]["physics_spindle_frequency_hz"] is None
    assert stage_4.feature_quality["null_policy"].startswith("Undefined canonical")

    assert legacy_result.window_results
    for window in legacy_result.window_results:
        assert np.all(np.isfinite(window.imfs))
        assert np.isnan(window.chatter_probability)
        assert window.predicted_label == "not_evaluated"
        assert np.isnan(window.confidence)
        assert all(np.isfinite(value) for value in window.features.values())


def test_physics_mode_fails_fast_without_required_metadata():
    time, signal = _make_signal()
    config = _minimal_config([50.0], use_physics=True)
    with pytest.raises(ValueError, match="rpm is required"):
        process_recording(time, signal, config, metadata=None)
    with pytest.raises(ValueError, match="tooth_count is required"):
        process_recording(time, signal, config, metadata={"rpm": 600.0})


def test_physics_mode_uses_strict_metadata_and_independent_gates():
    time, signal = _make_signal(seed=4)
    config = _minimal_config([50.0], use_physics=True)
    result = process_recording(
        time,
        signal,
        config,
        metadata={"rpm": 600.0, "tooth_count": 2, "recording_id": "synthetic-1"},
    )
    assert result.stage_2 is not None and result.stage_4 is not None
    assert result.recording_id == "synthetic-1"
    assert result.stage_2.config["method"] == "physics_guided_independent_gates"
    assert result.stage_2.metadata["rpm"] == 600.0
    assert result.stage_2.metadata["tooth_count"] == 2
    assert result.stage_2.metrics["gate_normalisation"] == "independent_not_sum_normalised"
    assert all(
        row["weighting_method"] == "physics_guided_independent_gates"
        for row in result.stage_2.indicators
    )
    assert result.stage_4.metrics["physics_metadata_valid_for_all_windows"] is True
    assert result.stage_4.feature_rows[0]["physics_spindle_frequency_hz"] == pytest.approx(10.0)
    assert result.selected_parameters["lowpass_cutoff_hz"] == pytest.approx(490.0)
