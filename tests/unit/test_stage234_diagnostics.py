"""Acceptance-focused scientific-core tests for Stages 2--4."""

import numpy as np
import pytest

from pg_amcd.denoising import (
    bayes_shrink_threshold,
    select_best_wavelet,
    wavelet_denoise_with_diagnostics,
)
from pg_amcd.features import (
    FEATURE_SCHEMA_VERSION,
    aggregate_feature_results,
    extract_sliding_window_features,
    extract_window_feature_result,
    feature_schema,
)
from pg_amcd.weighting import (
    PhysicsMetadata,
    analyze_physics_guided_weighting,
    reconstruct_gated_signal,
    restore_physical_units,
    summarize_gate_stability,
    validate_physics_metadata,
)


PHYSICS_CONFIG = {
    "maiw": {
        "chatter_band_center": 125.0,
        "chatter_band_spread": 30.0,
    },
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


def _physics_components(fs=1000.0, n=2000):
    time = np.arange(n) / fs
    chatter = np.sin(2 * np.pi * 125.0 * time)
    forced = np.sin(2 * np.pi * 40.0 * time)
    broadband = 0.1 * np.sin(2 * np.pi * 260.0 * time)
    residual = np.zeros(n)
    imfs = np.vstack((chatter, forced, broadband, residual))
    return time, imfs, np.sum(imfs, axis=0)


def test_stage2_typed_result_has_every_indicator_and_explicit_residual():
    _, imfs, source = _physics_components()
    result = analyze_physics_guided_weighting(
        imfs,
        source,
        1000.0,
        {
            "rpm": 600.0,
            "tooth_count": 4,
            "stickout": 2.5,
            "depth_of_cut": 0.5,
            "feed_rate": 120.0,
            "tool_id": "T-1",
        },
        PHYSICS_CONFIG,
    )

    assert result.residual_policy == "last_row"
    assert result.residual_excluded is True
    assert result.physical_imf_count == imfs.shape[0] - 1
    assert result.gates.shape == (3,)
    assert np.all((result.gates >= 0.0) & (result.gates <= 1.0))
    assert result.metadata.tool_identifier == "T-1"
    assert result.indicators[0].gate > result.indicators[1].gate
    for row in result.indicators:
        values = np.array(
            [
                row.correlation,
                row.relative_energy,
                row.kurtosis,
                row.kurtosis_score,
                row.chatter_band_energy_ratio,
                row.spindle_harmonic_energy_ratio,
                row.tooth_harmonic_energy_ratio,
                row.forced_harmonic_energy_ratio,
                row.frequency_proximity,
                row.centre_frequency_hz,
                row.bandwidth_hz,
                row.spectral_entropy,
                row.gate,
            ]
        )
        assert np.all(np.isfinite(values))
    assert result.metrics.chatter_band_retention > 0.0
    assert np.isfinite(result.metrics.spectral_distortion)


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"tooth_count": 4}, "rpm"),
        ({"rpm": 600.0}, "tooth_count"),
        ({"rpm": "bad", "tooth_count": 4}, "rpm"),
        ({"rpm": 600.0, "tooth_count": 1.5}, "tooth_count"),
    ],
)
def test_stage2_metadata_never_falls_back(metadata, message):
    with pytest.raises(ValueError, match=message):
        validate_physics_metadata(metadata)


def test_stage2_strict_config_and_gate_stability_validation():
    _, imfs, source = _physics_components()
    with pytest.raises(ValueError, match="physics_gating"):
        analyze_physics_guided_weighting(
            imfs,
            source,
            1000.0,
            {"rpm": 600.0, "tooth_count": 4},
            {"maiw": PHYSICS_CONFIG["maiw"]},
        )
    stability = summarize_gate_stability(
        [np.array([0.9, 0.1, 0.7]), np.array([0.8, 0.2, 0.6])]
    )
    assert stability["selected_imf_consistency"] == 1.0
    assert stability["selected_count_by_seed"] == [2, 2]


def test_stage2_serialization_zero_energy_and_no_residual_policy():
    zero_imf = np.zeros((1, 128))
    result = analyze_physics_guided_weighting(
        zero_imf,
        np.zeros(128),
        1000.0,
        PhysicsMetadata(rpm=600.0, tooth_count=4),
        PHYSICS_CONFIG,
        residual_policy="none",
    )
    payload = result.to_dict(include_signal=True)
    assert payload["physical_imf_count"] == 1
    assert payload["residual_excluded"] is False
    assert len(payload["reconstructed_scaled"]) == 128
    assert result.indicators[0].relative_energy == 0.0
    assert result.indicators[0].spectral_entropy == 0.0
    assert result.metrics.spectral_distortion == 0.0


def test_stage2_scientific_input_validation_edges():
    _, imfs, source = _physics_components(n=128)
    metadata = {"rpm": 600.0, "tooth_count": 4}
    bad_calls = [
        (imfs, source[:, None], 1000.0, "last_row"),
        (imfs[:, :1], source[:1], 1000.0, "last_row"),
        (np.where(np.arange(imfs.size).reshape(imfs.shape) == 0, np.nan, imfs), source, 1000.0, "last_row"),
        (imfs, np.where(np.arange(source.size) == 0, np.nan, source), 1000.0, "last_row"),
        (imfs, source, 1000.0, "invalid"),
        (imfs[:1], source, 1000.0, "last_row"),
    ]
    for bad_imfs, bad_source, bad_fs, residual_policy in bad_calls:
        with pytest.raises(ValueError):
            analyze_physics_guided_weighting(
                bad_imfs,
                bad_source,
                bad_fs,
                metadata,
                PHYSICS_CONFIG,
                residual_policy=residual_policy,  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="mapping"):
        validate_physics_metadata(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stickout"):
        validate_physics_metadata({"rpm": 600, "tooth_count": 4, "stickout": np.inf})
    with pytest.raises(ValueError, match="harmonic_count"):
        malformed = {
            "maiw": PHYSICS_CONFIG["maiw"],
            "physics_gating": {**PHYSICS_CONFIG["physics_gating"], "harmonic_count": 1.5},
        }
        analyze_physics_guided_weighting(imfs, source, 1000.0, metadata, malformed)
    with pytest.raises(ValueError, match="measurable range"):
        out_of_band = {
            "maiw": {"chatter_band_center": 2000.0, "chatter_band_spread": 10.0},
            "physics_gating": PHYSICS_CONFIG["physics_gating"],
        }
        analyze_physics_guided_weighting(imfs, source, 1000.0, metadata, out_of_band)

    restored = restore_physical_units(np.ones(4), 2.0)
    np.testing.assert_array_equal(restored, np.full(4, 2.0))
    with pytest.raises(ValueError):
        restore_physical_units(np.ones((2, 2)), 2.0)
    with pytest.raises(ValueError, match="bounded"):
        reconstruct_gated_signal(imfs, np.array([1.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        summarize_gate_stability([])
    with pytest.raises(ValueError, match="non-finite"):
        summarize_gate_stability([np.array([0.5, np.nan])])


def test_stage3_diagnostics_store_coefficients_thresholds_overlap_and_length():
    fs = 1000.0
    n = 1001  # odd length exercises exact restoration
    time = np.arange(n) / fs
    clean = np.sin(2 * np.pi * 125.0 * time)
    noisy = clean + 0.25 * np.random.default_rng(4).standard_normal(n)
    result = wavelet_denoise_with_diagnostics(
        noisy,
        wavelet_name="db4",
        level=99,
        fs=fs,
        chatter_center=125.0,
        chatter_spread=50.0,
        threshold_mode="soft",
        min_noise_sigma=1e-8,
        clean_reference=clean,
    )

    assert result.requested_level == 99
    assert result.applied_level < result.requested_level
    assert result.denoised_signal.shape == noisy.shape
    assert len(result.detail_coefficients) == result.applied_level
    assert len(result.thresholded_detail_coefficients) == result.applied_level
    assert len(result.thresholds_by_level) == result.applied_level
    assert all(value >= 0.0 for value in result.thresholds_by_level.values())
    detail_rows = [row for row in result.level_diagnostics if not row.is_approximation]
    assert any(0.0 < row.chatter_overlap_fraction < 1.0 for row in detail_rows)
    assert result.metrics.synthetic_reference_rmse is not None
    assert result.metrics.synthetic_reference_snr_db is not None
    assert np.isfinite(result.metrics.synthetic_reference_rmse)


def test_stage3_validates_shape_mode_and_nonfinite_values():
    with pytest.raises(ValueError, match="one-dimensional"):
        wavelet_denoise_with_diagnostics(np.ones((2, 32)), wavelet_name="db2")
    signal = np.ones(128)
    signal[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        wavelet_denoise_with_diagnostics(signal, wavelet_name="db2")
    with pytest.raises(ValueError, match="threshold_mode"):
        wavelet_denoise_with_diagnostics(np.ones(128), wavelet_name="db2", threshold_mode="garrote")


def test_stage3_serialization_hard_unaware_and_constant_signal_branches():
    result = wavelet_denoise_with_diagnostics(
        np.zeros(128),
        wavelet_name="db2",
        level=3,
        fs=1000.0,
        chatter_center=125.0,
        chatter_spread=30.0,
        band_aware=False,
        threshold_mode="hard",
        clean_reference=np.zeros(128),
    )
    payload = result.to_dict(include_signal=True)
    assert payload["threshold_mode"] == "hard"
    assert payload["band_aware"] is False
    assert len(payload["denoised_signal"]) == 128
    assert all(row.threshold_scale == 1.0 for row in result.level_diagnostics[1:])
    assert result.metrics.correlation_before_after == 1.0
    assert result.metrics.synthetic_reference_snr_db == 0.0


def test_stage3_threshold_and_parameter_validation_edges():
    assert bayes_shrink_threshold(np.zeros(8), 0.1) == 0.0
    with pytest.raises(ValueError):
        bayes_shrink_threshold(np.ones((2, 2)), 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        bayes_shrink_threshold(np.array([0.0, np.nan]), 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        bayes_shrink_threshold(np.ones(8), -0.1)
    with pytest.raises(ValueError, match="integer"):
        wavelet_denoise_with_diagnostics(np.ones(128), wavelet_name="db2", level=2.5)
    with pytest.raises(ValueError, match="multipliers"):
        wavelet_denoise_with_diagnostics(
            np.ones(128), wavelet_name="db2", chatter_threshold_scale=-1.0
        )
    with pytest.raises(ValueError, match="measurable range"):
        wavelet_denoise_with_diagnostics(
            np.ones(128),
            wavelet_name="db2",
            fs=1000.0,
            chatter_center=2000.0,
            chatter_spread=10.0,
        )
    with pytest.raises(ValueError, match="clean_reference"):
        wavelet_denoise_with_diagnostics(
            np.ones(128), wavelet_name="db2", clean_reference=np.ones(127)
        )
    with pytest.raises(ValueError, match="candidate"):
        select_best_wavelet(np.ones(128), np.ones(128), [], 1000.0, 125.0, 30.0)


def _feature_inputs(growing=False, constant=False):
    fs = 1000.0
    n = 2000
    time = np.arange(n) / fs
    if constant:
        signal = np.zeros(n)
        imfs = np.zeros((4, n))
    else:
        envelope = np.linspace(0.1, 1.0, n) if growing else np.ones(n)
        chatter = envelope * np.sin(2 * np.pi * 125.0 * time)
        forced = 0.4 * np.sin(2 * np.pi * 40.0 * time)
        other = 0.1 * np.sin(2 * np.pi * 260.0 * time)
        signal = chatter + forced + other
        imfs = np.vstack((chatter, forced, other, np.zeros(n)))
    return fs, signal, imfs


def _feature_result(growing=False, constant=False, rpm=600.0, tooth_count=4):
    fs, signal, imfs = _feature_inputs(growing=growing, constant=constant)
    return extract_window_feature_result(
        signal,
        signal,
        signal,
        imfs,
        fs,
        rpm,
        tooth_count,
        chatter_center=125.0,
        chatter_spread=30.0,
        imf_gates=np.array([0.9, 0.1, 0.4]),
        wavelet_name="db4",
        wavelet_level=4,
    )


def test_stage4_schema_and_all_feature_families_are_present():
    result = _feature_result(growing=True)
    schema = feature_schema()
    assert schema["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert result.schema_version == FEATURE_SCHEMA_VERSION
    assert result.quality["all_defined_values_finite"] is True
    families = {definition.family for definition in result.definitions}
    assert {"time", "frequency", "imf", "wavelet", "early_chatter", "physics"} <= families
    required = {
        "time_std",
        "time_mean_absolute",
        "time_clearance_factor",
        "time_zero_crossing_rate",
        "freq_tooth_harmonic_ratio",
        "imf_count",
        "imf_1_gate",
        "imf_orthogonality_index",
        "wavelet_entropy",
        "wavelet_time_frequency_concentration",
        "early_hegr",
        "early_chatter_band_energy_growth",
        "physics_forced_vibration_energy",
        "physics_chatter_to_harmonic_energy_ratio",
    }
    assert required <= result.values.keys()
    assert result.values["physics_tooth_passing_frequency_hz"] == pytest.approx(40.0)
    assert result.values["freq_tooth_harmonic_ratio"] > 0.0
    assert set(result.values) == {definition.name for definition in result.definitions}


def test_stage4_hegr_detects_growing_hilbert_energy():
    fs = 1000.0
    time = np.arange(2000) / fs
    steady_signal = np.sin(2 * np.pi * 125.0 * time)
    growing_signal = np.linspace(0.1, 1.0, time.size) * steady_signal

    def extract(signal):
        layers = np.vstack((signal, np.zeros_like(signal), np.zeros_like(signal)))
        return extract_window_feature_result(
            signal,
            signal,
            signal,
            layers,
            fs,
            600.0,
            4,
            chatter_center=125.0,
            chatter_spread=30.0,
            imf_gates=np.array([0.9, 0.0]),
            wavelet_name="db4",
        )

    steady = extract(steady_signal)
    growing = extract(growing_signal)
    assert growing.values["early_hegr"] > steady.values["early_hegr"]
    assert growing.values["early_energy_growth_rate"] > steady.values["early_energy_growth_rate"]
    assert growing.traces["instantaneous_energy"].shape == (2000,)
    assert growing.traces["hegr_derivative"].shape == (2000,)


def test_stage4_undefined_values_are_null_with_reasons():
    result = _feature_result(constant=True, rpm=None, tooth_count=None)
    assert result.values["time_kurtosis"] is None
    assert "time_kurtosis" in result.undefined_reasons
    assert result.values["freq_centroid"] is None
    assert result.values["imf_max_energy_ratio"] is None
    assert result.values["physics_spindle_frequency_hz"] is None
    assert "RPM" in result.undefined_reasons["physics_spindle_frequency_hz"]
    assert result.quality["undefined_feature_count"] > 0


def test_stage4_sliding_windows_and_aggregate_helpers():
    fs, signal, imfs = _feature_inputs(growing=True)
    records = extract_sliding_window_features(
        signal,
        signal,
        signal,
        imfs,
        fs,
        600.0,
        4,
        window_seconds=1.0,
        overlap_ratio=0.5,
        imf_gates=np.array([0.9, 0.1, 0.4]),
        chatter_center=125.0,
        chatter_spread=30.0,
        wavelet_name="db4",
    )
    assert len(records) == 3
    assert records[1].start_index == 500
    aggregate = aggregate_feature_results(
        records,
        recording_ids=["r1", "r1", "r1"],
    )
    assert len(aggregate.rows) == 3
    assert aggregate.schema["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert "time_rms" in aggregate.summary
    assert aggregate.missingness["time_rms"]["missing_count"] == 0
    assert "time_rms" in aggregate.correlations["time_rms"]
    assert aggregate.to_dict()["schema"]["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_stage4_serialization_and_aggregate_missingness():
    defined = _feature_result(growing=True)
    undefined = _feature_result(constant=True, rpm=None, tooth_count=None)
    assert defined.schema_dict()["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert "traces" in defined.to_dict(include_traces=True)
    assert "traces" not in defined.to_dict()
    with pytest.raises(ValueError, match="undefined_fill"):
        defined.finite_values(np.nan)
    aggregate = aggregate_feature_results(
        [defined, undefined], recording_ids=["defined", "undefined"]
    )
    assert aggregate.missingness["time_kurtosis"]["missing_count"] == 1
    assert aggregate.missingness["time_kurtosis"]["reasons"]
    with pytest.raises(ValueError, match="At least one"):
        aggregate_feature_results([])
    with pytest.raises(ValueError, match="recording_ids"):
        aggregate_feature_results([defined], recording_ids=[])


def test_stage4_input_validation_edges():
    fs, signal, imfs = _feature_inputs()
    base_args = (signal, signal, signal, imfs, fs, 600.0, 4)
    with pytest.raises(ValueError, match="one-dimensional"):
        extract_window_feature_result(signal[:, None], *base_args[1:])
    with pytest.raises(ValueError, match="identical lengths"):
        extract_window_feature_result(signal[:-1], *base_args[1:])
    bad_imfs = imfs.copy()
    bad_imfs[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_window_feature_result(signal, signal, signal, bad_imfs, fs, 600.0, 4)
    with pytest.raises(ValueError, match="gate count"):
        extract_window_feature_result(*base_args, imf_gates=np.array([0.5]))
    with pytest.raises(ValueError, match="bounded"):
        extract_window_feature_result(*base_args, imf_gates=np.array([0.5, 0.5, 2.0]))
    with pytest.raises(ValueError, match="harmonic_count"):
        extract_window_feature_result(*base_args, harmonic_count=0)
    with pytest.raises(ValueError, match="overlap_ratio"):
        extract_sliding_window_features(
            signal, signal, signal, imfs, fs, 600.0, 4, overlap_ratio=1.0
        )
    with pytest.raises(ValueError, match="invalid window length"):
        extract_sliding_window_features(
            signal, signal, signal, imfs, fs, 600.0, 4, window_seconds=10.0
        )
