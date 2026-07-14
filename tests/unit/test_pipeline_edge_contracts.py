"""Validation and numerical edge contracts for the Stage 1--4 integration layer."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pg_amcd import pipeline


def _make_signal(fs: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(int(fs)) / fs
    signal = np.sin(2 * np.pi * 50 * time) + 0.25 * np.sin(2 * np.pi * 120 * time)
    return time, signal


def _minimal_config(cutoffs: list[float]) -> dict:
    return {
        "sampling_rate": 1000.0,
        "segment_points": 1000,
        "use_physics_gating": False,
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
        "physics_gating": {},
        "wavelet": {
            "wavelet_name": "db4",
            "level": 3,
            "threshold_mode": "soft",
            "minimum_noise_sigma": 1e-8,
        },
        "features": {"window_seconds": 1.0, "overlap_ratio": 0.75},
    }


def _direct_config() -> dict:
    return {
        "sampling_rate": 1000.0,
        "validation": {
            "sampling_rate_tolerance": 0.05,
            "timestamp_jitter_tolerance": 0.05,
            "minimum_duration_seconds": 0.0,
        },
        "preprocessing": {"low_pass_cutoff_hz": None},
    }


@pytest.mark.parametrize(
    ("time", "signal", "config", "message"),
    [
        (np.arange(4) + 0j, np.arange(4), _direct_config(), "real-valued"),
        (np.arange(4)[:, None], np.arange(4), _direct_config(), "one-dimensional"),
        (np.arange(4), np.arange(5), _direct_config(), "identical lengths"),
        (np.arange(2), np.arange(2), _direct_config(), "at least three"),
        (np.array([0.0, 0.001, np.nan]), np.arange(3), _direct_config(), "finite"),
        (np.arange(4) / 1000.0, np.ones(4), _direct_config(), "constant"),
        (
            np.arange(4) / 1000.0,
            np.arange(4),
            {**_direct_config(), "sampling_rate": None},
            "sampling_rate",
        ),
        (np.array([0.0, 0.001, 0.001]), np.arange(3), _direct_config(), "strictly increasing"),
    ],
)
def test_direct_array_validation_rejects_invalid_inputs(time, signal, config, message):
    with pytest.raises(ValueError, match=message):
        pipeline._validated_input_arrays(time, signal, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("sampling_rate", 1200.0), "differs from configured"),
        (("sampling_rate_tolerance", 1.0), "sampling_rate_tolerance"),
        (("timestamp_jitter_tolerance", -0.1), "timestamp_jitter_tolerance"),
        (("minimum_duration_seconds", -1.0), "minimum_duration_seconds"),
        (("minimum_duration_seconds", 2.0), "shorter"),
    ],
)
def test_direct_array_validation_rejects_invalid_timing_contracts(mutation, message):
    config = _direct_config()
    key, value = mutation
    if key == "sampling_rate":
        config[key] = value
    else:
        config["validation"][key] = value
    time = np.arange(1000) / 1000.0
    signal = np.sin(2 * np.pi * 30 * time)

    with pytest.raises(ValueError, match=message):
        pipeline._validated_input_arrays(time, signal, config)


def test_direct_array_validation_rejects_local_timestamp_jitter() -> None:
    time = np.arange(1000) / 1000.0
    time[500:] += 0.0002
    signal = np.sin(2 * np.pi * 30 * time)

    with pytest.raises(ValueError, match="Timestamp jitter"):
        pipeline._validated_input_arrays(time, signal, _direct_config())


def test_mapping_positive_and_preprocessing_helpers_are_strict() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        pipeline._mapping_section({"wavelet": []}, "wavelet")
    for value in (None, True, "bad", 0.0, np.inf):
        with pytest.raises(ValueError, match="finite positive"):
            pipeline._finite_positive(value, "value")

    automatic = pipeline._resolved_preprocessing(_direct_config(), 1000.0)
    assert automatic["lowpass_cutoff_hz"] == 490.0
    assert automatic["lowpass_resolution"].startswith("automatic")
    configured = deepcopy(_direct_config())
    configured["preprocessing"]["low_pass_cutoff_hz"] = 400.0
    assert (
        pipeline._resolved_preprocessing(configured, 1000.0)["lowpass_resolution"] == "configured"
    )
    configured["preprocessing"]["low_pass_cutoff_hz"] = 500.0
    with pytest.raises(ValueError, match="Nyquist"):
        pipeline._resolved_preprocessing(configured, 1000.0)


def test_gate_stability_seed_and_correlation_helpers_cover_edge_semantics() -> None:
    assert pipeline._gate_stability_seeds({"stability_seeds": [5, 7]}) == [5, 7]
    assert pipeline._gate_stability_seeds({"noise_seed": 10, "search_seeds": 2}) == [10, 11]
    with pytest.raises(ValueError, match="sequence"):
        pipeline._gate_stability_seeds({"stability_seeds": "5,7"})
    with pytest.raises(ValueError, match="non-empty and unique"):
        pipeline._gate_stability_seeds({"stability_seeds": [5, 5]})

    constant = np.ones(32)
    oscillation = np.sin(np.linspace(0, 4 * np.pi, 32))
    assert pipeline._safe_absolute_correlation(constant, oscillation) == 0.0
    assert pipeline._safe_absolute_correlation(oscillation, oscillation) == pytest.approx(1.0)


def test_legacy_metric_and_wavelet_row_helpers_handle_zero_energy() -> None:
    metrics = pipeline._legacy_weighting_metrics(
        np.zeros(128), np.zeros(128), 1000.0, 125.0, 30.0, 0.01
    )
    assert metrics["chatter_band_retention"] == 0.0
    assert metrics["out_of_band_attenuation"] == 1.0
    assert metrics["spectral_distortion"] == 0.0

    diagnostics = [
        SimpleNamespace(
            input_energy=0.0,
            is_approximation=True,
            chatter_overlap_fraction=0.0,
            to_dict=lambda: {"coefficient_name": "cA_1"},
        ),
        SimpleNamespace(
            input_energy=0.0,
            is_approximation=False,
            chatter_overlap_fraction=0.5,
            to_dict=lambda: {"coefficient_name": "cD_1"},
        ),
    ]
    rows = pipeline._threshold_rows(SimpleNamespace(level_diagnostics=diagnostics))
    assert rows[0]["is_detail"] is False
    assert rows[1]["is_detail"] is True
    assert all(row["energy_ratio"] == 0.0 for row in rows)


def test_process_recording_rejects_invalid_integration_configuration_before_ceemdan() -> None:
    time, signal = _make_signal()
    with pytest.raises(ValueError, match="config must be a mapping"):
        pipeline.process_recording(time, signal, [])
    with pytest.raises(ValueError, match="mode must be"):
        pipeline.process_recording(time, signal, _minimal_config([20.0]), mode="decision")

    string_cutoffs = _minimal_config([20.0])
    string_cutoffs["ceemdan"]["search_cutoffs"] = "20"
    with pytest.raises(ValueError, match="non-empty sequence"):
        pipeline.process_recording(time, signal, string_cutoffs)

    empty_cutoffs = _minimal_config([20.0])
    empty_cutoffs["ceemdan"]["search_cutoffs"] = []
    with pytest.raises(ValueError, match="must be non-empty"):
        pipeline.process_recording(time, signal, empty_cutoffs)

    boolean_segment = _minimal_config([20.0])
    boolean_segment["segment_points"] = True
    with pytest.raises(ValueError, match="positive integer"):
        pipeline.process_recording(time, signal, boolean_segment)

    fractional_segment = _minimal_config([20.0])
    fractional_segment["segment_points"] = 10.5
    with pytest.raises(ValueError, match="integer of at least 3"):
        pipeline.process_recording(time, signal, fractional_segment)
