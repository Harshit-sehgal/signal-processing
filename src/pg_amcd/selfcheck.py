"""Fast, deterministic scientific self-checks recorded in each run manifest."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pg_amcd.decomposition import (
    calculate_frequency_ordering_score,
    calculate_reconstruction_nrmse,
    decompose_ceemdan,
)
from pg_amcd.denoising import wavelet_denoise_with_diagnostics
from pg_amcd.features import extract_window_feature_result, feature_schema
from pg_amcd.weighting import analyze_physics_guided_weighting


def _capture(check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        details = check()
        return {"passed": True, "details": details}
    except (RuntimeError, ValueError, AssertionError) as exc:
        return {"passed": False, "error": str(exc)}


def _stage_1_unit() -> dict[str, Any]:
    fs = 1024.0
    time = np.arange(1024) / fs
    imfs = np.vstack(
        [
            np.sin(2.0 * np.pi * 220.0 * time),
            np.sin(2.0 * np.pi * 90.0 * time),
            np.sin(2.0 * np.pi * 25.0 * time),
        ]
    )
    source = np.sum(imfs, axis=0)
    ordering = calculate_frequency_ordering_score(imfs, fs)
    nrmse = calculate_reconstruction_nrmse(source, imfs, np.zeros_like(source))
    assert ordering == 1.0
    assert nrmse < 1e-12
    return {"frequency_ordering_score": ordering, "reconstruction_nrmse": nrmse}


def _stage_1_synthetic() -> dict[str, Any]:
    fs = 512.0
    time = np.arange(512) / fs
    source = np.sin(2.0 * np.pi * 35.0 * time) + 0.35 * np.sin(2.0 * np.pi * 110.0 * time)
    result = decompose_ceemdan(
        source,
        trials=2,
        epsilon=0.02,
        noise_seed=31415,
        sifting_iterations=2,
        parallel=False,
    )
    assert result.residual_verified
    assert result.reconstruction_nrmse < 1e-8
    return {
        "number_of_imfs": result.num_imfs,
        "reconstruction_nrmse": result.reconstruction_nrmse,
        "residual_verified": result.residual_verified,
    }


def _stage_2_unit_and_synthetic() -> dict[str, Any]:
    fs = 1000.0
    time = np.arange(1000) / fs
    chatter = np.sin(2.0 * np.pi * 300.0 * time)
    forced = np.sin(2.0 * np.pi * 20.0 * time)
    background = 0.15 * np.sin(2.0 * np.pi * 120.0 * time)
    components = np.vstack((chatter, forced, background, np.zeros_like(time)))
    source = np.sum(components, axis=0)
    config = {
        "maiw": {"chatter_band_center": 300.0, "chatter_band_spread": 40.0},
        "physics_gating": {
            "chatter_energy_weight": 4.0,
            "correlation_weight": 2.0,
            "kurtosis_weight": 1.0,
            "frequency_proximity_weight": 1.0,
            "harmonic_penalty": 5.0,
            "offset": 1.5,
            "harmonic_tolerance_hz": 3.0,
            "harmonic_count": 4,
            "kurtosis_scale": 10.0,
        },
    }
    result = analyze_physics_guided_weighting(
        components,
        source,
        fs,
        {"rpm": 600.0, "tooth_count": 2},
        config,
        strict_config=True,
    )
    assert result.gates.shape == (3,)
    assert np.all((result.gates >= 0.0) & (result.gates <= 1.0))
    assert result.gates[0] > result.gates[1]
    assert not np.isclose(np.sum(result.gates), 1.0)
    return {
        "gates": result.gates.tolist(),
        "chatter_gate_exceeds_forced_gate": True,
        "independent_gate_sum": float(np.sum(result.gates)),
    }


def _stage_3_unit_and_synthetic() -> dict[str, Any]:
    fs = 1000.0
    time = np.arange(1000) / fs
    clean = np.sin(2.0 * np.pi * 125.0 * time)
    noise = 0.45 * np.sin(2.0 * np.pi * 430.0 * time)
    noisy = clean + noise
    result = wavelet_denoise_with_diagnostics(
        noisy,
        wavelet_name="db4",
        level=4,
        fs=fs,
        chatter_center=125.0,
        chatter_spread=35.0,
        band_aware=True,
        chatter_threshold_scale=0.25,
        noise_threshold_scale=1.8,
        threshold_mode="soft",
        min_noise_sigma=1e-12,
        clean_reference=clean,
    )
    assert result.denoised_signal.shape == noisy.shape
    assert result.thresholds_by_level
    assert result.metrics.synthetic_reference_rmse is not None
    assert result.metrics.synthetic_reference_snr_db is not None
    return {
        "output_length": int(result.denoised_signal.size),
        "threshold_count": len(result.thresholds_by_level),
        "synthetic_reference_rmse": result.metrics.synthetic_reference_rmse,
        "synthetic_reference_snr_db": result.metrics.synthetic_reference_snr_db,
    }


def _stage_4_unit_and_synthetic() -> dict[str, Any]:
    fs = 1000.0
    time = np.arange(1000) / fs
    growing = (0.15 + time) * np.sin(2.0 * np.pi * 125.0 * time)
    imf_1 = growing.copy()
    imf_2 = 0.1 * np.sin(2.0 * np.pi * 30.0 * time)
    components = np.vstack((imf_1, imf_2, np.zeros_like(time)))
    result = extract_window_feature_result(
        growing,
        growing,
        growing,
        components,
        fs,
        rpm=600.0,
        tooth_count=2,
        chatter_center=125.0,
        chatter_spread=30.0,
        imf_gates=np.array([0.9, 0.2]),
        wavelet_name="db4",
        wavelet_level=3,
    )
    assert result.values["early_hegr"] is not None
    assert float(result.values["early_hegr"] or 0.0) > 0.0
    schema = feature_schema()
    names = {str(item["name"]) for item in schema["features"]}
    assert "early_hegr" in names
    assert not any("classifier" in name or "probability" in name for name in names)
    return {
        "feature_count": len(result.values),
        "undefined_count": len(result.undefined_reasons),
        "early_hegr": result.values["early_hegr"],
        "schema_version": result.schema_version,
    }


def run_scientific_self_checks() -> dict[str, Any]:
    """Run fast unit-level and synthetic checks for all four active stages."""

    stage_1_unit = _capture(_stage_1_unit)
    stage_1_synthetic = _capture(_stage_1_synthetic)
    stage_2 = _capture(_stage_2_unit_and_synthetic)
    stage_3 = _capture(_stage_3_unit_and_synthetic)
    stage_4 = _capture(_stage_4_unit_and_synthetic)
    return {
        "Stage_1": {"unit": stage_1_unit, "synthetic": stage_1_synthetic},
        "Stage_2": {"unit": stage_2, "synthetic": stage_2},
        "Stage_3": {"unit": stage_3, "synthetic": stage_3},
        "Stage_4": {"unit": stage_4, "synthetic": stage_4},
    }
