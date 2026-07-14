"""Focused acceptance tests for the explicit Stage 1 scientific core."""

from __future__ import annotations

import numpy as np
import pytest

from pg_amcd.decomposition import (
    calculate_decomposition_metrics,
    calculate_frequency_ordering_score,
    calculate_reconstruction_nrmse,
    calculate_seed_stability,
    decompose_ceemdan,
)
from pg_amcd.optimization import (
    calculate_chatter_band_distortion,
    optimize_cutoff,
)
from pg_amcd.preprocessing import (
    butter_bandpass_filter_sos,
    preprocess_signal_result,
)


def _sinusoidal_modes(
    frequencies: list[float],
    *,
    fs: float = 1000.0,
    n_samples: int = 2000,
) -> np.ndarray:
    time = np.arange(n_samples) / fs
    return np.stack([np.sin(2.0 * np.pi * frequency * time) for frequency in frequencies])


def test_preprocessing_result_preserves_physical_scale_and_parameters():
    fs = 1000.0
    time = np.arange(2000) / fs
    raw = 2.5 * np.sin(2 * np.pi * 50 * time) + 0.4 * np.sin(2 * np.pi * 200 * time)

    result = preprocess_signal_result(
        raw,
        20.0,
        400.0,
        fs,
        order=4,
        detrend_before_filter=True,
        scale_percentile=98.0,
    )

    np.testing.assert_allclose(
        result.restore_physical(result.scaled_signal),
        result.physical_signal,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.parameters.highpass_cutoff_hz == 20.0
    assert result.parameters.lowpass_cutoff_hz == 400.0
    assert result.parameters.filter_order == 4
    assert result.parameters.scale_percentile == 98.0
    assert np.percentile(np.abs(result.scaled_signal), 98.0) == pytest.approx(1.0)


def test_sos_filter_reports_the_real_filtfilt_padding_failure():
    with pytest.raises(ValueError, match="too short for zero-phase SOS filtering"):
        butter_bandpass_filter_sos(np.arange(7, dtype=float), 20.0, 400.0, 1000.0)


def test_ceemdan_result_explicitly_verifies_and_separates_residual():
    fs = 500.0
    time = np.arange(500) / fs
    signal = np.sin(2 * np.pi * 35 * time) + 0.4 * np.sin(2 * np.pi * 110 * time)

    result = decompose_ceemdan(
        signal,
        trials=2,
        epsilon=0.02,
        noise_seed=42,
        sifting_iterations=2,
        parallel=False,
    )

    assert result.num_imfs >= 1
    assert result.components.shape[0] == result.num_imfs + 1
    np.testing.assert_allclose(result.components[:-1], result.imfs)
    np.testing.assert_allclose(result.components[-1], result.residual)
    np.testing.assert_allclose(result.residual, signal - np.sum(result.imfs, axis=0))
    np.testing.assert_allclose(result.reconstruction, signal, rtol=1e-9, atol=1e-9)
    assert result.residual_verified is True
    assert result.residual_verification_nrmse < 1e-9
    assert result.reconstruction_nrmse < 1e-9
    assert result.parameters["algorithm"] == "CEEMDAN"
    assert result.parameters["noise_seed"] == 42


def test_stage1_metric_bundle_uses_physical_imfs_and_explicit_residual():
    fs = 1000.0
    imfs = _sinusoidal_modes([180.0, 90.0, 30.0], fs=fs)
    residual = np.linspace(-0.1, 0.1, imfs.shape[1])
    source = np.sum(imfs, axis=0) + residual

    metrics = calculate_decomposition_metrics(source, imfs, fs, residual=residual)

    assert metrics["number_of_imfs"] == 3
    assert metrics["reconstruction_nrmse"] < 1e-12
    assert metrics["frequency_ordering_score"] == 1.0
    assert "signed_orthogonality_index" in metrics
    assert "absolute_orthogonality_index" in metrics
    assert "mean_adjacent_imf_correlation" in metrics
    assert "maximum_adjacent_imf_correlation" in metrics
    assert "spectral_overlap" in metrics
    assert len(metrics["imf_metrics"]) == 3
    assert sum(row["energy_percentage"] for row in metrics["imf_metrics"]) == pytest.approx(100.0)
    assert all(0.0 <= row["spectral_entropy"] <= 1.0 for row in metrics["imf_metrics"])


def test_frequency_ordering_score_is_directionally_correct():
    fs = 1000.0
    descending = _sinusoidal_modes([200.0, 100.0, 40.0], fs=fs)
    ascending = descending[::-1]

    assert calculate_frequency_ordering_score(descending, fs) == 1.0
    assert calculate_frequency_ordering_score(ascending, fs) == 0.0


def test_structural_seed_stability_detects_change_despite_perfect_reconstruction():
    fs = 1000.0
    first = _sinusoidal_modes([200.0, 90.0, 35.0], fs=fs)
    changed = _sinusoidal_modes([170.0, 70.0, 20.0, 10.0], fs=fs)

    # Both decompositions can reconstruct their respective source perfectly;
    # reconstruction NRMSE therefore cannot establish structural stability.
    assert calculate_reconstruction_nrmse(np.sum(first, axis=0), first) < 1e-12
    assert calculate_reconstruction_nrmse(np.sum(changed, axis=0), changed) < 1e-12

    identical = calculate_seed_stability([first, first.copy()], fs, seeds=[1, 2])
    different = calculate_seed_stability([first, changed], fs, seeds=[1, 2])

    assert identical["instability_score"] < 1e-12
    assert identical["matched_imf_correlation_mean"] == pytest.approx(1.0)
    assert different["instability_score"] > identical["instability_score"]
    assert different["centre_frequency_instability"] > 0.0
    assert different["imf_count_range"] == 1
    assert different["matched_imf_correlation_mean"] < 1.0


def test_chatter_band_distortion_compares_raw_with_physical_candidate():
    fs = 1000.0
    time = np.arange(2000) / fs
    raw = np.sin(2 * np.pi * 125 * time)

    assert calculate_chatter_band_distortion(raw, raw, fs, 125.0, 25.0) == pytest.approx(0.0)
    assert calculate_chatter_band_distortion(raw, 0.5 * raw, fs, 125.0, 25.0) > 0.5


def test_cutoff_search_records_controlled_segment_and_structural_seed_metrics():
    fs = 500.0
    time = np.arange(500) / fs
    rng = np.random.default_rng(7)
    raw = (
        np.sin(2 * np.pi * 30 * time)
        + 0.5 * np.sin(2 * np.pi * 120 * time)
        + 0.05 * rng.standard_normal(time.size)
    )
    config = {
        "preprocessing": {
            "filter_order": 2,
            "low_pass_cutoff_hz": 200.0,
            "scale_percentile": 97.5,
        },
        "ceemdan": {
            "trials": 1,
            "search_trials": 1,
            "epsilon": 0.02,
            "noise_seed": 11,
            "sifting_iterations": 2,
            "stability_seeds": [11, 12],
            "parallel": False,
        },
        "maiw": {
            "chatter_band_center": 120.0,
            "chatter_band_spread": 30.0,
        },
    }

    result = optimize_cutoff(raw, [20.0, 50.0], config, fs)

    assert result.selected_cutoff in {20.0, 50.0}
    assert len(result.per_cutoff_metrics) == 2
    assert len({row["source_segment_sha256"] for row in result.per_cutoff_metrics}) == 1
    for row in result.per_cutoff_metrics:
        assert row["source_segment_samples"] == raw.size
        assert row["lowpass_cutoff_hz"] == 200.0
        assert row["filter_order"] == 2
        assert row["ceemdan_seeds"] == [11, 12]
        assert row["seed_stability"]["n_seeds"] == 2
        assert "centre_frequency_instability" in row["seed_stability"]
        assert "energy_distribution_l1" in row["seed_stability"]
        assert "matched_imf_correlation_mean" in row["seed_stability"]
        assert "spectral_overlap_standard_deviation" in row["seed_stability"]
        assert "imf_count_range" in row["seed_stability"]
        assert row["mean_reconstruction_nrmse"] < 1e-8
        assert np.isfinite(row["final_score"])
