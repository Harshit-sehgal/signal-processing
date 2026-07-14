"""Window-feature extraction tests (Goal 6.2)."""

import numpy as np
import pytest

from pg_amcd.features import (
    extract_sliding_window_features,
    extract_window_features,
    summarize_feature_repeatability,
)

REQUIRED_KEYS = [
    # time-domain
    "time_rms",
    "time_variance",
    "time_peak_to_peak",
    "time_crest_factor",
    "time_kurtosis",
    "time_skewness",
    "time_impulse_factor",
    "time_shape_factor",
    # frequency-domain
    "freq_centroid",
    "freq_spread",
    "freq_entropy",
    "freq_chatter_band_ratio",
    "freq_harmonics_ratio",
    "freq_peak",
    "freq_spectral_kurtosis",
    "freq_spindle_harmonic_ratio",
    "freq_sideband_ratio",
    # IMF
    "imf_max_energy_ratio",
    "imf1_correlation",
    "imf_centre_freq_mean",
    "imf_bandwidth_mean",
    "imf_entropy_mean",
    "imf_mode_mixing_index",
    # time-frequency
    "wavelet_high_freq_ratio",
]


def _window(n=256, n_imfs=5, seed=0):
    rng = np.random.default_rng(seed)
    sig = np.sin(2 * np.pi * 50.0 * np.arange(n) / 1000.0) + rng.normal(0, 0.1, n)
    imfs = rng.standard_normal((n_imfs, n))
    return sig, imfs


def test_extract_returns_all_required_keys():
    sig, imfs = _window()
    feats = extract_window_features(
        raw_window=sig,
        prep_physical_window=sig,
        denoised_physical_window=sig,
        imfs=imfs,
        fs=1000.0,
        rpm=600.0,
        tooth_count=1,
    )
    for k in REQUIRED_KEYS:
        assert k in feats, f"missing feature key: {k}"


def test_extract_values_finite():
    sig, imfs = _window()
    feats = extract_window_features(
        raw_window=sig,
        prep_physical_window=sig,
        denoised_physical_window=sig,
        imfs=imfs,
        fs=1000.0,
        rpm=600.0,
        tooth_count=1,
    )
    for v in feats.values():
        assert np.isfinite(v), f"non-finite feature: {v}"


def test_extract_deterministic():
    sig, imfs = _window()
    a = extract_window_features(sig, sig, sig, imfs, 1000.0, 600.0, 1)
    b = extract_window_features(sig, sig, sig, imfs, 1000.0, 600.0, 1)
    assert a == b


def test_extract_constant_signal_no_nan():
    sig = np.ones(256)
    imfs = np.ones((4, 256))
    feats = extract_window_features(sig, sig, sig, imfs, 1000.0, 600.0, 1)
    for v in feats.values():
        assert np.isfinite(v)


def test_repeated_stage4_extractions_are_compared_directly():
    signal, imfs = _window(n=256, n_imfs=4)
    first = extract_sliding_window_features(
        signal,
        signal,
        signal,
        imfs,
        1000.0,
        600.0,
        2,
        window_seconds=0.128,
        overlap_ratio=0.5,
        chatter_center=125.0,
        chatter_spread=50.0,
    )
    second = extract_sliding_window_features(
        signal,
        signal,
        signal,
        imfs,
        1000.0,
        600.0,
        2,
        window_seconds=0.128,
        overlap_ratio=0.5,
        chatter_center=125.0,
        chatter_spread=50.0,
    )

    summary = summarize_feature_repeatability(first, second)

    assert summary["repeat_count"] == 2
    assert summary["window_count"] == len(first)
    assert summary["deterministic"] is True
    assert summary["exact_value_match"] is True
    assert summary["maximum_absolute_difference"] == 0.0


def test_repeatability_comparison_rejects_incompatible_runs():
    signal, imfs = _window(n=256, n_imfs=4)
    records = extract_sliding_window_features(
        signal,
        signal,
        signal,
        imfs,
        1000.0,
        600.0,
        2,
        window_seconds=0.128,
        overlap_ratio=0.5,
        chatter_center=125.0,
        chatter_spread=50.0,
    )

    with pytest.raises(ValueError, match="at least one window"):
        summarize_feature_repeatability((), ())
    with pytest.raises(ValueError, match="same number of windows"):
        summarize_feature_repeatability(records, records[:-1])
    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_feature_repeatability(records, records, absolute_tolerance=-1.0)
