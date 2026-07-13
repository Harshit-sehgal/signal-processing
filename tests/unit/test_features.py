"""Window-feature extraction tests (Goal 6.2)."""

import numpy as np

from pg_amcd.features import extract_window_features

REQUIRED_KEYS = [
    # time-domain
    "time_rms", "time_variance", "time_peak_to_peak", "time_crest_factor",
    "time_kurtosis", "time_skewness", "time_impulse_factor", "time_shape_factor",
    # frequency-domain
    "freq_centroid", "freq_spread", "freq_entropy", "freq_chatter_band_ratio",
    "freq_harmonics_ratio", "freq_peak", "freq_spectral_kurtosis",
    "freq_spindle_harmonic_ratio", "freq_sideband_ratio",
    # IMF
    "imf_max_energy_ratio", "imf1_correlation", "imf_centre_freq_mean",
    "imf_bandwidth_mean", "imf_entropy_mean", "imf_mode_mixing_index",
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
        raw_window=sig, prep_physical_window=sig, denoised_physical_window=sig,
        imfs=imfs, fs=1000.0, rpm=600.0, tooth_count=1,
    )
    for k in REQUIRED_KEYS:
        assert k in feats, f"missing feature key: {k}"


def test_extract_values_finite():
    sig, imfs = _window()
    feats = extract_window_features(
        raw_window=sig, prep_physical_window=sig, denoised_physical_window=sig,
        imfs=imfs, fs=1000.0, rpm=600.0, tooth_count=1,
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
