"""Denoising / band-aware wavelet selection tests (Goal 5.6)."""

import numpy as np
import pytest

from pg_amcd.denoising import wavelet_denoise, select_best_wavelet
from pg_amcd.synthetic import generate_synthetic_signal


def test_wavelet_denoise_finite():
    fs = 1000.0
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=1, chatter_freq=125.0)
    out = wavelet_denoise(
        sig, wavelet_name="db4", level=3, fs=fs,
        chatter_center=125.0, chatter_spread=50.0, band_aware=True,
    )
    assert out.shape == sig.shape
    assert np.all(np.isfinite(out))


def test_wavelet_denoise_invalid_wavelet():
    fs = 1000.0
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=0.5, seed=2, chatter_freq=125.0)
    with pytest.raises(Exception):
        wavelet_denoise(
            sig, wavelet_name="not_a_wavelet", level=3, fs=fs,
            chatter_center=125.0, chatter_spread=50.0,
        )


def test_select_best_wavelet_returns_candidate():
    fs = 1000.0
    t, clean, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=3, chatter_freq=125.0, snr_db=25.0)
    rng = np.random.default_rng(4)
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)
    best, results = select_best_wavelet(
        noisy, clean, [("db4", 3), ("sym5", 3), ("coif1", 3)], fs, 125.0, 50.0
    )
    assert (best["wavelet"], best["level"]) in [("db4", 3), ("sym5", 3), ("coif1", 3)]
    assert len(results) == 3
    assert all(np.isfinite(r["snr_db"]) for r in results)


def test_wavelet_denoise_invalid_level():
    fs = 1000.0
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=0.5, seed=5, chatter_freq=125.0)
    with pytest.raises(ValueError):
        wavelet_denoise(
            sig, wavelet_name="db4", level=0, fs=fs,
            chatter_center=125.0, chatter_spread=50.0,
        )


def test_wavelet_denoise_too_short():
    fs = 1000.0
    sig = np.ones(2)
    with pytest.raises(ValueError):
        wavelet_denoise(
            sig, wavelet_name="db4", level=1, fs=fs,
            chatter_center=125.0, chatter_spread=50.0,
        )
