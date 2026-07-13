"""Decomposition tests (CEEMDAN + decomposition metrics)."""

import numpy as np

from pg_amcd.decomposition import (
    run_ceemdan,
    calculate_adjacent_imf_correlation,
    calculate_spectral_overlap,
    calculate_orthogonality_index,
)


def _make_signal(n=1000, fs=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.1, n)


def _decompose():
    sig = _make_signal()
    return run_ceemdan(sig, trials=2, epsilon=0.02, noise_seed=42, sifting_iterations=2)


def test_run_ceemdan_shape():
    imfs = _decompose()
    assert imfs.ndim == 2
    assert imfs.shape[1] == 1000
    assert imfs.shape[0] >= 2


def test_adjacent_correlation_finite():
    imfs = _decompose()
    mean_c, max_c = calculate_adjacent_imf_correlation(imfs)
    assert np.isfinite(mean_c) and np.isfinite(max_c)
    assert 0.0 <= max_c <= 1.0


def test_spectral_overlap_finite():
    imfs = _decompose()
    ov = calculate_spectral_overlap(imfs, 1000.0)
    assert np.isfinite(ov) and ov >= 0.0


def test_orthogonality_index_finite():
    sig = _make_signal()
    imfs = run_ceemdan(sig, trials=2, epsilon=0.02, noise_seed=42, sifting_iterations=2)
    oi = calculate_orthogonality_index(imfs, sig)
    assert np.isfinite(oi)
