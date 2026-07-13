"""Unit tests for the mathematical validation metric suite."""

import numpy as np
import pytest

from pg_amcd.validation import (
    reconstruction_nrmse,
    orthogonality_index,
    mode_mixing_index,
    energy_distribution,
    frequency_ordering_index,
    validate_decomposition,
)


@pytest.fixture
def fs():
    return 10_000.0


def _synthetic_imfs(n_samples=2000, fs=10_000.0, n_modes=4):
    """Build IMFs as decreasing-frequency sinusoids (ideal ordering)."""
    t = np.arange(n_samples) / fs
    imfs = []
    for k in range(1, n_modes + 1):
        freq = 50.0 * k
        imfs.append(np.sin(2 * np.pi * freq * t))
    return np.stack(imfs)


def test_nrmse_perfect_reconstruction():
    imfs = _synthetic_imfs()
    original = imfs.sum(axis=0)
    assert reconstruction_nrmse(original, imfs) < 1e-9


def test_nrmse_increases_with_corruption():
    imfs = _synthetic_imfs()
    original = imfs.sum(axis=0)
    noisy = original + 0.5 * np.random.RandomState(0).randn(original.size)
    assert reconstruction_nrmse(noisy, imfs) > 1e-3


def test_orthogonality_zero_for_orthogonal_modes():
    rng = np.random.RandomState(1)
    a = rng.randn(500)
    b = rng.randn(500)
    b = b - np.dot(a, b) / np.dot(a, a) * a  # make orthogonal to a
    imfs = np.stack([a, b])
    assert abs(orthogonality_index(imfs)) < 1e-9


def test_orthogonality_positive_for_correlated_modes():
    rng = np.random.RandomState(2)
    a = rng.randn(500)
    b = a + 0.1 * rng.randn(500)
    imfs = np.stack([a, b])
    assert orthogonality_index(imfs) > 0.0


def test_mode_mixing_index_range():
    imfs = _synthetic_imfs()
    mmi = mode_mixing_index(imfs)
    assert 0.0 <= mmi <= 1.0


def test_energy_distribution_sums_to_100():
    imfs = _synthetic_imfs()
    dist = energy_distribution(imfs)
    assert dist.shape[0] == imfs.shape[0]
    assert abs(dist.sum() - 100.0) < 1e-6


def test_frequency_ordering_high_for_ideal_imfs(fs):
    imfs = _synthetic_imfs(fs=fs)
    assert frequency_ordering_index(imfs, fs) > 0.8


def test_validate_decomposition_keys(fs):
    imfs = _synthetic_imfs(fs=fs)
    original = imfs.sum(axis=0)
    report = validate_decomposition(original, imfs, fs)
    assert set(report) == {
        "nrmse",
        "orthogonality_index",
        "mode_mixing_index",
        "frequency_ordering_index",
    }
    assert report["nrmse"] < 1e-9
