"""Weighting / reconstruction tests (Segment 2.3 shape validation)."""

import numpy as np
import pytest

from pg_amcd.weighting import (
    calculate_maiw_weights,
    calculate_physics_gated_weights,
    reconstruct_weighted_signal,
    reconstruct_gated_signal,
)

MAIW_CFG = {
    "maiw": {
        "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25,
        "chatter_band_center": 125.0, "chatter_band_spread": 50.0,
    }
}


def _make_imfs(n_imfs=4, n=200):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_imfs, n))


def test_maiw_weights_shape_and_sum():
    imfs = _make_imfs()
    original = np.sum(imfs[:-1], axis=0)
    W, C, E, K, F = calculate_maiw_weights(imfs, original, 1000.0, MAIW_CFG)
    assert W.shape[0] == imfs.shape[0] - 1
    assert np.isclose(np.sum(W), 1.0)


def test_physics_gated_weights_shape():
    imfs = _make_imfs()
    s_seg = np.sum(imfs[:-1], axis=0)
    gates, *_ = calculate_physics_gated_weights(imfs, s_seg, 1000.0, 1200.0, 1, MAIW_CFG)
    assert gates.shape[0] == imfs.shape[0] - 1
    assert np.all(np.isfinite(gates))


def test_reconstruction_shape_validation_1d():
    with pytest.raises(ValueError):
        reconstruct_weighted_signal(
            np.random.default_rng(1).standard_normal(100), np.array([0.5, 0.5])
        )


def test_reconstruction_weight_count_mismatch():
    imfs = _make_imfs(n_imfs=4, n=50)
    with pytest.raises(ValueError):
        reconstruct_weighted_signal(imfs, np.array([0.5, 0.5]))  # needs 3 weights


def test_maiw_requires_2d():
    with pytest.raises(ValueError):
        calculate_maiw_weights(
            np.random.default_rng(2).standard_normal(100), np.ones(100), 1000.0, MAIW_CFG
        )


def test_gated_equals_weighted():
    imfs = _make_imfs(n_imfs=5, n=100)
    gates = np.random.default_rng(3).random(4)
    np.testing.assert_allclose(
        reconstruct_gated_signal(imfs, gates),
        reconstruct_weighted_signal(imfs, gates),
    )


def test_maiw_zero_coefficients_raises():
    imfs = _make_imfs()
    original = np.sum(imfs[:-1], axis=0)
    cfg = {"maiw": {k: 0.0 for k in ("alpha", "beta", "gamma", "delta")}}
    with pytest.raises(ValueError):
        calculate_maiw_weights(imfs, original, 1000.0, cfg)
