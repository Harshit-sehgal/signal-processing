"""Weighting / reconstruction tests (Segment 2.3 shape validation)."""

import numpy as np
import pytest

from pg_amcd.weighting import (
    calculate_maiw_weights,
    calculate_physics_gated_weights,
    reconstruct_weighted_signal,
    reconstruct_gated_signal,
    summarize_matched_gate_stability,
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


def _sine_imfs(freqs, n=200, fs=1000.0):
    t = np.arange(n) / fs
    return np.asarray([np.sin(2 * np.pi * f * t) for f in freqs])


def test_matched_gate_stability_equal_counts():
    fs = 1000.0
    decomp_1 = _sine_imfs([120.0, 60.0, 30.0], n=512, fs=fs)
    decomp_2 = _sine_imfs([120.0, 60.0, 30.0], n=512, fs=fs)
    gates = [
        np.array([0.9, 0.2, 0.1]),
        np.array([0.85, 0.25, 0.15]),
    ]
    result = summarize_matched_gate_stability(
        gates, [decomp_1, decomp_2], fs, selection_threshold=0.5
    )
    assert result["available"] is True
    assert result["physical_imf_count"] == 3
    assert len(result["mean_gate_by_imf"]) == 3
    assert len(result["std_gate_by_imf"]) == 3
    assert np.allclose(result["mean_gate_by_imf"], [0.875, 0.225, 0.125], atol=1e-6)
    assert result["selected_imf_consistency"] == 1.0
    assert result["unmatched_mode_penalty"] == 0.0


def test_matched_gate_stability_mismatched_counts():
    fs = 1000.0
    decomp_1 = _sine_imfs([120.0, 60.0, 30.0], n=512, fs=fs)
    # Second seed has only the first two modes; the third mode is unmatched.
    decomp_2 = _sine_imfs([120.0, 60.0], n=512, fs=fs)
    gates = [np.array([0.9, 0.2, 0.1]), np.array([0.85, 0.25])]
    result = summarize_matched_gate_stability(
        gates, [decomp_1, decomp_2], fs, selection_threshold=0.5
    )
    assert result["available"] is True
    assert result["physical_imf_count"] == 3
    assert result["matched_counts"] == [3, 2]
    assert result["unmatched_mode_penalty"] == (1 / 3) / 2  # seed 2 unmatched proportion averaged with 0
    assert np.isclose(result["mean_gate_by_imf"][0], 0.875)
    assert np.isnan(result["std_gate_by_imf"][2]) or result["std_gate_by_imf"][2] == 0.0


def test_matched_gate_stability_single_seed():
    fs = 1000.0
    decomp = _sine_imfs([120.0, 60.0], n=512, fs=fs)
    gates = [np.array([0.3, 0.7])]
    result = summarize_matched_gate_stability(
        gates, [decomp], fs, selection_threshold=0.5
    )
    assert result["available"] is True
    assert result["n_seeds"] == 1
    assert result["mean_gate_by_imf"] == pytest.approx([0.3, 0.7])
    assert all(std == 0.0 for std in result["std_gate_by_imf"])
    assert result["selected_imf_consistency"] == 1.0


def test_matched_gate_stability_invalid_gate_bounds_raise():
    fs = 1000.0
    decomp = _sine_imfs([120.0], n=512, fs=fs)
    with pytest.raises(ValueError):
        summarize_matched_gate_stability(
            [np.array([1.2])], [decomp], fs, selection_threshold=0.5
        )


def test_matched_gate_stability_sample_count_mismatch_raises():
    fs = 1000.0
    decomp_1 = _sine_imfs([120.0], n=512, fs=fs)
    decomp_2 = _sine_imfs([120.0], n=300, fs=fs)
    with pytest.raises(ValueError):
        summarize_matched_gate_stability(
            [np.array([0.5]), np.array([0.5])],
            [decomp_1, decomp_2],
            fs,
            selection_threshold=0.5,
        )


def test_matched_gate_stability_reorders_permuted_modes():
    fs = 1000.0
    # Reference: 60 Hz mode first, 120 Hz mode second.
    decomp_1 = _sine_imfs([60.0, 120.0], n=512, fs=fs)
    # Target has the same two modes but in reverse order.
    decomp_2 = _sine_imfs([120.0, 60.0], n=512, fs=fs)
    # Gates follow the *physical* modes, so the 60 Hz gate is high in both seeds
    # even though its row index differs.
    gates = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    result = summarize_matched_gate_stability(
        gates, [decomp_1, decomp_2], fs, selection_threshold=0.5
    )
    assert result["available"] is True
    assert result["unmatched_mode_penalty"] == 0.0
    # After matching by centre frequency, both modes should have mean gates
    # close to [0.9, 0.1] (the order of reference IMFs).
    np.testing.assert_allclose(result["mean_gate_by_imf"], [0.9, 0.1], atol=1e-6)
    np.testing.assert_allclose(result["std_gate_by_imf"], [0.0, 0.0], atol=1e-6)
