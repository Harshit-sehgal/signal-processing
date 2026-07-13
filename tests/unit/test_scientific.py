"""Unit tests for Sprint 4 scientific signal-processing modules.

Covers: synthetic data generation + denoising metrics (Goal 5.4), controlled
cutoff optimisation + 5-component objective + multi-seed stability (Goals 5.1-5.3),
provenance hashing (Goal 4.3/4.4), independent IMF gating (Goal 5.5), band-aware
wavelet selection (Goal 5.6), and the pipeline wiring that populates `cutoff_search`.
"""

import os
import time

import numpy as np
from pytest import approx
import pytest

from pg_amcd.synthetic import generate_synthetic_signal, evaluate_denoising_performance
from pg_amcd.optimization import optimize_cutoff, compute_cutoff_objective, multi_seed_stability
from pg_amcd.provenance import compute_file_sha256, is_output_stale, compute_run_id
from pg_amcd.weighting import reconstruct_gated_signal, reconstruct_weighted_signal
from pg_amcd.denoising import select_best_wavelet
from pg_amcd.pipeline import process_recording
from pg_amcd.models import CutoffOptimizationResult


# --------------------------------------------------------------------------- #
# synthetic data + denoising metrics (Goal 5.4)
# --------------------------------------------------------------------------- #
def test_generate_synthetic_signal_shapes_and_components():
    fs = 1000.0
    t, sig, comp = generate_synthetic_signal(
        fs=fs, duration=1.0, seed=1, chatter_freq=125.0, chatter_onset=0.4
    )
    assert t.shape == sig.shape
    assert t.shape[0] == int(fs)
    assert np.all(np.isfinite(t)) and np.all(np.isfinite(sig))
    for key in ("forced", "chatter", "drift", "noise", "clean"):
        assert key in comp
        assert comp[key].shape == sig.shape
    # controlled onset: chatter energy is negligible before onset, present after
    onset_idx = int(0.4 * fs)
    pre_power = float(np.mean(comp["chatter"][:onset_idx] ** 2))
    post_power = float(np.mean(comp["chatter"][onset_idx:] ** 2))
    assert pre_power < 0.05 * post_power
    assert post_power > 0.0


def test_evaluate_denoising_performance_identity():
    fs = 1000.0
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=2, chatter_freq=125.0)
    m = evaluate_denoising_performance(sig, sig.copy(), fs, 125.0, 50.0)
    assert m["rmse"] < 1e-9
    assert m["spectral_distortion"] < 1e-9
    assert m["chatter_band_retention"] == approx(1.0, abs=1e-6)
    # identity: no energy removed, so noise-band attenuation is zero
    assert m["noise_band_attenuation"] == approx(0.0, abs=1e-6)
    assert np.isfinite(m["onset_detection_error"])
    assert m["snr_db"] > 50.0  # identical reference -> very high SNR


def test_evaluate_denoising_performance_signal_keys():
    fs = 1000.0
    t, clean, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=9, chatter_freq=125.0)
    rng = np.random.default_rng(0)
    noisy = clean + 0.3 * rng.standard_normal(clean.shape)
    m = evaluate_denoising_performance(clean, noisy, fs, 125.0, 50.0)
    for key in ("rmse", "snr_db", "spectral_distortion",
                "chatter_band_retention", "noise_band_attenuation", "onset_detection_error"):
        assert key in m
        assert np.isfinite(m[key]) or key == "snr_db"  # snr may be -inf for worse-than-noise
    assert m["rmse"] > 0.0


# --------------------------------------------------------------------------- #
# cutoff optimisation + 5-component objective + multi-seed (Goals 5.1-5.3)
# --------------------------------------------------------------------------- #
def _minimal_config(cutoffs, fs=1000.0):
    return {
        "sampling_rate": fs,
        "segment_points": 1000,
        "ceemdan": {
            "trials": 2, "search_trials": 1, "epsilon": 0.02,
            "noise_seed": 42, "sifting_iterations": 2,
            "search_cutoffs": cutoffs, "search_seeds": 2,
        },
        "maiw": {
            "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25,
            "chatter_band_center": 125.0, "chatter_band_spread": 50.0,
        },
        "wavelet": {"wavelet_name": "db4", "level": 3},
    }


def test_optimize_cutoff_returns_valid_candidate():
    fs = 1000.0
    cutoffs = [50.0, 150.0, 250.0]
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=3, chatter_freq=125.0)
    cfg = _minimal_config(cutoffs, fs)
    res = optimize_cutoff(sig, cutoffs, cfg, fs, n_seeds=2)
    assert isinstance(res, CutoffOptimizationResult)
    assert res.selected_cutoff in cutoffs
    assert len(res.per_cutoff_metrics) == len(cutoffs)
    for d in res.per_cutoff_metrics:
        for key in (
            "cutoff", "objective", "seed_instability", "final_score",
            "mean_reconstruction_nrmse",
        ):
            assert key in d
        assert np.isfinite(d["objective"])


def test_optimize_cutoff_deterministic_for_fixed_seed():
    fs = 1000.0
    cutoffs = [50.0, 150.0]
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=4, chatter_freq=125.0)
    cfg = _minimal_config(cutoffs, fs)
    r1 = optimize_cutoff(sig, cutoffs, cfg, fs, n_seeds=1)
    r2 = optimize_cutoff(sig, cutoffs, cfg, fs, n_seeds=1)
    assert r1.selected_cutoff == r2.selected_cutoff
    assert r1.per_cutoff_metrics == r2.per_cutoff_metrics


def test_multi_seed_stability_runs_all_seeds():
    seeds = [1, 2, 3]
    out = multi_seed_stability(lambda s: float(s), seeds)
    for key in ("mean", "std", "min", "max", "ci95_low", "ci95_high", "n_seeds"):
        assert key in out
    assert out["n_seeds"] == len(seeds)
    assert out["min"] == 1.0 and out["max"] == 3.0
    assert np.isfinite(out["mean"])


def test_objective_deterministic_and_finite():
    fs = 1000.0
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=0.5, seed=6, chatter_freq=125.0)
    cfg = _minimal_config([100.0], fs)
    imfs = sig[np.newaxis, :]
    b1 = compute_cutoff_objective(imfs, sig, fs, cfg)
    b2 = compute_cutoff_objective(imfs, sig, fs, cfg)
    assert np.isfinite(b1)
    assert b1 == b2  # deterministic for fixed inputs


# --------------------------------------------------------------------------- #
# provenance (Goals 4.3 / 4.4)
# --------------------------------------------------------------------------- #
def test_sha256_deterministic(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert compute_file_sha256(str(p)) == compute_file_sha256(str(p))


def test_is_output_stale_missing(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"data")
    assert is_output_stale(str(inp), [str(tmp_path / "missing.npz")]) is True


def test_is_output_stale_fresh(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"data")
    out = tmp_path / "out.npz"
    out.write_bytes(b"x")
    older = time.time() - 10
    os.utime(str(inp), (older, older))
    os.utime(str(out), (time.time(), time.time()))
    assert is_output_stale(str(inp), [str(out)]) is False


def test_compute_run_id_deterministic():
    a = compute_run_id("c", "g", ["a", "b"])
    b = compute_run_id("c", "g", ["a", "b"])
    assert a == b
    assert isinstance(a, str) and len(a) > 0


# --------------------------------------------------------------------------- #
# independent IMF gating (Goal 5.5)
# --------------------------------------------------------------------------- #
def test_reconstruct_gated_equals_weighted():
    rng = np.random.default_rng(0)
    imfs = rng.standard_normal((5, 100))
    gates = rng.random((4,))
    np.testing.assert_allclose(
        reconstruct_gated_signal(imfs, gates),
        reconstruct_weighted_signal(imfs, gates),
    )


# --------------------------------------------------------------------------- #
# band-aware wavelet selection (Goal 5.6)
# --------------------------------------------------------------------------- #
def test_select_best_wavelet_returns_best():
    fs = 1000.0
    t, clean, _ = generate_synthetic_signal(
        fs=fs, duration=1.0, seed=7, chatter_freq=125.0, snr_db=25.0
    )
    rng = np.random.default_rng(11)
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)
    candidates = [("db4", 3), ("sym5", 3), ("coif1", 3)]
    best, results = select_best_wavelet(noisy, clean, candidates, fs, 125.0, 50.0)
    assert len(results) == len(candidates)
    assert (best["wavelet"], best["level"]) in candidates
    assert all(np.isfinite(r["snr_db"]) for r in results)


# --------------------------------------------------------------------------- #
# pipeline wiring: cutoff_search populated (Goal 5.1)
# --------------------------------------------------------------------------- #
def test_pipeline_populates_cutoff_search():
    fs = 1000.0
    cutoffs = [50.0, 150.0, 250.0]
    t, sig, _ = generate_synthetic_signal(fs=fs, duration=1.0, seed=8, chatter_freq=125.0)
    cfg = _minimal_config(cutoffs, fs)
    res = process_recording(t, sig, cfg, mode="exploratory")
    assert res.selected_parameters["cutoff_frequency"] in cutoffs
    search = res.selected_parameters["cutoff_search"]
    assert isinstance(search, list) and len(search) == len(cutoffs)


def test_optimize_cutoff_empty_raises():
    cfg = _minimal_config([50.0, 150.0])
    rng = np.random.default_rng(7)
    seg = rng.standard_normal(500)
    with pytest.raises(ValueError):
        optimize_cutoff(seg, [], cfg, 1000.0)
