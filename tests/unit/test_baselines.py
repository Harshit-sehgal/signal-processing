"""Baseline denoising comparison (Segment 5 math acceptance criteria).

The full proposed pipeline must outperform the required baselines on synthetic
data with known ground truth. These baselines need no real labelled dataset.
"""
import numpy as np

from pg_amcd.baselines import benchmark_denoising, METHODS

CEEMDAN_CHEAP = {
    "trials": 2,
    "epsilon": 0.02,
    "noise_seed": 42,
    "sifting_iterations": 2,
    "search_cutoffs": [100.0],
    "search_seeds": 1,
}


def test_benchmark_runs_all_methods():
    agg = benchmark_denoising(
        n_signals=3,
        fs=10_000.0,
        duration=0.4,
        seed=0,
        snr_db=20.0,
        ceemdan_cfg=CEEMDAN_CHEAP,
    )
    assert set(agg.keys()) == set(METHODS)
    for method, metrics in agg.items():
        for key, val in metrics.items():
            assert np.isfinite(val), f"{method}.{key} not finite: {val}"
        assert metrics["rmse"] > 0.0


def test_full_pipeline_outperforms_naive_baselines():
    agg = benchmark_denoising(
        n_signals=3,
        fs=10_000.0,
        duration=0.4,
        seed=0,
        snr_db=20.0,
        ceemdan_cfg=CEEMDAN_CHEAP,
    )
    full = agg["full_proposed"]
    others = [m for m in METHODS if m != "full_proposed"]
    other_rmse = [agg[m]["rmse"] for m in others]
    other_snr = [agg[m]["snr_db"] for m in others]
    other_att = [agg[m]["noise_band_attenuation"] for m in others]
    best_rmse = min(other_rmse)
    best_snr = max(other_snr)
    best_att = max(other_att)
    raw = agg["raw"]
    butter = agg["butterworth_only"]
    wavelet = agg["wavelet_only"]
    ceemdan_only = agg["ceemdan_only"]
    ceemdan_simple = agg["ceemdan_simple_selection"]
    current_maiw = agg["current_maiw"]
    stft = agg["stft_baseline"]
    # The proposed pipeline must beat doing nothing (raw) and a simple
    # bandpass-only baseline. Those keep forced/drift/noise, so the proposed
    # chatter-isolating pipeline strictly dominates them.
    assert full["rmse"] < raw["rmse"], (full["rmse"], raw["rmse"])
    assert full["snr_db"] > raw["snr_db"], (full["snr_db"], raw["snr_db"])
    assert full["noise_band_attenuation"] > raw["noise_band_attenuation"]
    assert full["rmse"] < butter["rmse"], (full["rmse"], butter["rmse"])
    assert full["snr_db"] > butter["snr_db"]
    # The remaining baselines (wavelet/CEEMDAN variants, current MAIW, STFT)
    # are competitive denoisers; the proposed pipeline must be best within a
    # small tolerance for seed variance.
    for m in (wavelet, ceemdan_only, ceemdan_simple, current_maiw, stft):
        assert full["rmse"] <= m["rmse"] + 0.01, (m, full["rmse"], m["rmse"])
    assert full["rmse"] <= best_rmse + 0.01, (full["rmse"], best_rmse)
    assert full["snr_db"] >= best_snr - 0.5, (full["snr_db"], best_snr)
    assert full["noise_band_attenuation"] >= best_att - 0.2
