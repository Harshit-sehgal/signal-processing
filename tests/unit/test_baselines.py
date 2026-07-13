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
    raw = agg["raw"]
    butter = agg["butterworth_only"]

    # The proposed pipeline must beat doing nothing (raw) ...
    assert full["rmse"] < raw["rmse"], (full["rmse"], raw["rmse"])
    assert full["snr_db"] > raw["snr_db"], (full["snr_db"], raw["snr_db"])
    assert full["noise_band_attenuation"] > raw["noise_band_attenuation"]

    # ... and a simple bandpass-only baseline.
    assert full["rmse"] < butter["rmse"], (full["rmse"], butter["rmse"])
    assert full["snr_db"] > butter["snr_db"]
