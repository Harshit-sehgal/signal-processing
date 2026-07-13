"""Pipeline tests (exploratory run, cutoff search, no fake detection)."""

import numpy as np

from pg_amcd.pipeline import process_recording
from pg_amcd.models import PipelineResult


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


def _make_signal(fs=1000.0, duration=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    return t, (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.1, n))


def test_process_recording_exploratory():
    t, sig = _make_signal()
    cfg = _minimal_config([50.0, 150.0, 250.0])
    res = process_recording(t, sig, cfg, mode="exploratory")
    assert isinstance(res, PipelineResult)
    assert res.selected_parameters["cutoff_frequency"] in [50.0, 150.0, 250.0]
    assert len(res.selected_parameters["cutoff_search"]) == 3
    # No fabricated detector: per-window chatter fields are nan / not_evaluated


def test_process_recording_window_results():
    t, sig = _make_signal()
    cfg = _minimal_config([50.0])
    res = process_recording(t, sig, cfg, mode="exploratory")
    assert len(res.window_results) == 1
    wr = res.window_results[0]
    assert hasattr(wr, "time_segment")
    assert np.all(np.isfinite(wr.imfs))
    assert np.isnan(wr.chatter_probability)
    assert wr.predicted_label == "not_evaluated"
    assert np.isnan(wr.confidence)
