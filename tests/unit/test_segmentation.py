"""Segmentation tests (max-energy segment + sliding windows)."""

import numpy as np

from pg_amcd.segmentation import select_max_energy_segment_indices, generate_sliding_windows


def test_select_max_energy_segment():
    n = 5000
    sig = np.zeros(n)
    sig[1000:2000] = 5.0
    start, end = select_max_energy_segment_indices(sig, segment_points=1000)
    assert end - start == 1000
    assert start == 1000


def test_select_max_energy_short_signal():
    sig = np.ones(500)
    start, end = select_max_energy_segment_indices(sig, segment_points=1000)
    assert start == 0 and end == 500


def test_sliding_windows():
    fs = 1000.0
    n = 4000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 50 * t)
    wins = generate_sliding_windows(t, sig, fs, window_seconds=1.0, overlap_ratio=0.75)
    assert len(wins) > 0
    for w in wins:
        assert "start_idx" in w and "end_idx" in w
        assert w["end_idx"] - w["start_idx"] == int(fs)
        assert np.allclose(w["signal_segment"], sig[w["start_idx"]:w["end_idx"]])
        assert "time_segment" in w


def test_sliding_windows_invalid_overlap_does_not_hang():
    fs = 1000.0
    n = 4000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 50 * t)
    wins = generate_sliding_windows(t, sig, fs, window_seconds=1.0, overlap_ratio=1.5)
    assert isinstance(wins, list)
