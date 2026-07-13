"""Preprocessing tests (bandpass filter + robust scaling)."""

import numpy as np
import pytest

from pg_amcd.preprocessing import preprocess_signal, butter_bandpass_filter_sos


def _sig(n=2000, fs=1000.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)


def test_preprocess_returns_three():
    sig = _sig()
    phys, scaled, sf = preprocess_signal(sig, 20.0, 400.0, 1000.0)
    assert phys.shape == sig.shape
    assert scaled.shape == sig.shape
    assert np.isfinite(sf)


def test_preprocess_scaled_near_unit():
    sig = _sig()
    phys, scaled, sf = preprocess_signal(sig, 20.0, 400.0, 1000.0)
    assert np.all(np.isfinite(scaled))
    # robust 99.5th-percentile scaling places the bulk of the signal near unit amplitude
    assert np.percentile(np.abs(scaled), 99.5) == pytest.approx(1.0, rel=0.25)


def test_butter_invalid_cutoffs():
    sig = _sig(500)
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(sig, 100.0, 50.0, 1000.0)  # low > high
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(sig, 0.0, 100.0, 1000.0)  # low <= 0


def test_butter_invalid_order():
    sig = _sig(500)
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(sig, 20.0, 400.0, 1000.0, order=0)
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(sig, 20.0, 400.0, 1000.0, order=-1)


def test_butter_insufficient_samples():
    sig = np.ones(3)  # shorter than 2*order+1 (=7) for order=3
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(sig, 20.0, 400.0, 1000.0, order=3)
