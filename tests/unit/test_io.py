"""Input-validation tests (Segment 2.2 / Segment 3 edge cases)."""

import numpy as np
import pytest
import scipy.io

from pg_amcd.io import validate_and_load_signal


def _write_mat(path, t, signal, n_cols=2):
    if n_cols == 2:
        tsDS = np.column_stack((t, signal))
    else:
        extra = np.zeros((len(t), n_cols - 2))
        tsDS = np.column_stack((t, signal, extra))
    scipy.io.savemat(path, {"tsDS": tsDS})


def test_valid_load(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "ok.mat"
    _write_mat(str(p), t, signal)
    tt, sig, fs_est = validate_and_load_signal(str(p), configured_fs=fs)
    assert fs_est == pytest.approx(1000.0, rel=0.01)
    assert np.allclose(sig, signal)
    assert len(sig) == n


def test_empty_tsds(tmp_path):
    p = tmp_path / "empty.mat"
    scipy.io.savemat(str(p), {"tsDS": np.empty((0, 2))})
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=1000.0)


def test_non_numeric(tmp_path):
    p = tmp_path / "bad.mat"
    scipy.io.savemat(str(p), {"tsDS": np.array([["a", "b"], ["c", "d"]])})
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=1000.0)


def test_complex_input(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t) + 1j * 0.0
    p = tmp_path / "c.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs)


def test_duplicate_timestamps(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    t[100] = t[99]
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "dup.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs)


def test_timestamp_jitter(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs + np.linspace(0, 0.5, n)
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "jit.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs, max_timestamp_jitter=0.05)


def test_sampling_rate_deviation(tmp_path):
    n = 2000
    t = np.arange(n) / 2000.0  # implies 2000 Hz, configured 1000
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "fs.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=1000.0, tolerance=0.05)


def test_unrealistic_sampling_rate(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / (1e9)  # implies 1 GHz
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "crazy.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs, max_sampling_rate=1.0e7)


def test_more_than_two_columns(tmp_path):
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "multi.mat"
    _write_mat(str(p), t, signal, n_cols=4)
    tt, sig, fs_est = validate_and_load_signal(str(p), configured_fs=fs, signal_column=1)
    assert np.allclose(sig, signal)


def test_short_duration(tmp_path):
    fs = 1000.0
    n = 500
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t)
    p = tmp_path / "short.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs, min_duration_seconds=1.0)


def test_insufficient_samples(tmp_path):
    fs = 1000.0
    n = 1
    t = np.arange(n) / fs
    signal = np.array([0.5])
    p = tmp_path / "one.mat"
    _write_mat(str(p), t, signal)
    with pytest.raises(ValueError):
        validate_and_load_signal(str(p), configured_fs=fs, min_samples=2)
