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


def test_missing_unreadable_and_missing_variable_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_and_load_signal(tmp_path / "missing.mat", configured_fs=1000.0)

    corrupt = tmp_path / "corrupt.mat"
    corrupt.write_bytes(b"not a MAT file")
    with pytest.raises(ValueError, match="Failed to read MAT file"):
        validate_and_load_signal(corrupt, configured_fs=1000.0)

    missing_variable = tmp_path / "missing_variable.mat"
    scipy.io.savemat(missing_variable, {"other": np.ones(10)})
    with pytest.raises(ValueError, match="missing 'tsDS'"):
        validate_and_load_signal(missing_variable, configured_fs=1000.0)


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


def test_single_timestamp_interval_outlier_is_rejected(tmp_path):
    fs = 1000.0
    n = 2000
    time = np.arange(n) / fs
    time[100:] += 0.0004
    signal = np.sin(2 * np.pi * 50 * time)
    path = tmp_path / "interval_outlier.mat"
    _write_mat(str(path), time, signal)

    with pytest.raises(ValueError, match="Timestamp jitter"):
        validate_and_load_signal(path, configured_fs=fs, max_timestamp_jitter=0.05)


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


def test_invalid_matrix_shape_and_signal_column_are_rejected(tmp_path):
    one_column = tmp_path / "one_column.mat"
    scipy.io.savemat(one_column, {"tsDS": np.arange(20, dtype=float)[:, None]})
    with pytest.raises(ValueError, match="at least two columns"):
        validate_and_load_signal(one_column, configured_fs=1000.0)

    valid = tmp_path / "valid.mat"
    time = np.arange(2000) / 1000.0
    _write_mat(valid, time, np.sin(2 * np.pi * 50 * time))
    with pytest.raises(ValueError, match="out of range"):
        validate_and_load_signal(valid, configured_fs=1000.0, signal_column=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_time_or_signal_is_rejected(tmp_path, bad_value):
    time = np.arange(2000) / 1000.0
    signal = np.sin(2 * np.pi * 50 * time)
    signal[100] = bad_value
    path = tmp_path / "nonfinite.mat"
    _write_mat(path, time, signal)

    with pytest.raises(ValueError, match="only finite"):
        validate_and_load_signal(path, configured_fs=1000.0)


def test_constant_and_numerically_flat_signals_are_rejected(tmp_path):
    time = np.arange(2000) / 1000.0
    for name, signal in (
        ("constant", np.ones(time.size)),
        ("flat", 1.0 + 1e-18 * np.sin(2 * np.pi * 50 * time)),
    ):
        path = tmp_path / f"{name}.mat"
        _write_mat(path, time, signal)
        with pytest.raises(ValueError, match="constant or numerically flat"):
            validate_and_load_signal(path, configured_fs=1000.0)


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


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("configured_fs", None, "finite number"),
        ("configured_fs", "bad", "finite number"),
        ("configured_fs", 0.0, "configured_fs must be positive"),
        ("tolerance", 1.0, "tolerance must be in"),
        ("tolerance", -0.1, "at least"),
        ("max_timestamp_jitter", 1.0, "max_timestamp_jitter must be in"),
        ("min_samples", 1, "min_samples must be finite and at least 2"),
        ("min_samples", 2.5, "must be an integer"),
        ("signal_column", 1.5, "must be an integer"),
        ("min_sampling_rate", 0.0, "Sampling-rate bounds"),
        ("max_sampling_rate", 0.5, "Sampling-rate bounds"),
    ],
)
def test_invalid_validation_parameters_fail_before_mat_read(tmp_path, keyword, value, message):
    path = tmp_path / "valid.mat"
    time = np.arange(2000) / 1000.0
    _write_mat(path, time, np.sin(2 * np.pi * 50 * time))
    arguments = {"configured_fs": 1000.0, keyword: value}

    with pytest.raises(ValueError, match=message):
        validate_and_load_signal(path, **arguments)


def test_estimated_rate_below_configured_realistic_bound_is_rejected(tmp_path):
    time = np.arange(20) / 2.0
    signal = np.sin(2 * np.pi * 0.2 * time)
    path = tmp_path / "slow.mat"
    _write_mat(path, time, signal)

    with pytest.raises(ValueError, match="outside"):
        validate_and_load_signal(
            path,
            configured_fs=2.0,
            min_duration_seconds=0.0,
            min_sampling_rate=10.0,
        )
