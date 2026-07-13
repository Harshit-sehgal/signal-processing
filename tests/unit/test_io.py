import pytest
import numpy as np
import scipy.io
import os
import tempfile

from pg_amcd.io import validate_and_load_signal

def create_temp_mat(data_dict):
    fd, path = tempfile.mkstemp(suffix=".mat")
    os.close(fd)
    scipy.io.savemat(path, data_dict)
    return path

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_and_load_signal("non_existent_file.mat", 10000.0)

def test_missing_tsds():
    path = create_temp_mat({"wrong_variable": np.zeros((10, 2))})
    try:
        with pytest.raises(ValueError, match="is missing 'tsDS'"):
            validate_and_load_signal(path, 10000.0)
    finally:
        os.remove(path)

def test_invalid_shape():
    path = create_temp_mat({"tsDS": np.zeros((10, 1))}) # 2D but only 1 column
    try:
        with pytest.raises(ValueError, match="must be a 2D matrix"):
            validate_and_load_signal(path, 10000.0)
    finally:
        os.remove(path)

def test_nan_values():
    tsDS = np.zeros((100, 2))
    tsDS[:, 0] = np.arange(100) * 0.0001
    tsDS[50, 1] = np.nan
    path = create_temp_mat({"tsDS": tsDS})
    try:
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_and_load_signal(path, 10000.0)
    finally:
        os.remove(path)

def test_non_increasing_time():
    tsDS = np.zeros((100, 2))
    tsDS[:, 0] = np.arange(100) * 0.0001
    tsDS[50, 0] = 0.0001 # duplicated time
    tsDS[:, 1] = np.random.randn(100)
    path = create_temp_mat({"tsDS": tsDS})
    try:
        with pytest.raises(ValueError, match="not strictly increasing"):
            validate_and_load_signal(path, 10000.0)
    finally:
        os.remove(path)

def test_zero_variance():
    tsDS = np.zeros((100, 2))
    tsDS[:, 0] = np.arange(100) * 0.0001
    tsDS[:, 1] = 5.0 # constant signal
    path = create_temp_mat({"tsDS": tsDS})
    try:
        with pytest.raises(ValueError, match="variance is zero"):
            validate_and_load_signal(path, 10000.0)
    finally:
        os.remove(path)

def test_fs_tolerance_exceeded():
    tsDS = np.zeros((100, 2))
    tsDS[:, 0] = np.arange(100) * 0.0002 # 5000 Hz instead of 10000 Hz
    tsDS[:, 1] = np.random.randn(100)
    path = create_temp_mat({"tsDS": tsDS})
    try:
        with pytest.raises(ValueError, match="deviates from configured rate"):
            validate_and_load_signal(path, 10000.0, tolerance=0.05)
    finally:
        os.remove(path)

def test_valid_signal():
    tsDS = np.zeros((10001, 2))
    tsDS[:, 0] = np.arange(10001) * 0.0001 # exactly 1.0s duration
    tsDS[:, 1] = np.random.randn(10001)
    path = create_temp_mat({"tsDS": tsDS})
    try:
        time, signal, fs_est = validate_and_load_signal(path, 10000.0)
        assert len(time) == 10001
        assert len(signal) == 10001
        assert abs(fs_est - 10000.0) < 1e-5
    finally:
        os.remove(path)
