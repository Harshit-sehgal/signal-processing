import os
import numpy as np
import scipy.io
from typing import Tuple, Dict, Any

def validate_and_load_signal(
    file_path: str, 
    configured_fs: float, 
    tolerance: float = 0.05,
    min_duration_seconds: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Loads a MAT file containing a tsDS matrix, performs validation,
    estimates the sampling rate, and returns the time and signal arrays.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any validation checks fail.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vibration file not found: {file_path}")
        
    try:
        mat_data = scipy.io.loadmat(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read MAT file {file_path}: {e}")
        
    if 'tsDS' not in mat_data:
        raise ValueError(f"MAT file {file_path} is missing 'tsDS' variable.")
        
    tsDS = mat_data['tsDS']
    
    # 1. Shape check
    if len(tsDS.shape) != 2 or tsDS.shape[1] < 2:
        raise ValueError(f"tsDS must be a 2D matrix with at least 2 columns. Got shape: {tsDS.shape}")
        
    time = tsDS[:, 0]
    signal = tsDS[:, 1]
    
    # 2. NaN/Inf check
    if np.any(np.isnan(time)) or np.any(np.isnan(signal)):
        raise ValueError("Vibration signal contains NaN values.")
    if np.any(np.isinf(time)) or np.any(np.isinf(signal)):
        raise ValueError("Vibration signal contains infinite values.")
        
    # 3. Size and duration checks
    N = len(signal)
    if N < 2:
        raise ValueError(f"Signal contains too few samples ({N}).")
        
    # 4. Strictly increasing time check
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time column values are not strictly increasing.")
        
    # 5. Non-zero variance check
    sig_var = np.var(signal)
    if sig_var == 0:
        raise ValueError("Vibration signal variance is zero (constant signal).")
        
    # 6. Sampling rate estimation & tolerance validation
    fs_estimated = 1.0 / np.median(dt)
    if abs(fs_estimated - configured_fs) / configured_fs > tolerance:
        raise ValueError(
            f"Estimated sampling rate ({fs_estimated:.2f} Hz) deviates from "
            f"configured rate ({configured_fs:.2f} Hz) beyond {tolerance*100}% tolerance."
        )
        
    duration = time[-1] - time[0]
    if duration < min_duration_seconds:
        raise ValueError(f"Signal duration ({duration:.2f}s) is shorter than minimum required ({min_duration_seconds}s).")
        
    return time, signal, fs_estimated
