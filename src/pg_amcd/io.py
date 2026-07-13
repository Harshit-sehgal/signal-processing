import os
import numpy as np
import scipy.io
from typing import Tuple

def validate_and_load_signal(
    file_path: str,
    configured_fs: float,
    tolerance: float = 0.05,
    min_duration_seconds: float = 1.0,
    signal_column: int = 1,
    max_timestamp_jitter: float = 0.05,
    min_sampling_rate: float = 1.0,
    max_sampling_rate: float = 1.0e7,
    min_samples: int = 2,
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

    if "tsDS" not in mat_data:
        raise ValueError(f"MAT file {file_path} is missing 'tsDS' variable.")

    tsDS = mat_data["tsDS"]

    # 0. Empty array
    if tsDS.size == 0:
        raise ValueError(f"tsDS is empty in {file_path}.")

    # 1. Numeric array
    if not np.issubdtype(tsDS.dtype, np.number):
        raise ValueError(f"tsDS must be a numeric array in {file_path}; got {tsDS.dtype}.")

    # 2. Shape: at least 2 columns
    if len(tsDS.shape) != 2 or tsDS.shape[1] < 2:
        raise ValueError(f"tsDS must be a 2D matrix with at least 2 columns. Got shape: {tsDS.shape}")

    n_cols = tsDS.shape[1]
    if signal_column < 0 or signal_column >= n_cols:
        raise ValueError(f"signal_column={signal_column} out of range for tsDS with {n_cols} columns.")

    time = tsDS[:, 0]
    signal = tsDS[:, signal_column]

    # 3. Complex-valued input
    if np.iscomplexobj(time) or np.iscomplexobj(signal):
        raise ValueError(f"tsDS contains complex-valued data in {file_path}.")

    # 4. NaN / Inf check
    if np.any(np.isnan(time)) or np.any(np.isnan(signal)):
        raise ValueError("Vibration signal contains NaN values.")
    if np.any(np.isinf(time)) or np.any(np.isinf(signal)):
        raise ValueError("Vibration signal contains infinite values.")

    # 5. Size and duration checks
    N = len(signal)
    if N < min_samples:
        raise ValueError(f"Signal contains too few samples ({N} < {min_samples}).")

    # 6. Strictly increasing time + duplicate timestamps
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time column values are not strictly increasing (duplicate or decreasing timestamps).")

    # 7. Timestamp jitter (irregular sampling)
    if len(dt) > 1:
        relative_jitter = float(np.std(dt) / np.mean(dt))
        if relative_jitter > max_timestamp_jitter:
            raise ValueError(
                f"Timestamp jitter ({relative_jitter:.4f}) exceeds maximum allowed "
                f"({max_timestamp_jitter:.4f}); sampling is irregular."
            )

    # 8. Non-zero variance check
    sig_var = np.var(signal)
    if sig_var == 0:
        raise ValueError("Vibration signal variance is zero (constant signal).")

    # 9. Sampling rate estimation & tolerance validation
    fs_estimated = 1.0 / np.median(dt)
    if fs_estimated < min_sampling_rate or fs_estimated > max_sampling_rate:
        raise ValueError(
            f"Estimated sampling rate ({fs_estimated:.2f} Hz) is outside the realistic "
            f"range [{min_sampling_rate}, {max_sampling_rate}] Hz."
        )
    if abs(fs_estimated - configured_fs) / configured_fs > tolerance:
        raise ValueError(
            f"Estimated sampling rate ({fs_estimated:.2f} Hz) deviates from "
            f"configured rate ({configured_fs:.2f} Hz) beyond {tolerance*100}% tolerance."
        )

    # 10. Duration check.
    # Use sample count over the estimated sampling rate so that a signal with
    # exactly N samples at fs Hz is accepted as N/fs seconds (e.g. 10000
    # samples at 10 kHz is 1.0000 s, not 0.9999 s from floating-point time).
    duration = len(signal) / fs_estimated
    # Tolerance absorbs floating-point noise in fs_estimated so that a signal
    # with exactly N samples at the rated fs (N/fs seconds) is not
    # rejected at the boundary.
    if duration < min_duration_seconds - 1e-6:
        raise ValueError(
            f"Signal duration ({duration:.4f}s) is shorter than minimum "
            f"required ({min_duration_seconds}s)."
        )

    return time, signal, fs_estimated
