"""Validated loading of machining-vibration MAT recordings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and at least {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}; got {value!r}.")
    return result


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    number = _finite_number(value, name, minimum=float(minimum))
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer of at least {minimum}; got {value!r}.")
    return int(number)


def validate_and_load_signal(
    file_path: str | Path,
    configured_fs: float,
    tolerance: float = 0.05,
    min_duration_seconds: float = 1.0,
    signal_column: int = 1,
    max_timestamp_jitter: float = 0.05,
    min_sampling_rate: float = 1.0,
    max_sampling_rate: float = 1.0e7,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load and validate a real-valued ``tsDS`` time/signal matrix.

    Timestamp jitter is the largest absolute deviation from the median sample
    interval, divided by that median interval. The estimated sampling rate is
    ``1 / median(diff(time))`` and must agree with ``configured_fs`` within the
    relative ``tolerance``.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Vibration file not found: {path}")

    configured_rate = _finite_number(configured_fs, "configured_fs", minimum=0.0)
    if configured_rate <= 0.0:
        raise ValueError("configured_fs must be positive.")
    relative_tolerance = _finite_number(tolerance, "tolerance", minimum=0.0)
    if relative_tolerance >= 1.0:
        raise ValueError("tolerance must be in [0, 1).")
    duration_requirement = _finite_number(
        min_duration_seconds, "min_duration_seconds", minimum=0.0
    )
    jitter_tolerance = _finite_number(
        max_timestamp_jitter, "max_timestamp_jitter", minimum=0.0
    )
    if jitter_tolerance >= 1.0:
        raise ValueError("max_timestamp_jitter must be in [0, 1).")
    minimum_rate = _finite_number(min_sampling_rate, "min_sampling_rate", minimum=0.0)
    maximum_rate = _finite_number(max_sampling_rate, "max_sampling_rate", minimum=0.0)
    if minimum_rate <= 0.0 or maximum_rate <= minimum_rate:
        raise ValueError("Sampling-rate bounds must satisfy 0 < minimum < maximum.")
    sample_requirement = _positive_integer(min_samples, "min_samples", minimum=2)
    selected_column = _positive_integer(signal_column, "signal_column", minimum=0)

    try:
        mat_data = scipy.io.loadmat(path)
    except Exception as exc:  # SciPy exposes several backend-specific parse errors.
        raise ValueError(f"Failed to read MAT file {path}: {exc}") from exc
    if "tsDS" not in mat_data:
        raise ValueError(f"MAT file {path} is missing 'tsDS' variable.")

    matrix = np.asarray(mat_data["tsDS"])
    if matrix.size == 0:
        raise ValueError(f"tsDS is empty in {path}.")
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(
            f"tsDS must be a 2D matrix with at least two columns; got {matrix.shape}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"tsDS must be numeric in {path}; got {matrix.dtype}.")
    if selected_column >= matrix.shape[1]:
        raise ValueError(
            f"signal_column={selected_column} is out of range for {matrix.shape[1]} columns."
        )

    time_values = matrix[:, 0]
    signal_values = matrix[:, selected_column]
    if np.iscomplexobj(time_values) or np.iscomplexobj(signal_values):
        raise ValueError(f"tsDS contains complex-valued data in {path}.")
    time_array = np.asarray(time_values, dtype=float)
    signal_array = np.asarray(signal_values, dtype=float)
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(signal_array)):
        raise ValueError("Time and signal columns must contain only finite values.")
    if signal_array.size < sample_requirement:
        raise ValueError(
            f"Signal contains too few samples ({signal_array.size} < {sample_requirement})."
        )

    intervals = np.diff(time_array)
    if np.any(intervals <= 0.0):
        raise ValueError("Time values must be strictly increasing without duplicates.")
    median_interval = float(np.median(intervals))
    relative_jitter = float(
        np.max(np.abs(intervals - median_interval)) / median_interval
    )
    if relative_jitter > jitter_tolerance:
        raise ValueError(
            f"Timestamp jitter ({relative_jitter:.6g}) exceeds the configured maximum "
            f"({jitter_tolerance:.6g})."
        )

    signal_scale = float(np.max(np.abs(signal_array)))
    if signal_scale <= np.finfo(float).tiny or float(np.std(signal_array)) <= (
        np.finfo(float).eps * signal_scale
    ):
        raise ValueError("Vibration signal is constant or numerically flat.")

    estimated_rate = 1.0 / median_interval
    if estimated_rate < minimum_rate or estimated_rate > maximum_rate:
        raise ValueError(
            f"Estimated sampling rate ({estimated_rate:.9g} Hz) is outside "
            f"[{minimum_rate:.9g}, {maximum_rate:.9g}] Hz."
        )
    relative_rate_error = abs(estimated_rate - configured_rate) / configured_rate
    if relative_rate_error > relative_tolerance:
        raise ValueError(
            f"Estimated sampling rate ({estimated_rate:.9g} Hz) differs from configured "
            f"rate ({configured_rate:.9g} Hz) by {relative_rate_error:.3%}, exceeding "
            f"the {relative_tolerance:.3%} tolerance."
        )

    duration = signal_array.size / estimated_rate
    if duration < duration_requirement - 1e-6:
        raise ValueError(
            f"Signal duration ({duration:.6g} s) is shorter than the required "
            f"{duration_requirement:.6g} s."
        )
    return time_array.copy(), signal_array.copy(), float(estimated_rate)
