"""Amplitude-traceable preprocessing for PG-AMCD Stage 1.

The public :func:`preprocess_signal` tuple API is retained for compatibility.
New code can use :func:`preprocess_signal_result` to retain the resolved filter
and scaling parameters alongside the physical and numerically scaled signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal


@dataclass(frozen=True)
class PreprocessingParameters:
    """Resolved, serialisable preprocessing parameters.

    ``highpass_cutoff_hz`` is the lower edge of the pass band and
    ``lowpass_cutoff_hz`` is the upper edge.  The explicit names avoid the
    historically ambiguous term ``cutoff``.
    """

    highpass_cutoff_hz: float
    lowpass_cutoff_hz: float
    sampling_rate_hz: float
    filter_order: int = 3
    detrend_type: str = "linear"
    detrend_before_filter: bool = False
    scale_percentile: float = 99.5
    padtype: Optional[str] = "odd"
    padlen: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["high_pass_cutoff_hz"] = self.highpass_cutoff_hz
        values["low_pass_cutoff_hz"] = self.lowpass_cutoff_hz
        return values

    @property
    def high_pass_cutoff_hz(self) -> float:
        """Canonical alias for the lower/high-pass band edge."""

        return self.highpass_cutoff_hz

    @property
    def low_pass_cutoff_hz(self) -> float:
        """Canonical alias for the upper/low-pass band edge."""

        return self.lowpass_cutoff_hz


@dataclass(frozen=True)
class PreprocessingResult:
    """Physical and scaled preprocessing outputs with full scale traceability."""

    physical_signal: np.ndarray
    scaled_signal: np.ndarray
    scale_factor: float
    parameters: PreprocessingParameters

    def restore_physical(self, scaled_signal: np.ndarray) -> np.ndarray:
        """Restore a compatible scaled signal to the original physical units."""

        return np.asarray(scaled_signal, dtype=float) * self.scale_factor


def _as_finite_1d_signal(signal: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(signal):
        raise ValueError("Signal must be real-valued.")
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Signal must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Signal must contain at least one sample.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or infinite values.")
    return arr


def _validate_filter_parameters(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int,
    padlen: Optional[int],
) -> Tuple[np.ndarray, float, float, float, int, Optional[int]]:
    arr = _as_finite_1d_signal(signal)
    fs = float(fs)
    low_cutoff = float(low_cutoff)
    high_cutoff = float(high_cutoff)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling rate must be a finite positive value, got {fs}.")
    if not np.isfinite(low_cutoff) or not np.isfinite(high_cutoff):
        raise ValueError("Filter cutoffs must be finite.")
    try:
        numeric_order = float(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Filter order must be a positive integer, got {order}.") from exc
    if (
        isinstance(order, bool)
        or not np.isfinite(numeric_order)
        or not numeric_order.is_integer()
        or numeric_order <= 0
    ):
        raise ValueError(f"Filter order must be a positive integer, got {order}.")
    order = int(numeric_order)

    nyquist = 0.5 * fs
    if low_cutoff <= 0 or high_cutoff >= nyquist or low_cutoff >= high_cutoff:
        raise ValueError(
            "Invalid band-pass bounds: "
            f"high-pass={low_cutoff:g} Hz, low-pass={high_cutoff:g} Hz, "
            f"Nyquist={nyquist:g} Hz. Required: 0 < high-pass < low-pass < Nyquist."
        )
    if padlen is not None:
        try:
            numeric_padlen = float(padlen)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"padlen must be a non-negative integer or None, got {padlen}."
            ) from exc
        if (
            isinstance(padlen, bool)
            or not np.isfinite(numeric_padlen)
            or not numeric_padlen.is_integer()
            or numeric_padlen < 0
        ):
            raise ValueError(f"padlen must be a non-negative integer or None, got {padlen}.")
        padlen = int(numeric_padlen)
        if arr.size <= padlen:
            raise ValueError(
                f"Signal length {arr.size} must be greater than configured padlen {padlen}."
            )

    return arr, low_cutoff, high_cutoff, fs, order, padlen


def butter_bandpass_filter_sos(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 3,
    *,
    padtype: Optional[str] = "odd",
    padlen: Optional[int] = None,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass using second-order sections.

    The historical parameter names are retained: ``low_cutoff`` is the lower
    pass-band edge (the high-pass cutoff) and ``high_cutoff`` is the upper edge
    (the low-pass cutoff).

    ``scipy.signal.sosfiltfilt`` chooses a section-dependent padding length.
    We deliberately let SciPy compute and validate that default instead of
    using the old and incorrect ``2 * order + 1`` approximation.
    """

    arr, low_cutoff, high_cutoff, fs, order, padlen = _validate_filter_parameters(
        signal, low_cutoff, high_cutoff, fs, order, padlen
    )
    sos = scipy.signal.butter(
        order,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    try:
        filtered = scipy.signal.sosfiltfilt(
            sos,
            arr,
            padtype=padtype,
            padlen=padlen,
        )
    except ValueError as exc:
        # Preserve SciPy's exact pad length in a Stage-1-friendly error while
        # making the failed operation clear to CLI callers.
        raise ValueError(
            f"Signal length {arr.size} is too short for zero-phase SOS filtering: {exc}"
        ) from exc

    if not np.all(np.isfinite(filtered)):
        raise ValueError("SOS filtering produced non-finite values.")
    return np.asarray(filtered, dtype=float)


def preprocess_signal_result(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 3,
    *,
    detrend_type: str = "linear",
    detrend_before_filter: bool = False,
    scale_percentile: float = 99.5,
    padtype: Optional[str] = "odd",
    padlen: Optional[int] = None,
) -> PreprocessingResult:
    """Detrend, filter, and separately scale a signal for numerical processing.

    The physical signal is never normalised.  A single robust amplitude scale
    is derived from it and returned explicitly, so every later-stage scaled
    signal can be traced back to the physical units.
    """

    arr = _as_finite_1d_signal(signal)
    if detrend_type not in {"linear", "constant"}:
        raise ValueError(
            f"detrend_type must be 'linear' or 'constant', got {detrend_type!r}."
        )
    scale_percentile = float(scale_percentile)
    if not np.isfinite(scale_percentile) or not 0.0 < scale_percentile <= 100.0:
        raise ValueError(
            f"scale_percentile must be in (0, 100], got {scale_percentile}."
        )

    if detrend_before_filter:
        detrended = scipy.signal.detrend(arr, type=detrend_type)
        physical = butter_bandpass_filter_sos(
            detrended,
            low_cutoff,
            high_cutoff,
            fs,
            order=order,
            padtype=padtype,
            padlen=padlen,
        )
    else:
        filtered = butter_bandpass_filter_sos(
            arr,
            low_cutoff,
            high_cutoff,
            fs,
            order=order,
            padtype=padtype,
            padlen=padlen,
        )
        physical = scipy.signal.detrend(filtered, type=detrend_type)

    physical = np.asarray(physical, dtype=float)
    if not np.all(np.isfinite(physical)):
        raise ValueError("Preprocessing produced non-finite physical values.")

    scale_factor = float(np.percentile(np.abs(physical), scale_percentile))
    if not np.isfinite(scale_factor) or scale_factor <= np.finfo(float).tiny:
        raise ValueError(
            "Preprocessed signal has no numerically meaningful amplitude; "
            "cannot derive a stable scale factor."
        )
    scaled = physical / scale_factor
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaling produced non-finite values.")

    params = PreprocessingParameters(
        highpass_cutoff_hz=float(low_cutoff),
        lowpass_cutoff_hz=float(high_cutoff),
        sampling_rate_hz=float(fs),
        filter_order=int(order),
        detrend_type=detrend_type,
        detrend_before_filter=bool(detrend_before_filter),
        scale_percentile=scale_percentile,
        padtype=padtype,
        padlen=None if padlen is None else int(padlen),
    )
    return PreprocessingResult(
        physical_signal=physical,
        scaled_signal=np.asarray(scaled, dtype=float),
        scale_factor=scale_factor,
        parameters=params,
    )


def preprocess_signal(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 3,
    *,
    detrend_type: str = "linear",
    detrend_before_filter: bool = False,
    scale_percentile: float = 99.5,
    padtype: Optional[str] = "odd",
    padlen: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper returning physical signal, scaled signal, scale.

    Prefer :func:`preprocess_signal_result` when provenance parameters are also
    required.
    """

    result = preprocess_signal_result(
        signal,
        low_cutoff,
        high_cutoff,
        fs,
        order=order,
        detrend_type=detrend_type,
        detrend_before_filter=detrend_before_filter,
        scale_percentile=scale_percentile,
        padtype=padtype,
        padlen=padlen,
    )
    return result.physical_signal, result.scaled_signal, result.scale_factor
