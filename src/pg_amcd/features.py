"""Stage 4: versioned, transparent multi-family feature extraction.

Stage 4 stops at extraction: this module contains no feature selection,
classifier, probability, or decision logic.  The canonical typed API is
:func:`extract_window_feature_result`; :func:`extract_window_features` remains
as a finite-value compatibility wrapper for existing callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pywt
import scipy.signal
import scipy.stats


FEATURE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureDefinition:
    """One entry in the versioned Stage 4 feature schema."""

    name: str
    family: str
    description: str
    unit: str
    required_source_stage: str
    metadata_required: bool
    dimensionless: bool
    undefined_handling: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Feature values plus explicit undefined reasons and diagnostic traces."""

    values: Dict[str, Optional[float]]
    undefined_reasons: Dict[str, str]
    definitions: Tuple[FeatureDefinition, ...]
    traces: Dict[str, np.ndarray]
    quality: Dict[str, Any]
    schema_version: str = FEATURE_SCHEMA_VERSION

    def finite_values(self, undefined_fill: float = 0.0) -> Dict[str, float]:
        """Return legacy finite scalars, replacing only explicitly undefined values."""

        fill = float(undefined_fill)
        if not math.isfinite(fill):
            raise ValueError("undefined_fill must be finite.")
        return {
            name: fill if value is None else float(value) for name, value in self.values.items()
        }

    def schema_dict(self) -> Dict[str, Any]:
        return {
            "feature_schema_version": self.schema_version,
            "features": [definition.to_dict() for definition in self.definitions],
        }

    def to_dict(self, *, include_traces: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "feature_schema_version": self.schema_version,
            "features": dict(self.values),
            "undefined_features": dict(self.undefined_reasons),
            "quality": dict(self.quality),
        }
        if include_traces:
            payload["traces"] = {name: values.tolist() for name, values in self.traces.items()}
        return payload


@dataclass(frozen=True)
class WindowFeatureRecord:
    """One deterministic sliding-window result."""

    window_index: int
    start_index: int
    end_index: int
    start_time_seconds: float
    end_time_seconds: float
    result: FeatureExtractionResult


@dataclass(frozen=True)
class FeatureAggregateResult:
    """Run-level tables computable directly from per-window typed results."""

    rows: Tuple[Dict[str, Any], ...]
    summary: Dict[str, Dict[str, Optional[float]]]
    missingness: Dict[str, Dict[str, Any]]
    correlations: Dict[str, Dict[str, Optional[float]]]
    schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": list(self.rows),
            "summary": self.summary,
            "missingness": self.missingness,
            "correlations": self.correlations,
            "schema": self.schema,
        }


def _definition(
    name: str,
    family: str,
    description: str,
    unit: str = "dimensionless",
    stage: str = "Stage_3",
    metadata: bool = False,
    dimensionless: bool = True,
    undefined: str = "Stored as null with a reason when its denominator or input is invalid.",
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        family=family,
        description=description,
        unit=unit,
        required_source_stage=stage,
        metadata_required=metadata,
        dimensionless=dimensionless,
        undefined_handling=undefined,
    )


_BASE_DEFINITION_LIST = [
    # Time domain
    _definition("time_rms", "time", "sqrt(mean(x^2))", "signal unit", dimensionless=False),
    _definition(
        "time_variance", "time", "population variance of x", "signal unit^2", dimensionless=False
    ),
    _definition(
        "time_std", "time", "population standard deviation of x", "signal unit", dimensionless=False
    ),
    _definition("time_peak_to_peak", "time", "max(x)-min(x)", "signal unit", dimensionless=False),
    _definition("time_mean_absolute", "time", "mean(abs(x))", "signal unit", dimensionless=False),
    _definition("time_crest_factor", "time", "max(abs(x))/RMS"),
    _definition("time_kurtosis", "time", "Pearson kurtosis E[(x-mu)^4]/sigma^4"),
    _definition("time_skewness", "time", "standardised third central moment"),
    _definition("time_impulse_factor", "time", "max(abs(x))/mean(abs(x))"),
    _definition("time_shape_factor", "time", "RMS/mean(abs(x))"),
    _definition(
        "time_clearance_factor",
        "time",
        "max(abs(x))/mean(sqrt(abs(x)))^2",
    ),
    _definition("time_zero_crossing_rate", "time", "sign changes divided by N-1"),
    # Frequency domain
    _definition(
        "freq_centroid", "frequency", "PSD-weighted mean frequency", "Hz", dimensionless=False
    ),
    _definition(
        "freq_spread",
        "frequency",
        "PSD-weighted standard deviation around centroid",
        "Hz",
        dimensionless=False,
    ),
    _definition(
        "freq_entropy",
        "frequency",
        "Shannon entropy of normalised PSD divided by log2(number of bins)",
    ),
    _definition(
        "freq_peak", "frequency", "frequency of maximum Welch PSD", "Hz", dimensionless=False
    ),
    _definition(
        "freq_chatter_band_ratio",
        "frequency",
        "chatter-band PSD energy divided by total PSD energy",
    ),
    _definition(
        "freq_spindle_harmonic_ratio",
        "frequency",
        "PSD energy near spindle harmonics divided by total PSD energy",
        metadata=True,
    ),
    _definition(
        "freq_tooth_harmonic_ratio",
        "frequency",
        "PSD energy near tooth-passing harmonics divided by total PSD energy",
        metadata=True,
    ),
    _definition(
        "freq_harmonics_ratio",
        "frequency",
        "compatibility alias of tooth-passing harmonic energy ratio",
        metadata=True,
    ),
    _definition(
        "freq_sideband_ratio",
        "frequency",
        "tooth sideband energy divided by tooth-harmonic energy",
        metadata=True,
    ),
    _definition(
        "freq_spectral_kurtosis", "frequency", "Pearson kurtosis across non-negative Welch PSD bins"
    ),
    # IMF domain
    _definition(
        "imf_count",
        "imf",
        "number of physical IMFs after explicit residual exclusion",
        "count",
        "Stage_1",
        dimensionless=False,
    ),
    _definition(
        "imf_max_energy_ratio", "imf", "maximum physical-IMF energy fraction", stage="Stage_1"
    ),
    _definition(
        "imf1_correlation",
        "imf",
        "absolute correlation of IMF 1 with the preprocessed source",
        stage="Stage_1",
    ),
    _definition(
        "imf_centre_freq_mean",
        "imf",
        "mean physical-IMF centre frequency",
        "Hz",
        "Stage_1",
        dimensionless=False,
    ),
    _definition(
        "imf_bandwidth_mean",
        "imf",
        "mean physical-IMF PSD bandwidth",
        "Hz",
        "Stage_1",
        dimensionless=False,
    ),
    _definition(
        "imf_entropy_mean", "imf", "mean normalised physical-IMF spectral entropy", stage="Stage_1"
    ),
    _definition(
        "imf_mode_mixing_index",
        "imf",
        "mean absolute adjacent physical-IMF correlation",
        stage="Stage_1",
    ),
    _definition(
        "imf_max_adjacent_correlation",
        "imf",
        "maximum absolute adjacent physical-IMF correlation",
        stage="Stage_1",
    ),
    _definition(
        "imf_orthogonality_index", "imf", "2*sum(i<j)<c_i,c_j>/sum_i||c_i||^2", stage="Stage_1"
    ),
    _definition(
        "imf_frequency_ordering_score",
        "imf",
        "fraction of adjacent physical IMF centre frequencies in non-increasing order",
        stage="Stage_1",
    ),
    _definition(
        "imf_selected_count",
        "imf",
        "number of gates at or above the configured selection threshold",
        "count",
        "Stage_2",
        dimensionless=False,
    ),
    # Wavelet / time-frequency
    _definition(
        "wavelet_high_freq_ratio",
        "wavelet",
        "energy in cD1 and cD2 divided by total coefficient energy",
    ),
    _definition(
        "wavelet_entropy",
        "wavelet",
        "Shannon entropy of normalised coefficient energies divided by log2(number of subbands)",
    ),
    _definition(
        "wavelet_time_frequency_concentration",
        "wavelet",
        "maximum STFT-bin energy divided by total STFT energy",
    ),
    _definition(
        "wavelet_dominant_ridge_hz",
        "wavelet",
        "median frequency of the maximum-energy STFT bin per frame",
        "Hz",
        dimensionless=False,
    ),
    # Early chatter
    _definition(
        "early_instantaneous_amplitude_mean",
        "early_chatter",
        "mean magnitude of scipy.signal.hilbert(x)",
        "signal unit",
        dimensionless=False,
    ),
    _definition(
        "early_instantaneous_amplitude_max",
        "early_chatter",
        "maximum magnitude of scipy.signal.hilbert(x)",
        "signal unit",
        dimensionless=False,
    ),
    _definition(
        "early_instantaneous_energy_mean",
        "early_chatter",
        "mean squared analytic-signal magnitude",
        "signal unit^2",
        dimensionless=False,
    ),
    _definition(
        "early_instantaneous_energy_max",
        "early_chatter",
        "maximum squared analytic-signal magnitude",
        "signal unit^2",
        dimensionless=False,
    ),
    _definition(
        "early_energy_growth_rate",
        "early_chatter",
        "least-squares slope of smoothed Hilbert instantaneous energy versus time",
        "signal unit^2/s",
        dimensionless=False,
    ),
    _definition(
        "early_hegr",
        "early_chatter",
        "Hilbert Energy Growth Rate: mean(max(dE_s(t)/dt, 0)), where E_s is a 10 ms moving-average of |hilbert(x)|^2",
        "signal unit^2/s",
        dimensionless=False,
    ),
    _definition(
        "early_chatter_band_energy_growth",
        "early_chatter",
        "least-squares slope of per-frame chatter-band STFT energy",
        "signal unit^2/s",
        dimensionless=False,
    ),
    _definition(
        "early_short_term_spectral_energy_growth",
        "early_chatter",
        "least-squares slope of total per-frame STFT energy",
        "signal unit^2/s",
        dimensionless=False,
    ),
    # Physics guided
    _definition(
        "physics_chatter_band_energy",
        "physics",
        "Welch PSD energy within the configured chatter band",
        "signal unit^2",
        metadata=False,
        dimensionless=False,
    ),
    _definition(
        "physics_frequency_proximity",
        "physics",
        "Gaussian proximity of peak frequency to the configured chatter-band interval",
    ),
    _definition(
        "physics_spindle_frequency_hz",
        "physics",
        "RPM/60",
        "Hz",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_tooth_passing_frequency_hz",
        "physics",
        "RPM*tooth_count/60",
        "Hz",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_peak_to_spindle_harmonic_distance_hz",
        "physics",
        "distance from spectral peak to nearest spindle harmonic",
        "Hz",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_peak_to_tooth_harmonic_distance_hz",
        "physics",
        "distance from spectral peak to nearest tooth-passing harmonic",
        "Hz",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_sideband_strength",
        "physics",
        "Welch PSD energy around tooth harmonics plus/minus spindle frequency",
        "signal unit^2",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_forced_vibration_energy",
        "physics",
        "Welch PSD energy in the union of spindle and tooth harmonic windows",
        "signal unit^2",
        metadata=True,
        dimensionless=False,
    ),
    _definition(
        "physics_chatter_to_harmonic_energy_ratio",
        "physics",
        "chatter-band energy divided by forced-harmonic energy",
        metadata=True,
    ),
]

_BASE_DEFINITIONS = {definition.name: definition for definition in _BASE_DEFINITION_LIST}


def feature_schema() -> Dict[str, Any]:
    """Return the static, versioned schema (dynamic IMF/level entries are per result)."""

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": [definition.to_dict() for definition in _BASE_DEFINITION_LIST],
        "dynamic_feature_patterns": [
            "freq_band_energy_ratio_{band_name}",
            "imf_{index}_{energy_ratio|centre_frequency_hz|bandwidth_hz|spectral_entropy|kurtosis|source_correlation|gate}",
            "wavelet_{coefficient_name}_{energy|energy_ratio}",
        ],
    }


class _FeatureCollector:
    def __init__(self) -> None:
        self.values: Dict[str, Optional[float]] = {}
        self.undefined: Dict[str, str] = {}
        self.definitions: Dict[str, FeatureDefinition] = {}

    def add(
        self,
        name: str,
        value: Optional[float],
        *,
        reason: Optional[str] = None,
        definition: Optional[FeatureDefinition] = None,
    ) -> None:
        resolved_definition = definition or _BASE_DEFINITIONS.get(name)
        if resolved_definition is None:
            raise KeyError(f"Feature {name!r} has no schema definition.")
        self.definitions[name] = resolved_definition
        if value is None:
            self.values[name] = None
            self.undefined[name] = reason or "Feature is undefined for this window."
            return
        number = float(value)
        if not math.isfinite(number):
            self.values[name] = None
            self.undefined[name] = reason or "Calculation produced a non-finite value."
            return
        self.values[name] = number


def _as_finite_signal(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _finite_positive(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denominator <= np.finfo(float).eps:
        return None
    return float(abs(np.dot(ac, bc) / denominator))


def _pearson_kurtosis(values: np.ndarray) -> Optional[float]:
    if float(np.std(values)) <= np.finfo(float).eps:
        return None
    result = float(scipy.stats.kurtosis(values, fisher=False, bias=False))
    return result if math.isfinite(result) else None


def _skewness(values: np.ndarray) -> Optional[float]:
    if float(np.std(values)) <= np.finfo(float).eps:
        return None
    result = float(scipy.stats.skew(values, bias=False))
    return result if math.isfinite(result) else None


def _normalised_entropy(nonnegative_values: np.ndarray) -> Optional[float]:
    total = float(np.sum(nonnegative_values))
    if total <= 0 or nonnegative_values.size <= 1:
        return None
    probability = nonnegative_values / total
    probability = probability[probability > 0]
    entropy = -float(np.sum(probability * np.log2(probability)))
    return entropy / math.log2(nonnegative_values.size)


def _band_power(freqs: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    if freqs.size <= 1:
        return float(np.sum(psd[mask]))
    return float(np.sum(psd[mask]) * np.mean(np.diff(freqs)))


def _harmonic_mask(
    freqs: np.ndarray, fundamental: float, count: int, tolerance_hz: float
) -> np.ndarray:
    mask = np.zeros(freqs.shape, dtype=bool)
    for order in range(1, count + 1):
        harmonic = fundamental * order
        if harmonic > freqs[-1] + tolerance_hz:
            break
        mask |= np.abs(freqs - harmonic) <= tolerance_hz
    return mask


def _linear_slope(time_values: np.ndarray, values: np.ndarray) -> Optional[float]:
    if time_values.size < 2 or float(np.ptp(time_values)) <= 0:
        return None
    centered_time = time_values - np.mean(time_values)
    denominator = float(np.sum(np.square(centered_time)))
    if denominator <= 0:
        return None
    return float(np.sum(centered_time * (values - np.mean(values))) / denominator)


def _dynamic_definition(
    name: str,
    family: str,
    description: str,
    unit: str,
    stage: str,
    *,
    metadata: bool = False,
    dimensionless: bool = True,
) -> FeatureDefinition:
    return _definition(
        name,
        family,
        description,
        unit,
        stage,
        metadata,
        dimensionless,
    )


def _valid_physics_metadata(
    rpm: Optional[float], tooth_count: Optional[int]
) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    try:
        rpm_value = float(rpm) if rpm is not None and not isinstance(rpm, bool) else float("nan")
    except (TypeError, ValueError):
        rpm_value = float("nan")
    try:
        tooth_value_float = (
            float(tooth_count)
            if tooth_count is not None and not isinstance(tooth_count, bool)
            else float("nan")
        )
    except (TypeError, ValueError):
        tooth_value_float = float("nan")
    if not math.isfinite(rpm_value) or rpm_value <= 0:
        return None, None, "RPM is missing, non-finite, or non-positive."
    if (
        not math.isfinite(tooth_value_float)
        or not tooth_value_float.is_integer()
        or tooth_value_float < 1
    ):
        return None, None, "tooth_count is missing or is not a positive integer."
    return rpm_value, int(tooth_value_float), None


def extract_window_feature_result(
    raw_window: np.ndarray,
    prep_physical_window: np.ndarray,
    denoised_physical_window: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    rpm: Optional[float],
    tooth_count: Optional[int],
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0,
    *,
    imf_gates: Optional[np.ndarray] = None,
    residual_last_row: bool = True,
    selected_gate_threshold: float = 0.5,
    wavelet_name: str = "db8",
    wavelet_level: int = 4,
    harmonic_count: int = 5,
    harmonic_tolerance_hz: float = 15.0,
    sideband_tolerance_hz: float = 10.0,
    band_energy_ranges: Optional[
        Mapping[str, Tuple[float, float]] | Sequence[Tuple[float, float]]
    ] = None,
    strict_chatter_band: bool = True,
) -> FeatureExtractionResult:
    """Extract all Stage 4 families from one canonical Stage 1--3 window.

    HEGR is defined exactly as ``mean(max(dE_s/dt, 0))`` where
    ``E_s`` is a 10 ms moving average of Hilbert instantaneous energy.  The
    instantaneous-energy and HEGR arrays are returned in ``result.traces`` so
    timeline artifacts do not need to recompute or reinterpret the formula.
    """

    raw = _as_finite_signal(raw_window, "raw_window")
    preprocessed = _as_finite_signal(prep_physical_window, "prep_physical_window")
    denoised = _as_finite_signal(denoised_physical_window, "denoised_physical_window")
    if raw.size != preprocessed.size or raw.size != denoised.size:
        raise ValueError("Raw, preprocessed, and denoised windows must have identical lengths.")
    sampling_rate = _finite_positive(fs, "sampling rate")
    center = _finite_positive(chatter_center, "chatter_center")
    spread = _finite_positive(chatter_spread, "chatter_spread")
    tolerance = _finite_positive(harmonic_tolerance_hz, "harmonic_tolerance_hz")
    sideband_tolerance = _finite_positive(sideband_tolerance_hz, "sideband_tolerance_hz")
    if not isinstance(harmonic_count, (int, np.integer)) or harmonic_count < 1:
        raise ValueError("harmonic_count must be a positive integer.")
    if not 0.0 <= float(selected_gate_threshold) <= 1.0:
        raise ValueError("selected_gate_threshold must be in [0, 1].")

    imf_array = np.asarray(imfs, dtype=float)
    if imf_array.ndim != 2 or imf_array.shape[1] != denoised.size:
        raise ValueError("IMFs must be a 2-D array with the same sample length as the window.")
    if not np.all(np.isfinite(imf_array)):
        raise ValueError("IMFs contain non-finite values.")
    physical_count = imf_array.shape[0] - (1 if residual_last_row else 0)
    if physical_count < 1:
        raise ValueError("At least one physical IMF is required after residual handling.")
    physical_imfs = imf_array[:physical_count]

    gates: Optional[np.ndarray] = None
    if imf_gates is not None:
        gates = np.asarray(imf_gates, dtype=float)
        if gates.ndim != 1 or gates.size != physical_count:
            raise ValueError("IMF gate count must match the physical IMF count.")
        if not np.all(np.isfinite(gates)) or np.any((gates < 0.0) | (gates > 1.0)):
            raise ValueError("IMF gates must be finite and bounded in [0, 1].")

    collector = _FeatureCollector()
    traces: Dict[str, np.ndarray] = {
        "raw_signal": raw.copy(),
        "preprocessed_physical_signal": preprocessed.copy(),
        "denoised_physical_signal": denoised.copy(),
    }

    # ------------------------------------------------------------------ time
    rms = float(np.sqrt(np.mean(np.square(denoised))))
    variance = float(np.var(denoised))
    std = float(np.sqrt(variance))
    max_abs = float(np.max(np.abs(denoised)))
    mean_abs = float(np.mean(np.abs(denoised)))
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(denoised))))
    collector.add("time_rms", rms)
    collector.add("time_variance", variance)
    collector.add("time_std", std)
    collector.add("time_peak_to_peak", float(np.ptp(denoised)))
    collector.add("time_mean_absolute", mean_abs)
    collector.add("time_crest_factor", max_abs / rms if rms > 0 else None, reason="RMS is zero.")
    collector.add("time_kurtosis", _pearson_kurtosis(denoised), reason="Signal variance is zero.")
    collector.add("time_skewness", _skewness(denoised), reason="Signal variance is zero.")
    collector.add(
        "time_impulse_factor",
        max_abs / mean_abs if mean_abs > 0 else None,
        reason="Mean absolute value is zero.",
    )
    collector.add(
        "time_shape_factor",
        rms / mean_abs if mean_abs > 0 else None,
        reason="Mean absolute value is zero.",
    )
    collector.add(
        "time_clearance_factor",
        max_abs / mean_sqrt_abs**2 if mean_sqrt_abs > 0 else None,
        reason="Mean square-root absolute amplitude is zero.",
    )
    crossings = np.count_nonzero(np.signbit(denoised[:-1]) != np.signbit(denoised[1:]))
    collector.add("time_zero_crossing_rate", crossings / (denoised.size - 1))

    # ------------------------------------------------------------ frequency
    freqs, psd = scipy.signal.welch(denoised, fs=sampling_rate, nperseg=min(denoised.size, 1024))
    total_psd = float(np.sum(psd))
    chatter_band = (
        max(0.0, center - spread),
        min(sampling_rate / 2.0, center + spread),
    )
    if chatter_band[0] >= chatter_band[1] and strict_chatter_band:
        raise ValueError("Configured chatter band does not overlap [0, Nyquist].")
    chatter_mask = (freqs >= chatter_band[0]) & (freqs <= chatter_band[1])
    if total_psd > 0:
        psd_probability = psd / total_psd
        centroid = float(np.sum(freqs * psd_probability))
        spectral_spread = float(np.sqrt(np.sum(np.square(freqs - centroid) * psd_probability)))
        collector.add("freq_centroid", centroid)
        collector.add("freq_spread", spectral_spread)
        collector.add("freq_entropy", _normalised_entropy(psd))
        collector.add("freq_peak", float(freqs[np.argmax(psd)]))
        collector.add("freq_chatter_band_ratio", float(np.sum(psd[chatter_mask]) / total_psd))
        collector.add(
            "freq_spectral_kurtosis", _pearson_kurtosis(psd), reason="PSD variance is zero."
        )
    else:
        for feature_name in (
            "freq_centroid",
            "freq_spread",
            "freq_entropy",
            "freq_peak",
            "freq_chatter_band_ratio",
            "freq_spectral_kurtosis",
        ):
            collector.add(feature_name, None, reason="Welch PSD has zero total energy.")

    if band_energy_ranges is None:
        ranges = {
            "below_chatter": (0.0, chatter_band[0]),
            "chatter": chatter_band,
            "above_chatter": (chatter_band[1], sampling_rate / 2.0),
        }
    elif isinstance(band_energy_ranges, Mapping):
        ranges = dict(band_energy_ranges)
    else:
        ranges = {
            f"band_{index + 1}_{float(limits[0]):g}_{float(limits[1]):g}_hz": limits
            for index, limits in enumerate(band_energy_ranges)
        }
    for band_name, limits in ranges.items():
        if len(limits) != 2:
            raise ValueError(f"Band {band_name!r} must contain exactly (low_hz, high_hz).")
        low, high = float(limits[0]), float(limits[1])
        feature_name = f"freq_band_energy_ratio_{band_name}"
        definition = _dynamic_definition(
            feature_name,
            "frequency",
            f"PSD energy in [{low}, {high}] Hz divided by total PSD energy",
            "dimensionless",
            "Stage_3",
        )
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high <= low:
            collector.add(
                feature_name,
                None,
                reason="Configured band bounds are invalid.",
                definition=definition,
            )
        else:
            mask = (freqs >= low) & (freqs <= min(high, sampling_rate / 2.0))
            collector.add(
                feature_name,
                float(np.sum(psd[mask]) / total_psd) if total_psd > 0 else None,
                reason="Welch PSD has zero total energy.",
                definition=definition,
            )

    rpm_value, tooth_value, metadata_error = _valid_physics_metadata(rpm, tooth_count)
    spindle_mask = np.zeros(freqs.shape, dtype=bool)
    tooth_mask = np.zeros(freqs.shape, dtype=bool)
    sideband_mask = np.zeros(freqs.shape, dtype=bool)
    if metadata_error is None and rpm_value is not None and tooth_value is not None:
        spindle_frequency = rpm_value / 60.0
        tooth_frequency = spindle_frequency * tooth_value
        spindle_mask = _harmonic_mask(freqs, spindle_frequency, harmonic_count, tolerance)
        tooth_mask = _harmonic_mask(freqs, tooth_frequency, harmonic_count, tolerance)
        for order in range(1, harmonic_count + 1):
            harmonic = tooth_frequency * order
            for sideband in (harmonic - spindle_frequency, harmonic + spindle_frequency):
                if sideband > 0:
                    sideband_mask |= np.abs(freqs - sideband) <= sideband_tolerance
        spindle_ratio = float(np.sum(psd[spindle_mask]) / total_psd) if total_psd > 0 else None
        tooth_ratio = float(np.sum(psd[tooth_mask]) / total_psd) if total_psd > 0 else None
        tooth_energy = _band_power(freqs, psd, tooth_mask)
        sideband_energy = _band_power(freqs, psd, sideband_mask)
        collector.add(
            "freq_spindle_harmonic_ratio", spindle_ratio, reason="Welch PSD has zero total energy."
        )
        collector.add(
            "freq_tooth_harmonic_ratio", tooth_ratio, reason="Welch PSD has zero total energy."
        )
        collector.add(
            "freq_harmonics_ratio", tooth_ratio, reason="Welch PSD has zero total energy."
        )
        collector.add(
            "freq_sideband_ratio",
            sideband_energy / tooth_energy if tooth_energy > 0 else None,
            reason="Tooth-harmonic energy is zero.",
        )
    else:
        for feature_name in (
            "freq_spindle_harmonic_ratio",
            "freq_tooth_harmonic_ratio",
            "freq_harmonics_ratio",
            "freq_sideband_ratio",
        ):
            collector.add(feature_name, None, reason=metadata_error)

    # ------------------------------------------------------------------- IMF
    collector.add("imf_count", float(physical_count))
    imf_energies = np.sum(np.square(physical_imfs), axis=1)
    total_imf_energy = float(np.sum(imf_energies))
    centre_frequencies: List[float] = []
    bandwidths: List[float] = []
    entropies: List[float] = []
    correlations: List[Optional[float]] = []
    for index, imf in enumerate(physical_imfs, start=1):
        imf_freqs, imf_psd = scipy.signal.welch(imf, fs=sampling_rate, nperseg=min(imf.size, 1024))
        imf_total = float(np.sum(imf_psd))
        if imf_total > 0:
            probability = imf_psd / imf_total
            imf_center = float(np.sum(imf_freqs * probability))
            imf_bandwidth = float(np.sqrt(np.sum(np.square(imf_freqs - imf_center) * probability)))
            imf_entropy = _normalised_entropy(imf_psd)
        else:
            imf_center = 0.0
            imf_bandwidth = 0.0
            imf_entropy = None
        centre_frequencies.append(imf_center)
        bandwidths.append(imf_bandwidth)
        if imf_entropy is not None:
            entropies.append(imf_entropy)
        correlation = _safe_correlation(imf, preprocessed)
        correlations.append(correlation)
        dynamic_values = {
            "energy_ratio": (
                float(imf_energies[index - 1] / total_imf_energy) if total_imf_energy > 0 else None
            ),
            "centre_frequency_hz": imf_center if imf_total > 0 else None,
            "bandwidth_hz": imf_bandwidth if imf_total > 0 else None,
            "spectral_entropy": imf_entropy,
            "kurtosis": _pearson_kurtosis(imf),
            "source_correlation": correlation,
            "gate": float(gates[index - 1]) if gates is not None else None,
        }
        dynamic_metadata = {
            "energy_ratio": (
                "IMF energy divided by total physical-IMF energy",
                "dimensionless",
                True,
                "Stage_1",
            ),
            "centre_frequency_hz": ("PSD-weighted IMF centre frequency", "Hz", False, "Stage_1"),
            "bandwidth_hz": ("PSD-weighted IMF bandwidth", "Hz", False, "Stage_1"),
            "spectral_entropy": (
                "normalised IMF spectral entropy",
                "dimensionless",
                True,
                "Stage_1",
            ),
            "kurtosis": ("Pearson kurtosis of the IMF", "dimensionless", True, "Stage_1"),
            "source_correlation": (
                "absolute correlation with preprocessed source",
                "dimensionless",
                True,
                "Stage_1",
            ),
            "gate": ("independent Stage 2 relevance gate", "dimensionless", True, "Stage_2"),
        }
        for suffix, value in dynamic_values.items():
            feature_name = f"imf_{index}_{suffix}"
            description, unit, dimensionless, stage = dynamic_metadata[suffix]
            collector.add(
                feature_name,
                value,
                reason=(
                    "Stage 2 gates were not supplied."
                    if suffix == "gate" and gates is None
                    else "IMF or source has zero energy/variance."
                ),
                definition=_dynamic_definition(
                    feature_name,
                    "imf",
                    description,
                    unit,
                    stage,
                    dimensionless=dimensionless,
                ),
            )

    collector.add(
        "imf_max_energy_ratio",
        float(np.max(imf_energies) / total_imf_energy) if total_imf_energy > 0 else None,
        reason="Physical IMF energy is zero.",
    )
    collector.add("imf1_correlation", correlations[0], reason="IMF 1 or source variance is zero.")
    collector.add("imf_centre_freq_mean", float(np.mean(centre_frequencies)))
    collector.add("imf_bandwidth_mean", float(np.mean(bandwidths)))
    collector.add(
        "imf_entropy_mean",
        float(np.mean(entropies)) if entropies else None,
        reason="All IMF PSD energies are zero.",
    )

    adjacent_correlations = [
        _safe_correlation(physical_imfs[index], physical_imfs[index + 1])
        for index in range(physical_count - 1)
    ]
    finite_adjacent = [value for value in adjacent_correlations if value is not None]
    collector.add(
        "imf_mode_mixing_index",
        float(np.mean(finite_adjacent)) if finite_adjacent else None,
        reason="Fewer than two non-constant physical IMFs are available.",
    )
    collector.add(
        "imf_max_adjacent_correlation",
        float(np.max(finite_adjacent)) if finite_adjacent else None,
        reason="Fewer than two non-constant physical IMFs are available.",
    )
    total_layer_energy = float(np.sum(np.square(imf_array)))
    if total_layer_energy > 0:
        cross = 0.0
        for left in range(imf_array.shape[0]):
            for right in range(left + 1, imf_array.shape[0]):
                cross += float(np.dot(imf_array[left], imf_array[right]))
        collector.add("imf_orthogonality_index", 2.0 * cross / total_layer_energy)
    else:
        collector.add(
            "imf_orthogonality_index", None, reason="All decomposition layers have zero energy."
        )
    collector.add(
        "imf_frequency_ordering_score",
        float(np.mean(np.diff(centre_frequencies) <= 0)) if physical_count >= 2 else None,
        reason="At least two physical IMFs are required.",
    )
    collector.add(
        "imf_selected_count",
        float(np.count_nonzero(gates >= selected_gate_threshold)) if gates is not None else None,
        reason="Stage 2 gates were not supplied.",
    )

    # ------------------------------------------------------ wavelet / STFT
    try:
        wavelet = pywt.Wavelet(wavelet_name)
        max_wavelet_level = pywt.dwt_max_level(denoised.size, wavelet.dec_len)
        if max_wavelet_level < 1:
            raise ValueError("window is too short for the selected wavelet")
        applied_wavelet_level = min(int(wavelet_level), max_wavelet_level)
        if applied_wavelet_level < 1:
            raise ValueError("wavelet_level must be at least 1")
        coefficients = pywt.wavedec(denoised, wavelet, level=applied_wavelet_level)
        coefficient_names = [f"cA_{applied_wavelet_level}"] + [
            f"cD_{level_index}" for level_index in range(applied_wavelet_level, 0, -1)
        ]
        coefficient_energies = np.array(
            [float(np.sum(np.square(coefficient))) for coefficient in coefficients]
        )
        total_coefficient_energy = float(np.sum(coefficient_energies))
        for coefficient_name, coefficient_energy in zip(coefficient_names, coefficient_energies):
            for suffix, value, unit, dimensionless in (
                ("energy", coefficient_energy, "signal unit^2", False),
                (
                    "energy_ratio",
                    coefficient_energy / total_coefficient_energy
                    if total_coefficient_energy > 0
                    else None,
                    "dimensionless",
                    True,
                ),
            ):
                feature_name = f"wavelet_{coefficient_name}_{suffix}"
                collector.add(
                    feature_name,
                    value,
                    reason="Total wavelet coefficient energy is zero.",
                    definition=_dynamic_definition(
                        feature_name,
                        "wavelet",
                        f"{coefficient_name} coefficient {suffix.replace('_', ' ')}",
                        unit,
                        "Stage_3",
                        dimensionless=dimensionless,
                    ),
                )
        high_frequency_energy = float(
            np.sum(coefficient_energies[-min(2, applied_wavelet_level) :])
        )
        collector.add(
            "wavelet_high_freq_ratio",
            high_frequency_energy / total_coefficient_energy
            if total_coefficient_energy > 0
            else None,
            reason="Total wavelet coefficient energy is zero.",
        )
        collector.add(
            "wavelet_entropy",
            _normalised_entropy(coefficient_energies),
            reason="Total wavelet coefficient energy is zero.",
        )
    except (ValueError, TypeError) as exc:
        collector.add("wavelet_high_freq_ratio", None, reason=str(exc))
        collector.add("wavelet_entropy", None, reason=str(exc))

    stft_nperseg = min(denoised.size, max(32, int(round(0.05 * sampling_rate))))
    stft_noverlap = stft_nperseg // 2
    stft_freqs, stft_times, stft_values = scipy.signal.stft(
        denoised,
        fs=sampling_rate,
        nperseg=stft_nperseg,
        noverlap=stft_noverlap,
        boundary=None,
        padded=False,
    )
    stft_energy = np.square(np.abs(stft_values))
    total_stft_energy = float(np.sum(stft_energy))
    collector.add(
        "wavelet_time_frequency_concentration",
        float(np.max(stft_energy) / total_stft_energy) if total_stft_energy > 0 else None,
        reason="STFT energy is zero.",
    )
    if stft_energy.shape[1] > 0 and total_stft_energy > 0:
        ridge = stft_freqs[np.argmax(stft_energy, axis=0)]
        collector.add("wavelet_dominant_ridge_hz", float(np.median(ridge)))
        traces["dominant_time_frequency_ridge_hz"] = ridge.astype(float)
    else:
        collector.add("wavelet_dominant_ridge_hz", None, reason="STFT has no non-zero frames.")

    # ----------------------------------------------------------- early chatter
    analytic_signal = scipy.signal.hilbert(denoised)
    instantaneous_amplitude = np.abs(analytic_signal)
    instantaneous_energy = np.square(instantaneous_amplitude)
    smoothing_samples = min(denoised.size, max(3, int(round(0.010 * sampling_rate))))
    smoothing_kernel = np.ones(smoothing_samples, dtype=float) / smoothing_samples
    smoothed_energy = np.convolve(instantaneous_energy, smoothing_kernel, mode="same")
    energy_derivative = np.gradient(smoothed_energy, 1.0 / sampling_rate)
    edge = smoothing_samples // 2
    valid_derivative = (
        energy_derivative[edge:-edge]
        if edge > 0 and 2 * edge < denoised.size
        else energy_derivative
    )
    time_axis = np.arange(denoised.size, dtype=float) / sampling_rate
    collector.add("early_instantaneous_amplitude_mean", float(np.mean(instantaneous_amplitude)))
    collector.add("early_instantaneous_amplitude_max", float(np.max(instantaneous_amplitude)))
    collector.add("early_instantaneous_energy_mean", float(np.mean(instantaneous_energy)))
    collector.add("early_instantaneous_energy_max", float(np.max(instantaneous_energy)))
    collector.add(
        "early_energy_growth_rate",
        _linear_slope(time_axis, smoothed_energy),
        reason="Window duration is insufficient for a slope.",
    )
    collector.add("early_hegr", float(np.mean(np.maximum(valid_derivative, 0.0))))
    traces["instantaneous_amplitude"] = instantaneous_amplitude
    traces["instantaneous_energy"] = instantaneous_energy
    traces["smoothed_instantaneous_energy"] = smoothed_energy
    traces["hegr_derivative"] = energy_derivative

    total_energy_by_frame = np.sum(stft_energy, axis=0)
    chatter_stft_mask = (stft_freqs >= chatter_band[0]) & (stft_freqs <= chatter_band[1])
    chatter_energy_by_frame = np.sum(stft_energy[chatter_stft_mask], axis=0)
    collector.add(
        "early_chatter_band_energy_growth",
        _linear_slope(stft_times, chatter_energy_by_frame),
        reason="At least two STFT frames are required.",
    )
    collector.add(
        "early_short_term_spectral_energy_growth",
        _linear_slope(stft_times, total_energy_by_frame),
        reason="At least two STFT frames are required.",
    )
    traces["short_term_time_seconds"] = stft_times
    traces["short_term_chatter_band_energy"] = chatter_energy_by_frame
    traces["short_term_spectral_energy"] = total_energy_by_frame

    # ------------------------------------------------------------- physics
    chatter_energy = _band_power(freqs, psd, chatter_mask)
    collector.add("physics_chatter_band_energy", chatter_energy)
    peak_frequency = float(freqs[np.argmax(psd)]) if total_psd > 0 else None
    if peak_frequency is None:
        collector.add(
            "physics_frequency_proximity", None, reason="Welch PSD has zero total energy."
        )
    else:
        distance_to_band = (
            chatter_band[0] - peak_frequency
            if peak_frequency < chatter_band[0]
            else peak_frequency - chatter_band[1]
            if peak_frequency > chatter_band[1]
            else 0.0
        )
        collector.add(
            "physics_frequency_proximity", float(np.exp(-0.5 * (distance_to_band / spread) ** 2))
        )

    physics_feature_names = (
        "physics_spindle_frequency_hz",
        "physics_tooth_passing_frequency_hz",
        "physics_peak_to_spindle_harmonic_distance_hz",
        "physics_peak_to_tooth_harmonic_distance_hz",
        "physics_sideband_strength",
        "physics_forced_vibration_energy",
        "physics_chatter_to_harmonic_energy_ratio",
    )
    if metadata_error is not None or rpm_value is None or tooth_value is None:
        for feature_name in physics_feature_names:
            collector.add(feature_name, None, reason=metadata_error)
    else:
        spindle_frequency = rpm_value / 60.0
        tooth_frequency = spindle_frequency * tooth_value
        spindle_harmonics = np.array(
            [spindle_frequency * order for order in range(1, harmonic_count + 1)]
        )
        tooth_harmonics = np.array(
            [tooth_frequency * order for order in range(1, harmonic_count + 1)]
        )
        collector.add("physics_spindle_frequency_hz", spindle_frequency)
        collector.add("physics_tooth_passing_frequency_hz", tooth_frequency)
        collector.add(
            "physics_peak_to_spindle_harmonic_distance_hz",
            float(np.min(np.abs(spindle_harmonics - peak_frequency)))
            if peak_frequency is not None
            else None,
            reason="Welch PSD has zero total energy.",
        )
        collector.add(
            "physics_peak_to_tooth_harmonic_distance_hz",
            float(np.min(np.abs(tooth_harmonics - peak_frequency)))
            if peak_frequency is not None
            else None,
            reason="Welch PSD has zero total energy.",
        )
        sideband_energy = _band_power(freqs, psd, sideband_mask)
        forced_mask = spindle_mask | tooth_mask
        forced_energy = _band_power(freqs, psd, forced_mask)
        collector.add("physics_sideband_strength", sideband_energy)
        collector.add("physics_forced_vibration_energy", forced_energy)
        collector.add(
            "physics_chatter_to_harmonic_energy_ratio",
            chatter_energy / forced_energy if forced_energy > 0 else None,
            reason="Forced-harmonic energy is zero.",
        )

    defined_count = sum(value is not None for value in collector.values.values())
    quality = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "total_feature_count": len(collector.values),
        "defined_feature_count": defined_count,
        "undefined_feature_count": len(collector.undefined),
        "defined_fraction": defined_count / max(1, len(collector.values)),
        "all_defined_values_finite": all(
            value is None or math.isfinite(value) for value in collector.values.values()
        ),
        "physics_metadata_valid": metadata_error is None,
        "physics_metadata_error": metadata_error,
        "physical_imf_count": physical_count,
        "residual_handling": "excluded_last_row" if residual_last_row else "no_residual_row",
    }
    return FeatureExtractionResult(
        values=collector.values,
        undefined_reasons=collector.undefined,
        definitions=tuple(collector.definitions.values()),
        traces=traces,
        quality=quality,
    )


def extract_window_features(
    raw_window: np.ndarray,
    prep_physical_window: np.ndarray,
    denoised_physical_window: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    rpm: float,
    tooth_count: int,
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0,
) -> Dict[str, float]:
    """Compatibility wrapper returning finite scalar features.

    Undefined values are filled with ``0.0`` only in this legacy view.  New
    artifact writers must use :func:`extract_window_feature_result` so the null
    value and its reason are preserved.
    """

    return extract_window_feature_result(
        raw_window,
        prep_physical_window,
        denoised_physical_window,
        imfs,
        fs,
        rpm,
        tooth_count,
        chatter_center,
        chatter_spread,
        strict_chatter_band=False,
    ).finite_values(undefined_fill=0.0)


def extract_sliding_window_features(
    raw_signal: np.ndarray,
    preprocessed_physical_signal: np.ndarray,
    denoised_physical_signal: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    rpm: Optional[float],
    tooth_count: Optional[int],
    *,
    window_seconds: float = 1.0,
    overlap_ratio: float = 0.75,
    imf_gates: Optional[np.ndarray] = None,
    **feature_kwargs: Any,
) -> Tuple[WindowFeatureRecord, ...]:
    """Extract deterministic per-window Stage 4 results from aligned full arrays."""

    sampling_rate = _finite_positive(fs, "sampling rate")
    duration = _finite_positive(window_seconds, "window_seconds")
    if not 0.0 <= float(overlap_ratio) < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1).")
    raw = _as_finite_signal(raw_signal, "raw_signal")
    preprocessed = _as_finite_signal(preprocessed_physical_signal, "preprocessed_physical_signal")
    denoised = _as_finite_signal(denoised_physical_signal, "denoised_physical_signal")
    imf_array = np.asarray(imfs, dtype=float)
    if raw.size != preprocessed.size or raw.size != denoised.size:
        raise ValueError("All full-stage signals must have identical lengths.")
    if imf_array.ndim != 2 or imf_array.shape[1] != raw.size:
        raise ValueError("Full IMF array must align with the full signals.")
    window_samples = int(round(duration * sampling_rate))
    if window_samples < 2 or window_samples > raw.size:
        raise ValueError("window_seconds produces an invalid window length.")
    step_samples = max(1, int(round(window_samples * (1.0 - float(overlap_ratio)))))
    records = []
    for window_index, start in enumerate(range(0, raw.size - window_samples + 1, step_samples)):
        end = start + window_samples
        result = extract_window_feature_result(
            raw[start:end],
            preprocessed[start:end],
            denoised[start:end],
            imf_array[:, start:end],
            sampling_rate,
            rpm,
            tooth_count,
            imf_gates=imf_gates,
            **feature_kwargs,
        )
        records.append(
            WindowFeatureRecord(
                window_index=window_index,
                start_index=start,
                end_index=end,
                start_time_seconds=start / sampling_rate,
                end_time_seconds=(end - 1) / sampling_rate,
                result=result,
            )
        )
    return tuple(records)


def summarize_feature_repeatability(
    reference: Sequence[WindowFeatureRecord],
    repeated: Sequence[WindowFeatureRecord],
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compare two Stage 4 extractions made from identical canonical inputs.

    This is a repeat-extraction check, not a coefficient of variation across
    different windows. Undefined values must remain undefined for the same
    reason, window boundaries and schemas must match, and all defined values
    must agree within the stated numerical tolerance.
    """

    if not reference or not repeated:
        raise ValueError("Both repeatability runs must contain at least one window.")
    if len(reference) != len(repeated):
        raise ValueError("Repeated feature runs must contain the same number of windows.")
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("Repeatability tolerances must be finite and non-negative.")

    feature_names = sorted(
        {name for record in (*tuple(reference), *tuple(repeated)) for name in record.result.values}
    )
    feature_maximum_differences = {name: 0.0 for name in feature_names}
    feature_comparison_counts = {name: 0 for name in feature_names}
    undefined_pattern_match = True
    undefined_reason_match = True
    schema_match = True
    window_alignment_match = True
    exact_value_match = True
    all_values_within_tolerance = True

    for first, second in zip(reference, repeated):
        window_alignment_match = window_alignment_match and (
            first.window_index,
            first.start_index,
            first.end_index,
        ) == (
            second.window_index,
            second.start_index,
            second.end_index,
        )
        schema_match = schema_match and (
            first.result.schema_version == second.result.schema_version
            and first.result.definitions == second.result.definitions
            and set(first.result.values) == set(second.result.values)
        )
        for name in feature_names:
            left = first.result.values.get(name)
            right = second.result.values.get(name)
            if left is None or right is None:
                undefined_pattern_match = undefined_pattern_match and left is None and right is None
                if left is None and right is None:
                    undefined_reason_match = undefined_reason_match and (
                        first.result.undefined_reasons.get(name)
                        == second.result.undefined_reasons.get(name)
                    )
                continue
            left_value = float(left)
            right_value = float(right)
            difference = abs(left_value - right_value)
            feature_comparison_counts[name] += 1
            feature_maximum_differences[name] = max(feature_maximum_differences[name], difference)
            exact_value_match = exact_value_match and left_value == right_value
            all_values_within_tolerance = all_values_within_tolerance and bool(
                np.isclose(left_value, right_value, atol=atol, rtol=rtol)
            )
    deterministic = bool(
        window_alignment_match
        and schema_match
        and undefined_pattern_match
        and undefined_reason_match
        and all_values_within_tolerance
    )
    return {
        "method": "two Stage 4 extractions from identical canonical Stage 1-3 arrays",
        "repeat_count": 2,
        "window_count": len(reference),
        "feature_count": len(feature_names),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "window_alignment_match": bool(window_alignment_match),
        "schema_match": bool(schema_match),
        "undefined_pattern_match": bool(undefined_pattern_match),
        "undefined_reason_match": bool(undefined_reason_match),
        "exact_value_match": bool(exact_value_match),
        "all_values_within_tolerance": bool(all_values_within_tolerance),
        "deterministic": deterministic,
        "maximum_absolute_difference": float(
            max(feature_maximum_differences.values(), default=0.0)
        ),
        "feature_maximum_absolute_difference": feature_maximum_differences,
        "feature_comparison_count": feature_comparison_counts,
    }


def aggregate_feature_results(
    results: Sequence[FeatureExtractionResult | WindowFeatureRecord],
    *,
    recording_ids: Optional[Sequence[str]] = None,
) -> FeatureAggregateResult:
    """Build aggregate rows, summaries, missingness, correlations, and schema."""

    if not results:
        raise ValueError("At least one feature result is required for aggregation.")
    extracted = [item.result if isinstance(item, WindowFeatureRecord) else item for item in results]
    if recording_ids is not None and len(recording_ids) != len(extracted):
        raise ValueError("recording_ids length must match the number of feature results.")
    ids = (
        list(recording_ids)
        if recording_ids is not None
        else [str(index) for index in range(len(extracted))]
    )
    feature_names = sorted({name for result in extracted for name in result.values})
    rows: List[Dict[str, Any]] = []
    for recording_id, result in zip(ids, extracted):
        row: Dict[str, Any] = {
            "recording_id": recording_id,
            "feature_schema_version": result.schema_version,
        }
        row.update({name: result.values.get(name) for name in feature_names})
        rows.append(row)

    summary: Dict[str, Dict[str, Optional[float]]] = {}
    missingness: Dict[str, Dict[str, Any]] = {}
    for feature_name in feature_names:
        defined: List[float] = []
        for result in extracted:
            feature_value = result.values.get(feature_name)
            if feature_value is not None:
                defined.append(float(feature_value))
        summary[feature_name] = {
            "count": float(len(defined)),
            "mean": float(np.mean(defined)) if defined else None,
            "std": float(np.std(defined)) if defined else None,
            "min": float(np.min(defined)) if defined else None,
            "max": float(np.max(defined)) if defined else None,
        }
        reasons: Dict[str, int] = {}
        for result in extracted:
            if result.values.get(feature_name) is None:
                reason = result.undefined_reasons.get(
                    feature_name, "Feature absent from this schema."
                )
                reasons[reason] = reasons.get(reason, 0) + 1
        missingness[feature_name] = {
            "missing_count": len(extracted) - len(defined),
            "missing_fraction": (len(extracted) - len(defined)) / len(extracted),
            "reasons": reasons,
        }

    correlations: Dict[str, Dict[str, Optional[float]]] = {name: {} for name in feature_names}
    for left_name in feature_names:
        for right_name in feature_names:
            pairs = [
                (result.values.get(left_name), result.values.get(right_name))
                for result in extracted
            ]
            pairs = [
                (float(left), float(right))
                for left, right in pairs
                if left is not None and right is not None
            ]
            if len(pairs) < 2:
                correlations[left_name][right_name] = None
                continue
            left_values = np.array([pair[0] for pair in pairs])
            right_values = np.array([pair[1] for pair in pairs])
            if (
                np.std(left_values) <= np.finfo(float).eps
                or np.std(right_values) <= np.finfo(float).eps
            ):
                correlations[left_name][right_name] = None
            else:
                correlations[left_name][right_name] = float(
                    np.corrcoef(left_values, right_values)[0, 1]
                )

    definitions: Dict[str, FeatureDefinition] = {}
    for result in extracted:
        for definition in result.definitions:
            definitions.setdefault(definition.name, definition)
    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": [definitions[name].to_dict() for name in sorted(definitions)],
    }
    return FeatureAggregateResult(
        rows=tuple(rows),
        summary=summary,
        missingness=missingness,
        correlations=correlations,
        schema=schema,
    )
