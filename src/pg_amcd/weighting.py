"""Stage 2: multi-criteria and physics-guided IMF weighting.

The compatibility functions at the bottom of this module retain the original
tuple-returning API.  New code should prefer :func:`analyze_physics_guided_weighting`,
which validates all scientific inputs and returns the complete per-IMF indicator
table, independent gates, reconstruction, and quantitative diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.signal
import scipy.special
import scipy.stats


ResidualPolicy = Literal["last_row", "none"]


# These defaults exist only for the historical tuple-returning wrapper.  The
# canonical diagnostic API is strict by default and requires every gate
# coefficient to be present in the resolved configuration.
_LEGACY_PHYSICS_DEFAULTS: Dict[str, float | int] = {
    "chatter_energy_weight": 4.0,
    "correlation_weight": 2.0,
    "kurtosis_weight": 1.0,
    "frequency_proximity_weight": 1.0,
    "harmonic_penalty": 5.0,
    "offset": 1.5,
    "harmonic_tolerance_hz": 15.0,
    "harmonic_count": 5,
    "kurtosis_scale": 10.0,
}


@dataclass(frozen=True)
class PhysicsMetadata:
    """Validated machining metadata used by physics-guided Stage 2/4 logic."""

    rpm: float
    tooth_count: int
    stickout: Optional[float] = None
    depth_of_cut: Optional[float] = None
    feed_rate: Optional[float] = None
    tool_identifier: Optional[str] = None

    @property
    def spindle_frequency_hz(self) -> float:
        return self.rpm / 60.0

    @property
    def tooth_passing_frequency_hz(self) -> float:
        return self.spindle_frequency_hz * self.tooth_count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IMFIndicator:
    """All required Stage 2 indicators for one physical IMF."""

    imf_index: int
    correlation: float
    relative_energy: float
    kurtosis: float
    kurtosis_score: float
    chatter_band_energy_ratio: float
    spindle_harmonic_energy_ratio: float
    tooth_harmonic_energy_ratio: float
    forced_harmonic_energy_ratio: float
    frequency_proximity: float
    centre_frequency_hz: float
    bandwidth_hz: float
    spectral_entropy: float
    gate: float

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class WeightingMetrics:
    """Quantitative comparison of the source and gate-weighted reconstruction."""

    rms_before: float
    rms_after: float
    energy_before: float
    energy_after: float
    correlation_with_source: float
    chatter_band_retention: float
    spindle_harmonic_attenuation: float
    tooth_harmonic_attenuation: float
    out_of_band_attenuation: float
    spectral_distortion: float
    reconstruction_runtime_seconds: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class IMFWeightingResult:
    """Typed result of physics-guided independent IMF gating."""

    metadata: PhysicsMetadata
    indicators: Tuple[IMFIndicator, ...]
    gates: np.ndarray
    reconstructed_scaled: np.ndarray
    metrics: WeightingMetrics
    residual_policy: ResidualPolicy
    residual_excluded: bool
    coefficients: Dict[str, float | int]
    chatter_band_hz: Tuple[float, float]

    @property
    def physical_imf_count(self) -> int:
        return len(self.indicators)

    def to_dict(self, *, include_signal: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "indicators": [indicator.to_dict() for indicator in self.indicators],
            "gates": self.gates.tolist(),
            "metrics": self.metrics.to_dict(),
            "residual_policy": self.residual_policy,
            "residual_excluded": self.residual_excluded,
            "physical_imf_count": self.physical_imf_count,
            "coefficients": dict(self.coefficients),
            "chatter_band_hz": list(self.chatter_band_hz),
        }
        if include_signal:
            payload["reconstructed_scaled"] = self.reconstructed_scaled.tolist()
        return payload


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} is required and must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive, got {number}.")
    return number


def _optional_finite_float(value: Any, name: str) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _finite_float(value, name)


def validate_physics_metadata(
    metadata: PhysicsMetadata | Mapping[str, Any],
) -> PhysicsMetadata:
    """Validate required RPM/tooth count and preserve all optional physics fields.

    Missing or malformed RPM/tooth-count values are errors.  This function never
    substitutes arbitrary machining parameters.
    """

    if isinstance(metadata, PhysicsMetadata):
        raw: Mapping[str, Any] = metadata.to_dict()
    elif isinstance(metadata, Mapping):
        raw = metadata
    else:
        raise ValueError("Physics metadata must be a mapping or PhysicsMetadata instance.")

    rpm = _finite_float(raw.get("rpm"), "rpm", positive=True)
    tooth_raw = raw.get("tooth_count")
    if tooth_raw is None or isinstance(tooth_raw, bool):
        raise ValueError("tooth_count is required and must be a positive integer.")
    try:
        tooth_as_float = float(tooth_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tooth_count must be a positive integer, got {tooth_raw!r}.") from exc
    if not math.isfinite(tooth_as_float) or not tooth_as_float.is_integer():
        raise ValueError(f"tooth_count must be a positive integer, got {tooth_raw!r}.")
    tooth_count = int(tooth_as_float)
    if tooth_count < 1:
        raise ValueError(f"tooth_count must be at least 1, got {tooth_count}.")

    tool_identifier = raw.get("tool_identifier", raw.get("tool_id"))
    if tool_identifier is not None:
        tool_identifier = str(tool_identifier).strip() or None

    return PhysicsMetadata(
        rpm=rpm,
        tooth_count=tooth_count,
        stickout=_optional_finite_float(raw.get("stickout"), "stickout"),
        depth_of_cut=_optional_finite_float(raw.get("depth_of_cut"), "depth_of_cut"),
        feed_rate=_optional_finite_float(raw.get("feed_rate"), "feed_rate"),
        tool_identifier=tool_identifier,
    )


def _validate_imf_inputs(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
    residual_policy: ResidualPolicy,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    imf_array = np.asarray(imfs, dtype=float)
    source = np.asarray(original_signal, dtype=float)
    sampling_rate = _finite_float(fs, "sampling rate", positive=True)
    if imf_array.ndim != 2:
        raise ValueError("IMFs must be a two-dimensional array.")
    if source.ndim != 1:
        raise ValueError("Source signal must be a one-dimensional array.")
    if imf_array.shape[1] != source.size:
        raise ValueError("IMF and source-signal lengths differ.")
    if source.size < 2:
        raise ValueError("Source signal must contain at least two samples.")
    if not np.all(np.isfinite(imf_array)):
        raise ValueError("IMFs contain non-finite values.")
    if not np.all(np.isfinite(source)):
        raise ValueError("Source signal contains non-finite values.")
    if residual_policy not in ("last_row", "none"):
        raise ValueError("residual_policy must be either 'last_row' or 'none'.")
    physical_count = imf_array.shape[0] - (1 if residual_policy == "last_row" else 0)
    if physical_count < 1:
        raise ValueError("At least one physical IMF is required after residual handling.")
    return imf_array, source, sampling_rate, physical_count


def _resolved_physics_config(
    config: Mapping[str, Any], *, strict: bool
) -> Tuple[Dict[str, float | int], float, float]:
    pg_raw = config.get("physics_gating")
    if pg_raw is None:
        if strict:
            raise ValueError("Resolved config is missing the 'physics_gating' section.")
        pg_raw = {}
    if not isinstance(pg_raw, Mapping):
        raise ValueError("physics_gating configuration must be a mapping.")

    missing = [key for key in _LEGACY_PHYSICS_DEFAULTS if key not in pg_raw]
    if strict and missing:
        raise ValueError(
            "Resolved physics_gating config is missing required keys: " + ", ".join(missing)
        )
    resolved: Dict[str, float | int] = dict(_LEGACY_PHYSICS_DEFAULTS)
    resolved.update(pg_raw)

    for key in (
        "chatter_energy_weight",
        "correlation_weight",
        "kurtosis_weight",
        "frequency_proximity_weight",
        "harmonic_penalty",
        "offset",
    ):
        resolved[key] = _finite_float(resolved[key], f"physics_gating.{key}")
    resolved["harmonic_tolerance_hz"] = _finite_float(
        resolved["harmonic_tolerance_hz"],
        "physics_gating.harmonic_tolerance_hz",
        positive=True,
    )
    resolved["kurtosis_scale"] = _finite_float(
        resolved["kurtosis_scale"], "physics_gating.kurtosis_scale", positive=True
    )
    harmonic_count_float = _finite_float(
        resolved["harmonic_count"], "physics_gating.harmonic_count", positive=True
    )
    if not harmonic_count_float.is_integer():
        raise ValueError("physics_gating.harmonic_count must be a positive integer.")
    resolved["harmonic_count"] = int(harmonic_count_float)

    maiw_cfg = config.get("maiw")
    if not isinstance(maiw_cfg, Mapping):
        raise ValueError("Resolved config is missing the 'maiw' section.")
    if strict and (
        "chatter_band_center" not in maiw_cfg or "chatter_band_spread" not in maiw_cfg
    ):
        raise ValueError(
            "Resolved maiw config must contain chatter_band_center and chatter_band_spread."
        )
    centre = _finite_float(
        maiw_cfg.get("chatter_band_center", 1250.0), "maiw.chatter_band_center", positive=True
    )
    spread = _finite_float(
        maiw_cfg.get("chatter_band_spread", 500.0), "maiw.chatter_band_spread", positive=True
    )
    return resolved, centre, spread


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(abs(np.dot(a_centered, b_centered) / denominator))


def _normalised_spectral_entropy(psd: np.ndarray) -> float:
    total = float(np.sum(psd))
    if total <= 0 or psd.size <= 1:
        return 0.0
    probabilities = psd / total
    nonzero = probabilities > 0
    entropy = -float(np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    return entropy / math.log2(probabilities.size)


def _spectral_summary(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(signal.size, 1024))
    total = float(np.sum(psd))
    if total <= 0:
        return freqs, psd, 0.0, 0.0, 0.0
    probabilities = psd / total
    centre = float(np.sum(freqs * probabilities))
    bandwidth = float(np.sqrt(np.sum(np.square(freqs - centre) * probabilities)))
    return freqs, psd, centre, bandwidth, _normalised_spectral_entropy(psd)


def _harmonic_mask(
    freqs: np.ndarray, fundamental: float, count: int, tolerance_hz: float
) -> np.ndarray:
    mask = np.zeros(freqs.shape, dtype=bool)
    nyquist = float(freqs[-1]) if freqs.size else 0.0
    for order in range(1, count + 1):
        harmonic = fundamental * order
        if harmonic > nyquist + tolerance_hz:
            break
        mask |= np.abs(freqs - harmonic) <= tolerance_hz
    return mask


def _ratio_in_mask(psd: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.sum(psd))
    return float(np.sum(psd[mask]) / total) if total > 0 else 0.0


def _spectral_comparison_metrics(
    source: np.ndarray,
    reconstructed: np.ndarray,
    fs: float,
    chatter_band: Tuple[float, float],
    spindle_frequency: float,
    tooth_frequency: float,
    harmonic_count: int,
    tolerance_hz: float,
    runtime_seconds: float,
) -> WeightingMetrics:
    freqs, source_psd = scipy.signal.welch(
        source, fs=fs, nperseg=min(source.size, 1024)
    )
    _, reconstructed_psd = scipy.signal.welch(
        reconstructed, fs=fs, nperseg=min(reconstructed.size, 1024)
    )
    chatter_mask = (freqs >= chatter_band[0]) & (freqs <= chatter_band[1])
    spindle_mask = _harmonic_mask(freqs, spindle_frequency, harmonic_count, tolerance_hz)
    tooth_mask = _harmonic_mask(freqs, tooth_frequency, harmonic_count, tolerance_hz)
    out_of_band_mask = ~chatter_mask

    def retention(mask: np.ndarray) -> float:
        before = float(np.sum(source_psd[mask]))
        after = float(np.sum(reconstructed_psd[mask]))
        return after / before if before > 0 else 0.0

    source_total = float(np.sum(source_psd))
    reconstructed_total = float(np.sum(reconstructed_psd))
    if source_total > 0 and reconstructed_total > 0:
        source_norm = source_psd / source_total
        reconstructed_norm = reconstructed_psd / reconstructed_total
        distortion = float(np.sum(np.abs(source_norm - reconstructed_norm)))
    else:
        distortion = 0.0

    return WeightingMetrics(
        rms_before=float(np.sqrt(np.mean(np.square(source)))),
        rms_after=float(np.sqrt(np.mean(np.square(reconstructed)))),
        energy_before=float(np.sum(np.square(source))),
        energy_after=float(np.sum(np.square(reconstructed))),
        correlation_with_source=_safe_correlation(source, reconstructed),
        chatter_band_retention=retention(chatter_mask),
        spindle_harmonic_attenuation=1.0 - retention(spindle_mask),
        tooth_harmonic_attenuation=1.0 - retention(tooth_mask),
        out_of_band_attenuation=1.0 - retention(out_of_band_mask),
        spectral_distortion=distortion,
        reconstruction_runtime_seconds=runtime_seconds,
    )


def analyze_physics_guided_weighting(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
    metadata: PhysicsMetadata | Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    residual_policy: ResidualPolicy = "last_row",
    strict_config: bool = True,
) -> IMFWeightingResult:
    """Calculate all Stage 2 indicators and independent physics-guided gates.

    The residual policy is recorded in the returned result.  Gates are sigmoid
    relevance scores and are deliberately **not** normalised to sum to one.
    """

    started = time.perf_counter()
    imf_array, source, sampling_rate, physical_count = _validate_imf_inputs(
        imfs, original_signal, fs, residual_policy
    )
    physics = validate_physics_metadata(metadata)
    coefficients, chatter_center, chatter_spread = _resolved_physics_config(
        config, strict=strict_config
    )
    chatter_low = max(0.0, chatter_center - chatter_spread)
    chatter_high = min(sampling_rate / 2.0, chatter_center + chatter_spread)
    if chatter_low >= chatter_high:
        raise ValueError(
            "Configured chatter band does not overlap the measurable range [0, Nyquist]."
        )

    total_physical_energy = float(np.sum(np.square(imf_array[:physical_count])))
    indicators = []
    gates = np.empty(physical_count, dtype=float)
    harmonic_count = int(coefficients["harmonic_count"])
    tolerance = float(coefficients["harmonic_tolerance_hz"])

    for index in range(physical_count):
        imf = imf_array[index]
        freqs, psd, centre_frequency, bandwidth, entropy = _spectral_summary(
            imf, sampling_rate
        )
        chatter_mask = (freqs >= chatter_low) & (freqs <= chatter_high)
        spindle_mask = _harmonic_mask(
            freqs, physics.spindle_frequency_hz, harmonic_count, tolerance
        )
        tooth_mask = _harmonic_mask(
            freqs, physics.tooth_passing_frequency_hz, harmonic_count, tolerance
        )
        forced_mask = spindle_mask | tooth_mask

        raw_kurtosis = float(scipy.stats.kurtosis(imf, fisher=False, bias=False))
        if not math.isfinite(raw_kurtosis):
            raw_kurtosis = 0.0
        kurtosis_score = float(
            np.clip(
                (raw_kurtosis - 3.0) / float(coefficients["kurtosis_scale"]),
                0.0,
                1.0,
            )
        )
        correlation = _safe_correlation(imf, source)
        chatter_ratio = _ratio_in_mask(psd, chatter_mask)
        spindle_ratio = _ratio_in_mask(psd, spindle_mask)
        tooth_ratio = _ratio_in_mask(psd, tooth_mask)
        forced_ratio = _ratio_in_mask(psd, forced_mask)
        relative_energy = (
            float(np.sum(np.square(imf)) / total_physical_energy)
            if total_physical_energy > 0
            else 0.0
        )
        frequency_proximity = float(
            np.exp(-0.5 * np.square((centre_frequency - chatter_center) / chatter_spread))
        )
        score = (
            float(coefficients["chatter_energy_weight"]) * chatter_ratio
            + float(coefficients["correlation_weight"]) * correlation
            + float(coefficients["kurtosis_weight"]) * kurtosis_score
            + float(coefficients["frequency_proximity_weight"]) * frequency_proximity
            - float(coefficients["harmonic_penalty"]) * forced_ratio
            - float(coefficients["offset"])
        )
        gate = float(scipy.special.expit(score))
        gates[index] = gate
        indicators.append(
            IMFIndicator(
                imf_index=index,
                correlation=correlation,
                relative_energy=relative_energy,
                kurtosis=raw_kurtosis,
                kurtosis_score=kurtosis_score,
                chatter_band_energy_ratio=chatter_ratio,
                spindle_harmonic_energy_ratio=spindle_ratio,
                tooth_harmonic_energy_ratio=tooth_ratio,
                forced_harmonic_energy_ratio=forced_ratio,
                frequency_proximity=frequency_proximity,
                centre_frequency_hz=centre_frequency,
                bandwidth_hz=bandwidth,
                spectral_entropy=entropy,
                gate=gate,
            )
        )

    reconstructed = np.sum(imf_array[:physical_count] * gates[:, np.newaxis], axis=0)
    runtime = time.perf_counter() - started
    metrics = _spectral_comparison_metrics(
        source,
        reconstructed,
        sampling_rate,
        (chatter_low, chatter_high),
        physics.spindle_frequency_hz,
        physics.tooth_passing_frequency_hz,
        harmonic_count,
        tolerance,
        runtime,
    )
    return IMFWeightingResult(
        metadata=physics,
        indicators=tuple(indicators),
        gates=gates,
        reconstructed_scaled=reconstructed,
        metrics=metrics,
        residual_policy=residual_policy,
        residual_excluded=residual_policy == "last_row",
        coefficients=coefficients,
        chatter_band_hz=(chatter_low, chatter_high),
    )


def summarize_gate_stability(
    gate_vectors: Sequence[np.ndarray], *, selection_threshold: float = 0.5
) -> Dict[str, Any]:
    """Summarise gate-vector and selected-IMF stability across random seeds."""

    if not gate_vectors:
        raise ValueError("At least one gate vector is required.")
    threshold = _finite_float(selection_threshold, "selection_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("selection_threshold must be in [0, 1].")
    matrix = np.vstack([np.asarray(vector, dtype=float) for vector in gate_vectors])
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Gate vectors must be non-empty one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gate vectors contain non-finite values.")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Independent gates must be bounded in [0, 1].")
    selected = matrix >= threshold
    consistency = np.mean(np.all(selected == selected[0], axis=0))
    return {
        "n_seeds": int(matrix.shape[0]),
        "physical_imf_count": int(matrix.shape[1]),
        "mean_gate_by_imf": np.mean(matrix, axis=0).tolist(),
        "std_gate_by_imf": np.std(matrix, axis=0).tolist(),
        "mean_gate_std": float(np.mean(np.std(matrix, axis=0))),
        "max_gate_std": float(np.max(np.std(matrix, axis=0))),
        "selected_imf_consistency": float(consistency),
        "selected_count_by_seed": np.sum(selected, axis=1).astype(int).tolist(),
        "selection_threshold": threshold,
    }


def restore_physical_units(scaled_signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """Restore a scaled Stage 2/3 signal using the Stage 1 amplitude scale."""

    signal = np.asarray(scaled_signal, dtype=float)
    factor = _finite_float(scale_factor, "scale_factor", positive=True)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("Scaled reconstruction must be a finite one-dimensional signal.")
    return signal * factor


def normalize_indicator(values: np.ndarray) -> np.ndarray:
    """Safely normalise a legacy MAIW indicator so it sums to one."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Indicator values must be a non-empty one-dimensional array.")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(array))
    return array / total if total > 0 else np.ones_like(array) / array.size


def calculate_maiw_weights(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the original sum-normalised MAIW baseline weights."""

    imf_array, source, sampling_rate, num_weighted = _validate_imf_inputs(
        imfs, original_signal, fs, "last_row"
    )
    maiw_cfg = config["maiw"]
    alpha = _finite_float(maiw_cfg.get("alpha", 0.25), "maiw.alpha")
    beta = _finite_float(maiw_cfg.get("beta", 0.25), "maiw.beta")
    gamma = _finite_float(maiw_cfg.get("gamma", 0.25), "maiw.gamma")
    delta = _finite_float(maiw_cfg.get("delta", 0.25), "maiw.delta")
    if min(alpha, beta, gamma, delta) < 0:
        raise ValueError("MAIW coefficients must be non-negative.")
    if alpha + beta + gamma + delta <= 0:
        raise ValueError("MAIW coefficients (alpha+beta+gamma+delta) must be positive.")
    center = _finite_float(
        maiw_cfg.get("chatter_band_center", 1250.0), "maiw.chatter_band_center", positive=True
    )
    spread = _finite_float(
        maiw_cfg.get("chatter_band_spread", 500.0), "maiw.chatter_band_spread", positive=True
    )

    correlation = np.zeros(num_weighted)
    energy = np.sum(np.square(imf_array[:num_weighted]), axis=1)
    energy = energy / np.sum(energy) if np.sum(energy) > 0 else np.zeros_like(energy)
    kurtosis = np.array(
        [float(scipy.stats.kurtosis(row, fisher=False, bias=False)) for row in imf_array[:num_weighted]]
    )
    kurtosis = np.nan_to_num(kurtosis, nan=0.0, posinf=0.0, neginf=0.0)
    proximity = np.zeros(num_weighted)

    for index, imf in enumerate(imf_array[:num_weighted]):
        correlation[index] = _safe_correlation(imf, source)
        freqs, psd = scipy.signal.welch(
            imf, sampling_rate, nperseg=min(imf.size, 1024)
        )
        dominant_frequency = float(freqs[np.argmax(psd)]) if psd.size else 0.0
        proximity[index] = np.exp(
            -np.square(dominant_frequency - center) / (2.0 * np.square(spread))
        )

    weights = (
        alpha * normalize_indicator(correlation)
        + beta * normalize_indicator(energy)
        + gamma * normalize_indicator(kurtosis)
        + delta * normalize_indicator(proximity)
    )
    weight_sum = float(np.sum(weights))
    weights = weights / weight_sum if weight_sum > 0 else np.ones(num_weighted) / num_weighted
    return weights, correlation, energy, kurtosis, proximity


def calculate_physics_gated_weights(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
    rpm: float,
    tooth_count: int,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning the historical five-array tuple.

    The fifth array is the union of spindle- and tooth-harmonic energy ratios.
    New code should use :func:`analyze_physics_guided_weighting` to retain every
    indicator and all resolved metadata.
    """

    result = analyze_physics_guided_weighting(
        imfs,
        original_signal,
        fs,
        {"rpm": rpm, "tooth_count": tooth_count},
        config,
        residual_policy="last_row",
        strict_config=False,
    )
    correlation = np.array([row.correlation for row in result.indicators])
    chatter = np.array([row.chatter_band_energy_ratio for row in result.indicators])
    kurtosis = np.array([row.kurtosis_score for row in result.indicators])
    harmonics = np.array([row.forced_harmonic_energy_ratio for row in result.indicators])
    return result.gates.copy(), correlation, chatter, kurtosis, harmonics


def reconstruct_weighted_signal(imfs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reconstruct from physical IMFs, explicitly excluding the final residual."""

    imf_array = np.asarray(imfs, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if imf_array.ndim != 2:
        raise ValueError("IMFs must be a two-dimensional array.")
    if weight_array.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array.")
    if weight_array.size != imf_array.shape[0] - 1:
        raise ValueError("Weight count does not match physical IMF count.")
    if not np.all(np.isfinite(imf_array)) or not np.all(np.isfinite(weight_array)):
        raise ValueError("IMFs and weights must contain only finite values.")
    return np.sum(imf_array[:-1] * weight_array[:, np.newaxis], axis=0)


def reconstruct_gated_signal(imfs: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Reconstruct using independent IMF gates (not sum-normalised weights)."""

    gate_array = np.asarray(gates, dtype=float)
    if np.any((gate_array < 0.0) | (gate_array > 1.0)):
        raise ValueError("Independent IMF gates must be bounded in [0, 1].")
    return reconstruct_weighted_signal(imfs, gate_array)
