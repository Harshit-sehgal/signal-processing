"""Stage 3: reconstruction-level Bayesian adaptive wavelet denoising.

The canonical production approach in this module denoises the Stage 2 weighted
reconstruction.  :func:`wavelet_denoise_with_diagnostics` exposes the complete
coefficient, threshold, subband-overlap, and metric record.  The historical
array-returning :func:`wavelet_denoise` wrapper remains API-compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pywt
import scipy.signal


@dataclass(frozen=True)
class WaveletLevelDiagnostic:
    """One approximation/detail subband and its applied threshold."""

    coefficient_name: str
    level: int
    is_approximation: bool
    coefficient_count: int
    frequency_low_hz: float
    frequency_high_hz: float
    chatter_overlap_fraction: float
    input_energy: float
    output_energy: float
    threshold: float
    threshold_scale: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WaveletDenoisingMetrics:
    """Quantitative Stage 3 metrics, with reference metrics only when supplied."""

    rms_before: float
    rms_after: float
    energy_before: float
    energy_after: float
    correlation_before_after: float
    chatter_band_retention: float
    out_of_band_attenuation: float
    spectral_distortion: float
    transient_preservation: float
    estimated_noise_sigma: float
    synthetic_reference_rmse: Optional[float]
    synthetic_reference_snr_db: Optional[float]
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass(frozen=True)
class WaveletDenoisingResult:
    """Typed result containing everything needed for Stage 3 artifacts."""

    denoised_signal: np.ndarray
    approximation_coefficients: np.ndarray
    detail_coefficients: Dict[str, np.ndarray]
    thresholded_detail_coefficients: Dict[str, np.ndarray]
    level_diagnostics: Tuple[WaveletLevelDiagnostic, ...]
    metrics: WaveletDenoisingMetrics
    wavelet_name: str
    requested_level: int
    applied_level: int
    threshold_mode: str
    band_aware: bool
    chatter_band_hz: Tuple[float, float]
    input_length: int

    @property
    def thresholds_by_level(self) -> Dict[str, float]:
        return {
            row.coefficient_name: row.threshold
            for row in self.level_diagnostics
            if not row.is_approximation
        }

    def to_dict(self, *, include_signal: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "wavelet_name": self.wavelet_name,
            "requested_level": self.requested_level,
            "applied_level": self.applied_level,
            "threshold_mode": self.threshold_mode,
            "band_aware": self.band_aware,
            "chatter_band_hz": list(self.chatter_band_hz),
            "input_length": self.input_length,
            "output_length": int(self.denoised_signal.size),
            "thresholds_by_level": self.thresholds_by_level,
            "levels": [row.to_dict() for row in self.level_diagnostics],
            "metrics": self.metrics.to_dict(),
        }
        if include_signal:
            payload["denoised_signal"] = self.denoised_signal.tolist()
        return payload


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive, got {number}.")
    return number


def bayes_shrink_threshold(coeff: np.ndarray, noise_sigma: float) -> float:
    """Return the standard BayesShrink threshold ``sigma_n**2 / sigma_x``."""

    coefficient = np.asarray(coeff, dtype=float)
    sigma = _finite_number(noise_sigma, "noise_sigma")
    if coefficient.ndim != 1 or coefficient.size == 0:
        raise ValueError("Wavelet detail coefficients must be a non-empty 1-D array.")
    if not np.all(np.isfinite(coefficient)):
        raise ValueError("Wavelet detail coefficients contain non-finite values.")
    if sigma < 0:
        raise ValueError("noise_sigma must be non-negative.")
    variance_observed = float(np.mean(np.square(coefficient)))
    variance_signal = max(0.0, variance_observed - sigma**2)
    if variance_signal <= np.finfo(float).eps:
        return float(np.max(np.abs(coefficient)))
    return float(sigma**2 / np.sqrt(variance_signal))


def _subband_frequency_range(fs: float, level: int) -> Tuple[float, float]:
    """Ideal dyadic support of detail subband ``cD_level``."""

    return fs * 2.0 ** (-(level + 1)), fs * 2.0 ** (-level)


def _interval_overlap_fraction(
    interval: Tuple[float, float], target: Tuple[float, float]
) -> float:
    low, high = interval
    target_low, target_high = target
    width = high - low
    if width <= 0:
        return 0.0
    intersection = max(0.0, min(high, target_high) - max(low, target_low))
    return float(np.clip(intersection / width, 0.0, 1.0))


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a_centered, b_centered) / denominator)


def _denoising_metrics(
    source: np.ndarray,
    denoised: np.ndarray,
    fs: float,
    chatter_band: Tuple[float, float],
    noise_sigma: float,
    runtime_seconds: float,
    clean_reference: Optional[np.ndarray],
) -> WaveletDenoisingMetrics:
    freqs, before_psd = scipy.signal.welch(
        source, fs=fs, nperseg=min(source.size, 1024)
    )
    _, after_psd = scipy.signal.welch(
        denoised, fs=fs, nperseg=min(denoised.size, 1024)
    )
    chatter_mask = (freqs >= chatter_band[0]) & (freqs <= chatter_band[1])
    out_mask = ~chatter_mask

    def retention(mask: np.ndarray) -> float:
        before = float(np.sum(before_psd[mask]))
        after = float(np.sum(after_psd[mask]))
        return after / before if before > 0 else 0.0

    before_total = float(np.sum(before_psd))
    after_total = float(np.sum(after_psd))
    if before_total > 0 and after_total > 0:
        distortion = float(
            np.sum(np.abs(before_psd / before_total - after_psd / after_total))
        )
    else:
        distortion = 0.0

    # Transients are defined reproducibly as the top 5% of absolute input
    # amplitudes.  The metric is output/input energy on exactly those samples.
    transient_cutoff = float(np.quantile(np.abs(source), 0.95))
    transient_mask = np.abs(source) >= transient_cutoff
    transient_input = float(np.sum(np.square(source[transient_mask])))
    transient_output = float(np.sum(np.square(denoised[transient_mask])))
    transient_preservation = (
        transient_output / transient_input if transient_input > 0 else 0.0
    )

    reference_rmse: Optional[float] = None
    reference_snr: Optional[float] = None
    if clean_reference is not None:
        residual = clean_reference - denoised
        reference_rmse = float(np.sqrt(np.mean(np.square(residual))))
        signal_power = float(np.mean(np.square(clean_reference)))
        residual_power = float(np.mean(np.square(residual)))
        reference_snr = (
            float(10.0 * np.log10(signal_power / residual_power))
            if signal_power > 0 and residual_power > 0
            else float("inf") if signal_power > 0 else 0.0
        )

    return WaveletDenoisingMetrics(
        rms_before=float(np.sqrt(np.mean(np.square(source)))),
        rms_after=float(np.sqrt(np.mean(np.square(denoised)))),
        energy_before=float(np.sum(np.square(source))),
        energy_after=float(np.sum(np.square(denoised))),
        correlation_before_after=_safe_correlation(source, denoised),
        chatter_band_retention=retention(chatter_mask),
        out_of_band_attenuation=1.0 - retention(out_mask),
        spectral_distortion=distortion,
        transient_preservation=transient_preservation,
        estimated_noise_sigma=noise_sigma,
        synthetic_reference_rmse=reference_rmse,
        synthetic_reference_snr_db=reference_snr,
        runtime_seconds=runtime_seconds,
    )


def wavelet_denoise_with_diagnostics(
    signal: np.ndarray,
    wavelet_name: str = "db8",
    level: int = 4,
    fs: float = 10000.0,
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0,
    band_aware: bool = True,
    chatter_threshold_scale: float = 0.5,
    noise_threshold_scale: float = 1.4,
    threshold_mode: str = "soft",
    min_noise_sigma: float = 1e-6,
    minimum_noise_sigma: Optional[float] = None,
    clean_reference: Optional[np.ndarray] = None,
) -> WaveletDenoisingResult:
    """Denoise a weighted reconstruction and retain full per-level diagnostics.

    Noise is estimated from the finest detail coefficients (``cD_1``) using
    median absolute deviation.  For band-aware processing, the threshold scale
    interpolates between the chatter and noise multipliers using the *fraction*
    of each ideal dyadic detail band overlapped by the chatter band.
    """

    started = time.perf_counter()
    source = np.asarray(signal, dtype=float)
    if source.ndim != 1:
        raise ValueError("Signal for wavelet denoising must be one-dimensional.")
    if source.size < 2:
        raise ValueError(f"Signal too short for wavelet denoising: {source.size} samples")
    if not np.all(np.isfinite(source)):
        raise ValueError("Signal for wavelet denoising contains non-finite values.")
    sampling_rate = _finite_number(fs, "sampling rate", positive=True)
    center = _finite_number(chatter_center, "chatter_center", positive=True)
    spread = _finite_number(chatter_spread, "chatter_spread", positive=True)
    chatter_scale = _finite_number(chatter_threshold_scale, "chatter_threshold_scale")
    noise_scale = _finite_number(noise_threshold_scale, "noise_threshold_scale")
    sigma_setting = min_noise_sigma if minimum_noise_sigma is None else minimum_noise_sigma
    minimum_sigma = _finite_number(
        sigma_setting, "minimum_noise_sigma", positive=True
    )
    if chatter_scale < 0 or noise_scale < 0:
        raise ValueError("Threshold multipliers must be non-negative.")
    if threshold_mode not in {"soft", "hard"}:
        raise ValueError("threshold_mode must be either 'soft' or 'hard'.")
    if not isinstance(level, (int, np.integer)) or isinstance(level, bool) or level < 1:
        raise ValueError(f"Wavelet decomposition level must be an integer >= 1, got {level}")

    wavelet = pywt.Wavelet(wavelet_name)  # clear ValueError for invalid names
    max_level = pywt.dwt_max_level(source.size, wavelet.dec_len)
    if max_level < 1:
        raise ValueError(
            f"Signal length {source.size} is too short for wavelet '{wavelet_name}' "
            "(max decomposition level is 0)"
        )
    requested_level = int(level)
    applied_level = min(requested_level, max_level)
    chatter_band = (
        max(0.0, center - spread),
        min(sampling_rate / 2.0, center + spread),
    )
    if chatter_band[0] >= chatter_band[1]:
        raise ValueError(
            "Configured chatter band does not overlap the measurable range [0, Nyquist]."
        )

    reference: Optional[np.ndarray] = None
    if clean_reference is not None:
        reference = np.asarray(clean_reference, dtype=float)
        if reference.shape != source.shape or not np.all(np.isfinite(reference)):
            raise ValueError("clean_reference must be finite and have the same shape as signal.")

    # [cA_N, cD_N, cD_N-1, ..., cD_1]
    coefficients = pywt.wavedec(source, wavelet, level=applied_level)
    finest_detail = np.asarray(coefficients[-1], dtype=float)
    mad = float(np.median(np.abs(finest_detail - np.median(finest_detail))))
    noise_sigma = max(minimum_sigma, mad / 0.6745)

    denoised_coefficients = [np.asarray(coefficients[0], dtype=float).copy()]
    detail_coefficients: Dict[str, np.ndarray] = {}
    thresholded_details: Dict[str, np.ndarray] = {}
    level_rows = [
        WaveletLevelDiagnostic(
            coefficient_name=f"cA_{applied_level}",
            level=applied_level,
            is_approximation=True,
            coefficient_count=int(coefficients[0].size),
            frequency_low_hz=0.0,
            frequency_high_hz=sampling_rate * 2.0 ** (-(applied_level + 1)),
            chatter_overlap_fraction=0.0,
            input_energy=float(np.sum(np.square(coefficients[0]))),
            output_energy=float(np.sum(np.square(coefficients[0]))),
            threshold=0.0,
            threshold_scale=0.0,
        )
    ]

    for coefficient_index in range(1, len(coefficients)):
        detail_level = applied_level - coefficient_index + 1
        coefficient_name = f"cD_{detail_level}"
        coefficient = np.asarray(coefficients[coefficient_index], dtype=float)
        frequency_range = _subband_frequency_range(sampling_rate, detail_level)
        overlap_fraction = _interval_overlap_fraction(frequency_range, chatter_band)
        base_threshold = bayes_shrink_threshold(coefficient, noise_sigma)
        if band_aware:
            threshold_scale = (
                chatter_scale * overlap_fraction
                + noise_scale * (1.0 - overlap_fraction)
            )
        else:
            threshold_scale = 1.0
        threshold = base_threshold * threshold_scale
        thresholded = pywt.threshold(coefficient, threshold, mode=threshold_mode)
        denoised_coefficients.append(thresholded)
        detail_coefficients[coefficient_name] = coefficient.copy()
        thresholded_details[coefficient_name] = np.asarray(thresholded, dtype=float).copy()
        level_rows.append(
            WaveletLevelDiagnostic(
                coefficient_name=coefficient_name,
                level=detail_level,
                is_approximation=False,
                coefficient_count=int(coefficient.size),
                frequency_low_hz=frequency_range[0],
                frequency_high_hz=frequency_range[1],
                chatter_overlap_fraction=overlap_fraction,
                input_energy=float(np.sum(np.square(coefficient))),
                output_energy=float(np.sum(np.square(thresholded))),
                threshold=float(threshold),
                threshold_scale=float(threshold_scale),
            )
        )

    denoised = np.asarray(pywt.waverec(denoised_coefficients, wavelet), dtype=float)
    if denoised.size > source.size:
        denoised = denoised[: source.size]
    elif denoised.size < source.size:
        denoised = np.pad(denoised, (0, source.size - denoised.size), mode="edge")
    if denoised.shape != source.shape or not np.all(np.isfinite(denoised)):
        raise RuntimeError("Wavelet reconstruction did not produce a finite, length-preserving signal.")

    runtime = time.perf_counter() - started
    metrics = _denoising_metrics(
        source,
        denoised,
        sampling_rate,
        chatter_band,
        noise_sigma,
        runtime,
        reference,
    )
    return WaveletDenoisingResult(
        denoised_signal=denoised,
        approximation_coefficients=np.asarray(coefficients[0], dtype=float).copy(),
        detail_coefficients=detail_coefficients,
        thresholded_detail_coefficients=thresholded_details,
        level_diagnostics=tuple(level_rows),
        metrics=metrics,
        wavelet_name=wavelet.name,
        requested_level=requested_level,
        applied_level=applied_level,
        threshold_mode=threshold_mode,
        band_aware=bool(band_aware),
        chatter_band_hz=chatter_band,
        input_length=int(source.size),
    )


def wavelet_denoise(
    signal: np.ndarray,
    wavelet_name: str = "db8",
    level: int = 4,
    fs: float = 10000.0,
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0,
    band_aware: bool = True,
    chatter_threshold_scale: float = 0.5,
    noise_threshold_scale: float = 1.4,
    threshold_mode: str = "soft",
    min_noise_sigma: float = 1e-6,
    minimum_noise_sigma: Optional[float] = None,
) -> np.ndarray:
    """Compatibility wrapper returning only the denoised reconstruction."""

    return wavelet_denoise_with_diagnostics(
        signal,
        wavelet_name=wavelet_name,
        level=level,
        fs=fs,
        chatter_center=chatter_center,
        chatter_spread=chatter_spread,
        band_aware=band_aware,
        chatter_threshold_scale=chatter_threshold_scale,
        noise_threshold_scale=noise_threshold_scale,
        threshold_mode=threshold_mode,
        min_noise_sigma=min_noise_sigma,
        minimum_noise_sigma=minimum_noise_sigma,
    ).denoised_signal


def evaluate_wavelet_config(
    signal: np.ndarray,
    clean_reference: np.ndarray,
    wavelet_name: str,
    level: int,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
) -> dict:
    """Quantitative denoising quality for one synthetic-reference configuration."""

    from pg_amcd.synthetic import evaluate_denoising_performance

    denoised = wavelet_denoise(
        signal,
        wavelet_name=wavelet_name,
        level=level,
        fs=fs,
        chatter_center=chatter_center,
        chatter_spread=chatter_spread,
        band_aware=True,
    )
    return evaluate_denoising_performance(
        clean_reference, denoised, fs, chatter_center, chatter_spread
    )


def select_best_wavelet(
    signal: np.ndarray,
    clean_reference: np.ndarray,
    candidates: list,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
):
    """Evaluate candidate ``(wavelet, level)`` pairs and return highest reference SNR."""

    if not candidates:
        raise ValueError("At least one candidate wavelet configuration is required.")
    results = []
    for wavelet_name, level in candidates:
        metrics = evaluate_wavelet_config(
            signal,
            clean_reference,
            wavelet_name,
            level,
            fs,
            chatter_center,
            chatter_spread,
        )
        results.append({"wavelet": wavelet_name, "level": level, **metrics})
    best = max(results, key=lambda row: row["snr_db"])
    return best, results
