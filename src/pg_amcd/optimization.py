"""Controlled Stage 1 filter-cutoff selection and seed stability.

Every candidate high-pass cutoff is evaluated on the exact same caller-owned
raw source segment.  Candidate preprocessing happens once, and all configured
CEEMDAN seeds decompose that same scaled candidate signal.  The objective uses
structural seed stability rather than reconstruction-error variance.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.signal

from pg_amcd.decomposition import (
    calculate_adjacent_imf_correlation,
    calculate_decomposition_metrics,
    calculate_frequency_ordering_score,
    calculate_orthogonality_index,
    calculate_seed_stability,
    calculate_spectral_overlap,
    decompose_ceemdan,
)
from pg_amcd.models import CutoffOptimizationResult
from pg_amcd.preprocessing import preprocess_signal_result


_DEFAULT_OBJECTIVE_WEIGHTS = {
    "spectral_overlap": 0.20,
    "maximum_adjacent_correlation": 0.15,
    "absolute_orthogonality": 0.15,
    "frequency_ordering_penalty": 0.15,
    "seed_instability": 0.20,
    "chatter_band_distortion": 0.15,
}


def _as_finite_raw_segment(raw_segment: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(raw_segment):
        raise ValueError("raw_segment must be real-valued.")
    segment = np.asarray(raw_segment, dtype=float)
    if segment.ndim != 1 or segment.size < 3:
        raise ValueError("raw_segment must be a one-dimensional array with at least 3 samples.")
    if not np.all(np.isfinite(segment)):
        raise ValueError("raw_segment contains NaN or infinite values.")
    return segment


def _resolve_preprocessing_config(config: Dict[str, Any], fs: float) -> Dict[str, Any]:
    """Resolve explicit high-pass/low-pass/filter settings with legacy aliases."""

    pre = config.get("preprocessing", {})
    nyquist = 0.5 * float(fs)
    configured_lowpass = pre.get(
        "low_pass_cutoff_hz",
        pre.get(
            "lowpass_cutoff_hz",
            pre.get("upper_cutoff_hz", pre.get("high_cutoff_hz")),
        ),
    )
    lowpass = min(4000.0, nyquist - 10.0) if configured_lowpass is None else configured_lowpass
    configured_order = pre.get("filter_order", 3)
    try:
        numeric_order = float(configured_order)
    except (TypeError, ValueError) as exc:
        raise ValueError("preprocessing.filter_order must be a positive integer.") from exc
    if (
        isinstance(configured_order, bool)
        or not np.isfinite(numeric_order)
        or not numeric_order.is_integer()
        or numeric_order < 1
    ):
        raise ValueError("preprocessing.filter_order must be a positive integer.")

    configured_padlen = pre.get("padlen")
    if configured_padlen is not None:
        try:
            numeric_padlen = float(configured_padlen)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "preprocessing.padlen must be a non-negative integer or null."
            ) from exc
        if (
            isinstance(configured_padlen, bool)
            or not np.isfinite(numeric_padlen)
            or not numeric_padlen.is_integer()
            or numeric_padlen < 0
        ):
            raise ValueError("preprocessing.padlen must be a non-negative integer or null.")
        configured_padlen = int(numeric_padlen)

    params = {
        "lowpass_cutoff_hz": float(lowpass),
        "filter_order": int(numeric_order),
        "detrend_type": str(pre.get("detrend_type", "linear")),
        "detrend_before_filter": bool(pre.get("detrend_before_filter", False)),
        "scale_percentile": float(pre.get("scale_percentile", 99.5)),
        "padtype": pre.get("padtype", "odd"),
        "padlen": configured_padlen,
    }
    return params


def _resolve_objective_weights(config: Dict[str, Any]) -> Dict[str, float]:
    search_cfg = config.get("cutoff_search", {})
    configured = search_cfg.get("objective_weights", {})
    weights = {
        key: float(configured.get(key, default))
        for key, default in _DEFAULT_OBJECTIVE_WEIGHTS.items()
    }
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Cutoff objective weights must be finite and non-negative.")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("At least one cutoff objective weight must be positive.")
    return {key: value / total for key, value in weights.items()}


def _band_energy(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    nperseg = min(int(signal.size), 1024)
    freqs, psd = scipy.signal.welch(signal, fs=float(fs), nperseg=nperseg)
    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.maximum(psd[mask], 0.0)))


def calculate_chatter_band_distortion(
    raw_segment: np.ndarray,
    physical_preprocessed_segment: np.ndarray,
    fs: float,
    center_hz: float,
    spread_hz: float,
) -> float:
    """Symmetric chatter-band energy change caused by candidate filtering.

    This deliberately compares the candidate's *physical preprocessed signal*
    with the controlled raw segment.  Comparing the source with the sum of all
    CEEMDAN components is uninformative because a valid decomposition
    reconstructs its own source nearly perfectly.
    """

    raw = _as_finite_raw_segment(raw_segment)
    candidate = _as_finite_raw_segment(physical_preprocessed_segment)
    if raw.shape != candidate.shape:
        raise ValueError("Raw and preprocessed segments must have identical shapes.")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be finite and positive, got {fs}.")
    if not np.isfinite(float(center_hz)) or float(center_hz) < 0:
        raise ValueError("center_hz must be finite and non-negative.")
    if not np.isfinite(float(spread_hz)) or float(spread_hz) <= 0:
        raise ValueError("spread_hz must be finite and positive.")

    low = max(0.0, float(center_hz) - float(spread_hz))
    high = min(0.5 * fs, float(center_hz) + float(spread_hz))
    if high <= low:
        return 0.0
    raw_energy = _band_energy(raw, fs, low, high)
    candidate_energy = _band_energy(candidate, fs, low, high)
    if raw_energy <= np.finfo(float).tiny:
        return 0.0 if candidate_energy <= np.finfo(float).tiny else 1.0

    # A log ratio penalises attenuation and amplification symmetrically, then
    # maps the result smoothly into [0, 1].
    ratio = max(candidate_energy, np.finfo(float).tiny) / raw_energy
    return float(np.clip(1.0 - np.exp(-abs(np.log(ratio))), 0.0, 1.0))


def _objective_from_components(
    *,
    spectral_overlap: float,
    maximum_adjacent_correlation: float,
    absolute_orthogonality: float,
    frequency_ordering_score: float,
    seed_instability: float,
    chatter_band_distortion: float,
    weights: Dict[str, float],
) -> float:
    components = {
        "spectral_overlap": np.clip(spectral_overlap, 0.0, 1.0),
        "maximum_adjacent_correlation": np.clip(maximum_adjacent_correlation, 0.0, 1.0),
        "absolute_orthogonality": np.clip(absolute_orthogonality, 0.0, 1.0),
        "frequency_ordering_penalty": np.clip(1.0 - frequency_ordering_score, 0.0, 1.0),
        "seed_instability": np.clip(seed_instability, 0.0, 1.0),
        "chatter_band_distortion": np.clip(chatter_band_distortion, 0.0, 1.0),
    }
    return float(sum(weights[key] * float(value) for key, value in components.items()))


def compute_cutoff_objective(
    imfs: np.ndarray,
    original: np.ndarray,
    fs: float,
    config: Dict[str, Any],
    seed_instability: float = 0.0,
    *,
    chatter_band_distortion: float = 0.0,
) -> float:
    """Compatibility objective for a legacy ``physical IMFs + residual`` matrix.

    Lower is better.  New cutoff search uses the same components but supplies a
    structural multi-seed score and a raw-vs-filtered chatter-band distortion.
    """

    components = np.asarray(imfs, dtype=float)
    if components.ndim != 2 or components.shape[0] < 1:
        raise ValueError("imfs must contain at least one component.")
    if components.shape[0] == 1:
        # Preserve the historical objective's useful single-mode edge case.
        # There is no adjacent pair and no inferred residual in this form.
        physical_imfs = components
        max_adjacent = 0.0
        overlap = 0.0
    else:
        physical_imfs = components[:-1]
        _, max_adjacent = calculate_adjacent_imf_correlation(components)
        overlap = calculate_spectral_overlap(components, fs)
    signed_oi = calculate_orthogonality_index(components, original)
    ordering = calculate_frequency_ordering_score(physical_imfs, fs)
    return _objective_from_components(
        spectral_overlap=overlap,
        maximum_adjacent_correlation=max_adjacent,
        absolute_orthogonality=abs(signed_oi),
        frequency_ordering_score=ordering,
        seed_instability=float(seed_instability),
        chatter_band_distortion=float(chatter_band_distortion),
        weights=_resolve_objective_weights(config),
    )


def _resolve_seed_values(
    ceemdan_cfg: Dict[str, Any],
    n_seeds: int,
    seeds: Optional[Sequence[int]],
) -> List[int]:
    if seeds is not None:
        resolved = [int(seed) for seed in seeds]
    else:
        configured = ceemdan_cfg.get(
            "search_seed_values",
            ceemdan_cfg.get("stability_seeds"),
        )
        if configured is not None:
            resolved = [int(seed) for seed in configured]
        else:
            if isinstance(n_seeds, bool) or int(n_seeds) != n_seeds or int(n_seeds) < 1:
                raise ValueError(f"n_seeds must be a positive integer, got {n_seeds}.")
            base = int(ceemdan_cfg.get("noise_seed", 42))
            resolved = [base + index for index in range(int(n_seeds))]
    if not resolved:
        raise ValueError("At least one CEEMDAN search seed is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("CEEMDAN search seeds must be unique.")
    return resolved


def optimize_cutoff(
    raw_segment: np.ndarray,
    candidate_cutoffs: Sequence[float],
    config: Dict[str, Any],
    fs: float,
    n_seeds: int = 2,
    seeds: Optional[Sequence[int]] = None,
) -> CutoffOptimizationResult:
    """Select a high-pass cutoff on one immutable controlled source segment."""

    segment = _as_finite_raw_segment(raw_segment)
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be finite and positive, got {fs}.")
    if not candidate_cutoffs:
        raise ValueError("candidate_cutoffs must be a non-empty sequence.")
    candidates = [float(cutoff) for cutoff in candidate_cutoffs]
    if any(not np.isfinite(cutoff) for cutoff in candidates):
        raise ValueError("candidate_cutoffs must all be finite.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_cutoffs must be unique.")

    preprocessing_cfg = _resolve_preprocessing_config(config, fs)
    lowpass_cutoff = float(preprocessing_cfg["lowpass_cutoff_hz"])
    if any(cutoff <= 0 or cutoff >= lowpass_cutoff for cutoff in candidates):
        raise ValueError(
            "Every candidate high-pass cutoff must satisfy "
            f"0 < cutoff < configured low-pass cutoff ({lowpass_cutoff:g} Hz)."
        )

    ceemdan_cfg = config.get("ceemdan", {})
    seed_values = _resolve_seed_values(ceemdan_cfg, n_seeds, seeds)
    search_trials = int(ceemdan_cfg.get("search_trials", ceemdan_cfg.get("trials", 50)))
    search_sift = int(
        ceemdan_cfg.get(
            "search_sifting_iterations",
            ceemdan_cfg.get("sifting_iterations", 16),
        )
    )
    epsilon = float(ceemdan_cfg.get("epsilon", 0.02))
    weights = _resolve_objective_weights(config)

    maiw = config.get("maiw", {})
    chatter_center = float(maiw.get("chatter_band_center", 1250.0))
    chatter_spread = float(maiw.get("chatter_band_spread", 500.0))
    source_checksum = hashlib.sha256(np.ascontiguousarray(segment).view(np.uint8)).hexdigest()

    ceemdan_options = {
        "parallel": bool(ceemdan_cfg.get("parallel", True)),
        "processes": ceemdan_cfg.get("processes"),
        "max_imf": int(ceemdan_cfg.get("max_imf", -1)),
        "noise_scale": float(ceemdan_cfg.get("noise_scale", 1.0)),
        "noise_kind": str(ceemdan_cfg.get("noise_kind", "normal")),
        "range_threshold": float(ceemdan_cfg.get("range_threshold", 0.01)),
        "total_power_threshold": float(ceemdan_cfg.get("total_power_threshold", 0.05)),
        "beta_progress": bool(ceemdan_cfg.get("beta_progress", True)),
    }

    per_cutoff: List[Dict[str, Any]] = []
    for cutoff in candidates:
        # Preprocess this candidate once.  Every seed below consumes the exact
        # same scaled array, and every candidate derives from the same immutable
        # raw segment captured above.
        prep = preprocess_signal_result(
            segment,
            cutoff,
            lowpass_cutoff,
            fs,
            order=preprocessing_cfg["filter_order"],
            detrend_type=preprocessing_cfg["detrend_type"],
            detrend_before_filter=preprocessing_cfg["detrend_before_filter"],
            scale_percentile=preprocessing_cfg["scale_percentile"],
            padtype=preprocessing_cfg["padtype"],
            padlen=preprocessing_cfg["padlen"],
        )
        decompositions = [
            decompose_ceemdan(
                prep.scaled_signal,
                search_trials,
                epsilon,
                seed,
                search_sift,
                **ceemdan_options,
            )
            for seed in seed_values
        ]
        seed_metrics = [
            calculate_decomposition_metrics(prep.scaled_signal, result, fs)
            for result in decompositions
        ]
        stability = calculate_seed_stability(decompositions, fs, seeds=seed_values)
        band_distortion = calculate_chatter_band_distortion(
            segment,
            prep.physical_signal,
            fs,
            chatter_center,
            chatter_spread,
        )

        spectral_overlap = float(np.mean([row["spectral_overlap"] for row in seed_metrics]))
        max_adjacent = float(
            np.mean([row["maximum_adjacent_imf_correlation"] for row in seed_metrics])
        )
        signed_oi = float(np.mean([row["signed_orthogonality_index"] for row in seed_metrics]))
        absolute_oi = float(np.mean([row["absolute_orthogonality_index"] for row in seed_metrics]))
        ordering = float(np.mean([row["frequency_ordering_score"] for row in seed_metrics]))
        objective = _objective_from_components(
            spectral_overlap=spectral_overlap,
            maximum_adjacent_correlation=max_adjacent,
            absolute_orthogonality=absolute_oi,
            frequency_ordering_score=ordering,
            seed_instability=stability["instability_score"],
            chatter_band_distortion=band_distortion,
            weights=weights,
        )

        per_cutoff.append(
            {
                # `cutoff` remains for backwards compatibility; the explicit
                # name records that it is the lower/high-pass band edge.
                "cutoff": float(cutoff),
                "highpass_cutoff_hz": float(cutoff),
                "lowpass_cutoff_hz": lowpass_cutoff,
                "high_pass_cutoff_hz": float(cutoff),
                "low_pass_cutoff_hz": lowpass_cutoff,
                "filter_order": int(preprocessing_cfg["filter_order"]),
                "source_segment_sha256": source_checksum,
                "source_segment_samples": int(segment.size),
                "scale_factor": float(prep.scale_factor),
                "objective": objective,
                "final_score": objective,
                "objective_weights": dict(weights),
                "spectral_overlap": spectral_overlap,
                "maximum_adjacent_correlation": max_adjacent,
                "signed_orthogonality_index": signed_oi,
                "absolute_orthogonality_index": absolute_oi,
                "frequency_ordering_score": ordering,
                "frequency_ordering_penalty": float(1.0 - ordering),
                "chatter_band_distortion": float(band_distortion),
                "seed_instability": float(stability["instability_score"]),
                "seed_stability": stability,
                "mean_reconstruction_nrmse": float(
                    np.mean([row["reconstruction_nrmse"] for row in seed_metrics])
                ),
                "ceemdan_seeds": list(seed_values),
            }
        )

    best = min(per_cutoff, key=lambda row: row["final_score"])
    return CutoffOptimizationResult(
        selected_cutoff=float(best["cutoff"]),
        per_cutoff_metrics=per_cutoff,
        best_score=float(best["final_score"]),
    )


def multi_seed_stability(func, seeds: Sequence[int]) -> Dict[str, float]:
    """Legacy scalar-run summary retained for non-decomposition experiments.

    CEEMDAN Stage 1 stability must use
    :func:`pg_amcd.decomposition.calculate_seed_stability` instead.
    """

    values = np.asarray([float(func(int(seed))) for seed in seeds], dtype=float)
    if values.size == 0:
        raise ValueError("seeds must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("func returned a non-finite value.")
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci = 1.96 * std / np.sqrt(values.size) if values.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": int(values.size),
    }
