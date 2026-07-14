"""Canonical CEEMDAN decomposition and Stage 1 scientific diagnostics.

PyEMD's :class:`~PyEMD.CEEMDAN` returns the final reconstruction remainder as
the last row of its component matrix.  This module verifies that relationship
at runtime and exposes physical IMFs and the residual separately through
:class:`CEEMDANResult`.  The historical :func:`run_ceemdan` array API remains a
thin compatibility wrapper around the explicit implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from itertools import combinations
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal
from scipy.optimize import linear_sum_assignment
from PyEMD import CEEMDAN


@dataclass(frozen=True)
class CEEMDANResult:
    """Explicit CEEMDAN result with a verified residual split.

    All arrays are in the same units as the input signal.  When the input is
    the Stage 1 scaled segment these are scaled units; multiplication by the
    preprocessing scale factor restores physical units.
    """

    imfs: np.ndarray
    residual: np.ndarray
    components: np.ndarray
    parameters: Dict[str, Any]
    runtime_seconds: float
    residual_verified: bool
    residual_verification_nrmse: float
    reconstruction_nrmse: float
    residual_source: str = (
        "final PyEMD CEEMDAN component, verified against input - sum(physical IMFs)"
    )

    @property
    def num_imfs(self) -> int:
        return int(self.imfs.shape[0])

    @property
    def reconstruction(self) -> np.ndarray:
        return np.sum(self.imfs, axis=0) + self.residual

    def as_metadata(self) -> Dict[str, Any]:
        """Return the JSON-safe non-array metadata for provenance artifacts."""

        return {
            "number_of_imfs": self.num_imfs,
            "parameters": dict(self.parameters),
            "runtime_seconds": float(self.runtime_seconds),
            "residual_verified": bool(self.residual_verified),
            "residual_verification_nrmse": float(self.residual_verification_nrmse),
            "reconstruction_nrmse": float(self.reconstruction_nrmse),
            "residual_source": self.residual_source,
        }


@dataclass(frozen=True)
class IMFMetrics:
    """One physical IMF's Stage 1 energy and frequency descriptors."""

    imf_index: int
    energy_percentage: float
    centre_frequency_hz: float
    bandwidth_hz: float
    spectral_entropy: float
    rms: float

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


@dataclass(frozen=True)
class OrthogonalityMetrics:
    """Signed and absolute forms of the global orthogonality index."""

    signed_index: float
    absolute_index: float
    pairwise_absolute_index: float


def _as_finite_signal(signal: np.ndarray, *, name: str = "signal") -> np.ndarray:
    if np.iscomplexobj(signal):
        raise ValueError(f"{name} must be real-valued.")
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least three samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _as_imf_matrix(imfs: np.ndarray, *, name: str = "imfs") -> np.ndarray:
    if np.iscomplexobj(imfs):
        raise ValueError(f"{name} must be real-valued.")
    arr = np.asarray(imfs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix; got shape {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 3:
        raise ValueError(f"{name} must have shape (n_modes >= 1, n_samples >= 3).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError(
            f"NRMSE inputs must have identical shapes, got {reference.shape} and {estimate.shape}."
        )
    denom = float(np.sqrt(np.mean(reference**2)))
    error = float(np.sqrt(np.mean((reference - estimate) ** 2)))
    if denom <= np.finfo(float).tiny:
        return 0.0 if error <= np.finfo(float).tiny else float("inf")
    return error / denom


def _emd_signal_version() -> str:
    try:
        return importlib_metadata.version("EMD-signal")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _coerce_integer(
    value: Any,
    *,
    name: str,
    minimum: int = 1,
    allow_minus_one: bool = False,
) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value}.") from exc
    if (
        isinstance(value, bool)
        or not np.isfinite(numeric)
        or not numeric.is_integer()
    ):
        raise ValueError(f"{name} must be an integer, got {value}.")
    resolved = int(numeric)
    if allow_minus_one and resolved == -1:
        return resolved
    if resolved < minimum:
        qualifier = f"at least {minimum}"
        if allow_minus_one:
            qualifier = f"-1 or {qualifier}"
        raise ValueError(f"{name} must be {qualifier}, got {value}.")
    return resolved


def decompose_ceemdan(
    signal: np.ndarray,
    trials: int,
    epsilon: float,
    noise_seed: int,
    sifting_iterations: int = 16,
    *,
    parallel: bool = True,
    processes: Optional[int] = None,
    max_imf: int = -1,
    noise_scale: float = 1.0,
    noise_kind: str = "normal",
    range_threshold: float = 0.01,
    total_power_threshold: float = 0.05,
    beta_progress: bool = True,
    residual_rtol: float = 1e-7,
    residual_atol: float = 1e-10,
) -> CEEMDANResult:
    """Run PyEMD CEEMDAN and explicitly verify/split its final residual row."""

    arr = _as_finite_signal(signal)
    if float(np.std(arr)) <= np.finfo(float).tiny:
        raise ValueError("CEEMDAN cannot decompose a constant or numerically flat signal.")
    trials = _coerce_integer(trials, name="trials")
    if not np.isfinite(float(epsilon)) or float(epsilon) < 0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}.")
    sifting_iterations = _coerce_integer(
        sifting_iterations,
        name="sifting_iterations",
    )
    max_imf = _coerce_integer(
        max_imf,
        name="max_imf",
        allow_minus_one=True,
    )
    noise_seed = _coerce_integer(
        noise_seed,
        name="noise_seed",
        minimum=0,
    )
    if noise_seed >= 2**32:
        raise ValueError(f"noise_seed must be smaller than 2**32, got {noise_seed}.")
    if processes is not None:
        processes = _coerce_integer(processes, name="processes")
    if noise_kind not in {"normal", "uniform"}:
        raise ValueError(f"noise_kind must be 'normal' or 'uniform', got {noise_kind!r}.")
    for value, label, allow_zero in (
        (noise_scale, "noise_scale", False),
        (range_threshold, "range_threshold", True),
        (total_power_threshold, "total_power_threshold", True),
    ):
        value = float(value)
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{label} must be finite and {relation}, got {value}.")
    for value, label in ((residual_rtol, "residual_rtol"), (residual_atol, "residual_atol")):
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{label} must be finite and non-negative, got {value}.")

    kwargs: Dict[str, Any] = {
        "trials": trials,
        "epsilon": float(epsilon),
        "FIXE": sifting_iterations,
        "parallel": bool(parallel),
        "noise_scale": float(noise_scale),
        "noise_kind": noise_kind,
        "range_thr": float(range_threshold),
        "total_power_thr": float(total_power_threshold),
        "beta_progress": bool(beta_progress),
    }
    if processes is not None:
        kwargs["processes"] = processes

    ceemdan = CEEMDAN(**kwargs)
    ceemdan.noise_seed(noise_seed)

    start = time.perf_counter()
    components = np.asarray(ceemdan(arr, max_imf=max_imf), dtype=float)
    runtime = time.perf_counter() - start

    if components.ndim != 2 or components.shape[1] != arr.size:
        raise RuntimeError(
            "PyEMD CEEMDAN returned an unexpected component matrix shape: "
            f"{components.shape}; expected (n_components, {arr.size})."
        )
    if components.shape[0] < 2:
        raise RuntimeError(
            "PyEMD CEEMDAN returned fewer than two components, so a physical IMF/residual "
            "split cannot be established."
        )
    if not np.all(np.isfinite(components)):
        raise RuntimeError("PyEMD CEEMDAN returned non-finite component values.")

    # PyEMD CEEMDAN appends `S - sum(all_cimfs)` as the final returned row.
    # Verify this installed-library contract for every run instead of silently
    # relying on row position downstream.
    imfs = np.asarray(components[:-1], dtype=float)
    returned_residual = np.asarray(components[-1], dtype=float)
    computed_residual = arr - np.sum(imfs, axis=0)
    source_rms = float(np.sqrt(np.mean(arr**2)))
    residual_error = float(
        np.sqrt(np.mean((computed_residual - returned_residual) ** 2)) / source_rms
    )
    residual_verified = bool(
        np.allclose(
            returned_residual,
            computed_residual,
            rtol=float(residual_rtol),
            atol=float(residual_atol),
        )
    )
    if not residual_verified:
        raise RuntimeError(
            "Unable to verify PyEMD CEEMDAN's final component as the reconstruction "
            f"residual (verification NRMSE={residual_error:.3e})."
        )

    reconstruction_error = _nrmse(arr, np.sum(imfs, axis=0) + returned_residual)
    parameters = {
        "algorithm": "CEEMDAN",
        "implementation": "PyEMD.CEEMDAN",
        "emd_signal_version": _emd_signal_version(),
        "trials": trials,
        "epsilon": float(epsilon),
        "noise_seed": noise_seed,
        "sifting_iterations": sifting_iterations,
        "parallel": bool(parallel),
        "processes": processes,
        "max_imf": max_imf,
        "noise_scale": float(noise_scale),
        "noise_kind": noise_kind,
        "range_threshold": float(range_threshold),
        "total_power_threshold": float(total_power_threshold),
        "beta_progress": bool(beta_progress),
    }
    return CEEMDANResult(
        imfs=imfs,
        residual=returned_residual,
        components=np.asarray(components, dtype=float),
        parameters=parameters,
        runtime_seconds=float(runtime),
        residual_verified=True,
        residual_verification_nrmse=float(residual_error),
        reconstruction_nrmse=float(reconstruction_error),
    )


def run_ceemdan(
    signal: np.ndarray,
    trials: int,
    epsilon: float,
    noise_seed: int,
    sifting_iterations: int = 16,
    **kwargs: Any,
) -> np.ndarray:
    """Compatibility wrapper returning ``physical IMFs + final residual``.

    New Stage 1 code should call :func:`decompose_ceemdan` and consume the
    explicit ``imfs`` and ``residual`` fields.
    """

    return decompose_ceemdan(
        signal,
        trials,
        epsilon,
        noise_seed,
        sifting_iterations,
        **kwargs,
    ).components


def calculate_reconstruction_nrmse(
    original_signal: np.ndarray,
    imfs: np.ndarray,
    residual: Optional[np.ndarray] = None,
) -> float:
    """NRMSE between the source and an explicit IMF-plus-residual reconstruction."""

    source = _as_finite_signal(original_signal, name="original_signal")
    modes = _as_imf_matrix(imfs)
    if modes.shape[1] != source.size:
        raise ValueError("IMF and source-signal lengths differ.")
    reconstruction = np.sum(modes, axis=0)
    if residual is not None:
        residual_arr = _as_finite_signal(residual, name="residual")
        if residual_arr.size != source.size:
            raise ValueError("Residual and source-signal lengths differ.")
        reconstruction = reconstruction + residual_arr
    return float(_nrmse(source, reconstruction))


def calculate_orthogonality_metrics(imfs: np.ndarray) -> OrthogonalityMetrics:
    """Return signed, absolute, and pairwise-absolute global OI values.

    Pass exactly the components intended for the OI calculation.  Stage 1's
    aggregate metric includes the explicitly identified residual, while
    adjacent-IMF and per-IMF metrics deliberately do not.
    """

    modes = _as_imf_matrix(imfs)
    cross_terms: List[float] = []
    for i in range(modes.shape[0]):
        for j in range(i + 1, modes.shape[0]):
            cross_terms.append(float(np.sum(modes[i] * modes[j])))
    total_energy = float(np.sum(modes**2))
    if total_energy <= np.finfo(float).tiny:
        return OrthogonalityMetrics(0.0, 0.0, 0.0)
    signed = 2.0 * float(np.sum(cross_terms)) / total_energy
    return OrthogonalityMetrics(
        signed_index=float(signed),
        absolute_index=float(abs(signed)),
        pairwise_absolute_index=float(2.0 * np.sum(np.abs(cross_terms)) / total_energy),
    )


def calculate_orthogonality_index(
    imfs: np.ndarray,
    original_signal: Optional[np.ndarray] = None,
) -> float:
    """Compatibility wrapper returning the signed orthogonality index.

    ``original_signal`` is accepted for API compatibility; the documented
    denominator is total component energy, so the source array is not needed.
    """

    if original_signal is not None:
        source = _as_finite_signal(original_signal, name="original_signal")
        if _as_imf_matrix(imfs).shape[1] != source.size:
            raise ValueError("IMF and source-signal lengths differ.")
    return calculate_orthogonality_metrics(imfs).signed_index


def calculate_adjacent_imf_correlations(imfs: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlations for adjacent *physical* IMFs."""

    modes = _as_imf_matrix(imfs)
    if modes.shape[0] < 2:
        return np.empty(0, dtype=float)
    correlations = []
    for i in range(modes.shape[0] - 1):
        a = modes[i] - float(np.mean(modes[i]))
        b = modes[i + 1] - float(np.mean(modes[i + 1]))
        denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
        correlations.append(
            0.0 if denom <= np.finfo(float).tiny else abs(float(np.sum(a * b) / denom))
        )
    return np.asarray(correlations, dtype=float)


def calculate_imf_correlation_matrix(imfs: np.ndarray) -> np.ndarray:
    """Absolute physical-IMF correlation matrix for Stage 1 heatmaps."""

    modes = _as_imf_matrix(imfs)
    matrix = np.eye(modes.shape[0], dtype=float)
    for i in range(modes.shape[0]):
        for j in range(i + 1, modes.shape[0]):
            a = modes[i] - float(np.mean(modes[i]))
            b = modes[j] - float(np.mean(modes[j]))
            denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
            corr = 0.0 if denom <= np.finfo(float).tiny else abs(float(np.sum(a * b) / denom))
            matrix[i, j] = matrix[j, i] = corr
    return matrix


def calculate_adjacent_imf_correlation(imfs: np.ndarray) -> Tuple[float, float]:
    """Compatibility metric for a legacy ``IMFs + residual`` matrix.

    The final row is excluded for compatibility with the historical return
    contract.  Explicit Stage 1 code should instead pass ``result.imfs`` to
    :func:`calculate_adjacent_imf_correlations`.
    """

    components = _as_imf_matrix(imfs)
    physical = components[:-1]
    if physical.shape[0] < 2:
        return 0.0, 0.0
    corrs = calculate_adjacent_imf_correlations(physical)
    return float(np.mean(corrs)), float(np.max(corrs))


def _normalised_welch_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(float(fs)) or float(fs) <= 0:
        raise ValueError(f"Sampling rate must be finite and positive, got {fs}.")
    nperseg = min(int(signal.size), 1024)
    freqs, psd = scipy.signal.welch(signal, fs=float(fs), nperseg=nperseg)
    psd = np.maximum(np.asarray(psd, dtype=float), 0.0)
    total = float(np.sum(psd))
    if total <= np.finfo(float).tiny:
        return np.asarray(freqs, dtype=float), np.zeros_like(psd)
    return np.asarray(freqs, dtype=float), psd / total


def calculate_adjacent_spectral_overlaps(imfs: np.ndarray, fs: float) -> np.ndarray:
    """Intersection of adjacent physical-IMF normalised Welch spectra."""

    modes = _as_imf_matrix(imfs)
    if modes.shape[0] < 2:
        return np.empty(0, dtype=float)
    spectra = [_normalised_welch_psd(mode, fs)[1] for mode in modes]
    overlaps = [float(np.sum(np.minimum(spectra[i], spectra[i + 1]))) for i in range(len(spectra) - 1)]
    return np.clip(np.asarray(overlaps, dtype=float), 0.0, 1.0)


def calculate_spectral_overlap(imfs: np.ndarray, fs: float) -> float:
    """Compatibility mean overlap for a legacy ``IMFs + residual`` matrix."""

    components = _as_imf_matrix(imfs)
    physical = components[:-1]
    if physical.shape[0] < 2:
        return 0.0
    overlaps = calculate_adjacent_spectral_overlaps(physical, fs)
    return float(np.mean(overlaps)) if overlaps.size else 0.0


def calculate_imf_metrics(imfs: np.ndarray, fs: float) -> List[IMFMetrics]:
    """Calculate energy, spectral centroid, bandwidth, and entropy per IMF."""

    modes = _as_imf_matrix(imfs)
    energies = np.sum(modes**2, axis=1)
    total_energy = float(np.sum(energies))
    if total_energy <= np.finfo(float).tiny:
        energy_pct = np.zeros(modes.shape[0], dtype=float)
    else:
        energy_pct = energies / total_energy * 100.0

    rows: List[IMFMetrics] = []
    for index, mode in enumerate(modes):
        freqs, probability = _normalised_welch_psd(mode, fs)
        if float(np.sum(probability)) <= np.finfo(float).tiny:
            centre = bandwidth = entropy = 0.0
        else:
            centre = float(np.sum(freqs * probability))
            bandwidth = float(np.sqrt(np.sum(((freqs - centre) ** 2) * probability)))
            positive = probability > 0
            raw_entropy = -float(np.sum(probability[positive] * np.log2(probability[positive])))
            max_entropy = float(np.log2(probability.size)) if probability.size > 1 else 0.0
            entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        rows.append(
            IMFMetrics(
                imf_index=index + 1,
                energy_percentage=float(energy_pct[index]),
                centre_frequency_hz=centre,
                bandwidth_hz=bandwidth,
                spectral_entropy=float(np.clip(entropy, 0.0, 1.0)),
                rms=float(np.sqrt(np.mean(mode**2))),
            )
        )
    return rows


def frequency_ordering_score_from_centres(centre_frequencies_hz: Sequence[float]) -> float:
    """Fraction of adjacent centres that follow descending EMD order.

    This direct adjacent-pair score is mathematically interpretable: 1 means
    every IMF centre frequency is non-increasing with IMF index and 0 means
    every adjacent pair is inverted.
    """

    centres = np.asarray(centre_frequencies_hz, dtype=float)
    if centres.ndim != 1 or centres.size == 0 or not np.all(np.isfinite(centres)):
        raise ValueError("centre_frequencies_hz must be a non-empty finite vector.")
    if centres.size == 1:
        return 1.0
    return float(np.mean(centres[:-1] >= centres[1:]))


def calculate_frequency_ordering_score(imfs: np.ndarray, fs: float) -> float:
    metrics = calculate_imf_metrics(imfs, fs)
    return frequency_ordering_score_from_centres([row.centre_frequency_hz for row in metrics])


def calculate_decomposition_metrics(
    original_signal: np.ndarray,
    decomposition: Union[CEEMDANResult, np.ndarray],
    fs: float,
    *,
    residual: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return the complete Stage 1 metric bundle from an explicit decomposition.

    Passing a raw matrix means that matrix contains physical IMFs only; provide
    ``residual=...`` explicitly.  This avoids silently interpreting a final row.
    """

    residual_arr: Optional[np.ndarray]
    runtime: Optional[float]
    if isinstance(decomposition, CEEMDANResult):
        physical_imfs = decomposition.imfs
        residual_arr = decomposition.residual
        runtime = decomposition.runtime_seconds
    else:
        physical_imfs = _as_imf_matrix(decomposition)
        residual_arr = residual
        runtime = None

    source = _as_finite_signal(original_signal, name="original_signal")
    physical_imfs = _as_imf_matrix(physical_imfs)
    if physical_imfs.shape[1] != source.size:
        raise ValueError("IMF and source-signal lengths differ.")
    if residual_arr is not None:
        residual_arr = _as_finite_signal(residual_arr, name="residual")
        if residual_arr.size != source.size:
            raise ValueError("Residual and source-signal lengths differ.")
        orthogonal_components = np.vstack((physical_imfs, residual_arr))
    else:
        orthogonal_components = physical_imfs

    correlations = calculate_adjacent_imf_correlations(physical_imfs)
    overlaps = calculate_adjacent_spectral_overlaps(physical_imfs, fs)
    orthogonality = calculate_orthogonality_metrics(orthogonal_components)
    per_imf = calculate_imf_metrics(physical_imfs, fs)

    metrics: Dict[str, Any] = {
        "number_of_imfs": int(physical_imfs.shape[0]),
        "reconstruction_nrmse": calculate_reconstruction_nrmse(
            source, physical_imfs, residual_arr
        ),
        "signed_orthogonality_index": orthogonality.signed_index,
        "absolute_orthogonality_index": orthogonality.absolute_index,
        "pairwise_absolute_orthogonality_index": orthogonality.pairwise_absolute_index,
        "mean_adjacent_imf_correlation": float(np.mean(correlations)) if correlations.size else 0.0,
        "maximum_adjacent_imf_correlation": float(np.max(correlations)) if correlations.size else 0.0,
        "adjacent_imf_correlations": correlations.tolist(),
        "spectral_overlap": float(np.mean(overlaps)) if overlaps.size else 0.0,
        "maximum_adjacent_spectral_overlap": float(np.max(overlaps)) if overlaps.size else 0.0,
        "adjacent_spectral_overlaps": overlaps.tolist(),
        "frequency_ordering_score": frequency_ordering_score_from_centres(
            [row.centre_frequency_hz for row in per_imf]
        ),
        "imf_metrics": [row.as_dict() for row in per_imf],
    }
    if runtime is not None:
        metrics["ceemdan_runtime_seconds"] = float(runtime)
    return metrics


def _physical_imfs_from_seed_item(item: Union[CEEMDANResult, np.ndarray]) -> np.ndarray:
    if isinstance(item, CEEMDANResult):
        return _as_imf_matrix(item.imfs)
    # A raw array supplied to this structural API is explicitly a physical-IMF
    # matrix.  No final-row residual inference is performed.
    return _as_imf_matrix(item)


def _absolute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(aa**2) * np.sum(bb**2)))
    if denom <= np.finfo(float).tiny:
        return 0.0
    return abs(float(np.sum(aa * bb) / denom))


def _match_seed_decompositions(
    first: np.ndarray,
    second: np.ndarray,
    fs: float,
) -> Dict[str, Any]:
    first_metrics = calculate_imf_metrics(first, fs)
    second_metrics = calculate_imf_metrics(second, fs)
    first_cf = np.asarray([m.centre_frequency_hz for m in first_metrics], dtype=float)
    second_cf = np.asarray([m.centre_frequency_hz for m in second_metrics], dtype=float)
    first_energy = np.asarray([m.energy_percentage for m in first_metrics], dtype=float) / 100.0
    second_energy = np.asarray([m.energy_percentage for m in second_metrics], dtype=float) / 100.0

    correlation = np.empty((first.shape[0], second.shape[0]), dtype=float)
    frequency_cost = np.empty_like(correlation)
    eps = max(float(fs) * 1e-12, np.finfo(float).eps)
    for i in range(first.shape[0]):
        for j in range(second.shape[0]):
            correlation[i, j] = _absolute_correlation(first[i], second[j])
            log_ratio = abs(float(np.log((first_cf[i] + eps) / (second_cf[j] + eps))))
            frequency_cost[i, j] = 1.0 - np.exp(-log_ratio)

    # Correlation dominates matching, while spectral proximity prevents a
    # phase-shifted neighbouring mode from being chosen solely by waveform fit.
    cost = 0.65 * (1.0 - correlation) + 0.35 * frequency_cost
    rows, cols = linear_sum_assignment(cost)
    matched_correlations = correlation[rows, cols]
    matched_log_ratios = np.abs(
        np.log((first_cf[rows] + eps) / (second_cf[cols] + eps))
    )

    energy_l1 = float(np.sum(np.abs(first_energy[rows] - second_energy[cols])))
    unmatched_first = sorted(set(range(first.shape[0])) - set(rows.tolist()))
    unmatched_second = sorted(set(range(second.shape[0])) - set(cols.tolist()))
    if unmatched_first:
        energy_l1 += float(np.sum(first_energy[unmatched_first]))
    if unmatched_second:
        energy_l1 += float(np.sum(second_energy[unmatched_second]))
    energy_l1 = min(1.0, 0.5 * energy_l1)

    return {
        "first_imf_count": int(first.shape[0]),
        "second_imf_count": int(second.shape[0]),
        "matched_pairs": [[int(i + 1), int(j + 1)] for i, j in zip(rows, cols)],
        "matched_imf_correlation_mean": float(np.mean(matched_correlations)),
        "matched_imf_correlation_minimum": float(np.min(matched_correlations)),
        "centre_frequency_log_ratio_mean": float(np.mean(matched_log_ratios)),
        "centre_frequency_instability": float(
            1.0 - np.exp(-float(np.mean(matched_log_ratios)))
        ),
        "energy_distribution_l1": energy_l1,
        "imf_count_mismatch_fraction": float(
            abs(first.shape[0] - second.shape[0]) / max(first.shape[0], second.shape[0])
        ),
    }


def calculate_seed_stability(
    decompositions: Sequence[Union[CEEMDANResult, np.ndarray]],
    fs: float,
    *,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Structurally compare CEEMDAN outputs across seeds.

    The score uses IMF centre frequencies, energy distributions, optimally
    matched IMF correlations, spectral-overlap variation, and IMF-count
    variation.  Reconstruction NRMSE is reported elsewhere and intentionally
    does not drive seed stability because valid CEEMDAN decompositions all
    reconstruct the source almost exactly.
    """

    if not decompositions:
        raise ValueError("At least one decomposition is required for seed stability.")
    physical_sets = [_physical_imfs_from_seed_item(item) for item in decompositions]
    sample_counts = {modes.shape[1] for modes in physical_sets}
    if len(sample_counts) != 1:
        raise ValueError("All seed decompositions must have the same sample count.")
    if seeds is not None and len(seeds) != len(physical_sets):
        raise ValueError("seeds length must match decompositions length.")
    seed_values = list(range(len(physical_sets))) if seeds is None else [int(s) for s in seeds]

    imf_counts = np.asarray([modes.shape[0] for modes in physical_sets], dtype=float)
    overlap_values = []
    per_seed = []
    for seed, modes in zip(seed_values, physical_sets):
        per_imf = calculate_imf_metrics(modes, fs)
        overlaps = calculate_adjacent_spectral_overlaps(modes, fs)
        overlap = float(np.mean(overlaps)) if overlaps.size else 0.0
        overlap_values.append(overlap)
        per_seed.append(
            {
                "seed": int(seed),
                "number_of_imfs": int(modes.shape[0]),
                "centre_frequencies_hz": [m.centre_frequency_hz for m in per_imf],
                "energy_percentages": [m.energy_percentage for m in per_imf],
                "spectral_overlap": overlap,
            }
        )

    comparisons = []
    for first_index, second_index in combinations(range(len(physical_sets)), 2):
        comparison = _match_seed_decompositions(
            physical_sets[first_index], physical_sets[second_index], fs
        )
        comparison["first_seed"] = int(seed_values[first_index])
        comparison["second_seed"] = int(seed_values[second_index])
        comparisons.append(comparison)

    if comparisons:
        centre_instability = float(
            np.mean([c["centre_frequency_instability"] for c in comparisons])
        )
        centre_log_ratio = float(
            np.mean([c["centre_frequency_log_ratio_mean"] for c in comparisons])
        )
        energy_instability = float(
            np.mean([c["energy_distribution_l1"] for c in comparisons])
        )
        matched_corr = float(
            np.mean([c["matched_imf_correlation_mean"] for c in comparisons])
        )
        matched_corr_min = float(
            np.min([c["matched_imf_correlation_minimum"] for c in comparisons])
        )
        count_mismatch = float(
            np.mean([c["imf_count_mismatch_fraction"] for c in comparisons])
        )
    else:
        centre_instability = centre_log_ratio = energy_instability = count_mismatch = 0.0
        matched_corr = matched_corr_min = 1.0

    overlap_std = float(np.std(overlap_values))
    structural_components = {
        "centre_frequency": centre_instability,
        "energy_distribution": energy_instability,
        "matched_imf_correlation": 1.0 - matched_corr,
        "spectral_overlap_variation": min(1.0, overlap_std),
        "imf_count_variation": count_mismatch,
    }
    instability_score = float(np.mean(list(structural_components.values())))

    return {
        "n_seeds": int(len(physical_sets)),
        "seeds": [int(seed) for seed in seed_values],
        "imf_counts": [int(value) for value in imf_counts],
        "imf_count_standard_deviation": float(np.std(imf_counts)),
        "imf_count_range": int(np.max(imf_counts) - np.min(imf_counts)),
        "imf_count_mismatch_fraction": count_mismatch,
        "centre_frequency_log_ratio_mean": centre_log_ratio,
        "centre_frequency_instability": centre_instability,
        "energy_distribution_l1": energy_instability,
        "matched_imf_correlation_mean": matched_corr,
        "matched_imf_correlation_minimum": matched_corr_min,
        "spectral_overlap_mean": float(np.mean(overlap_values)),
        "spectral_overlap_standard_deviation": overlap_std,
        "structural_components": structural_components,
        "instability_score": instability_score,
        "per_seed": per_seed,
        "pairwise_comparisons": comparisons,
    }


def calculate_composite_cutoff_score(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
) -> float:
    """Legacy three-factor decomposition badness score (lower is better)."""

    mean_corr, max_corr = calculate_adjacent_imf_correlation(imfs)
    del mean_corr  # maximum correlation is the historical objective component
    spectral_overlap = calculate_spectral_overlap(imfs, fs)
    oi = calculate_orthogonality_index(imfs, original_signal)
    return float(0.35 * spectral_overlap + 0.35 * max_corr + 0.30 * abs(oi))
