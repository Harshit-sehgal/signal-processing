"""Canonical Stage 1--4 processing for one machining-vibration recording.

This module is deliberately an integration layer.  Scientific calculations
live in their stage modules; :func:`process_recording` preserves the contracts
between them, including the explicit CEEMDAN residual and the single Stage 1
physical-unit scale factor.  Stage 4 ends at transparent feature extraction --
this module never creates labels, probabilities, or decisions.
"""

from __future__ import annotations

import math
import time as clock
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

from pg_amcd.decomposition import (
    CEEMDANResult,
    calculate_decomposition_metrics,
    calculate_imf_metrics,
    decompose_ceemdan,
)
from pg_amcd.denoising import wavelet_denoise_with_diagnostics
from pg_amcd.features import (
    FEATURE_SCHEMA_VERSION,
    WindowFeatureRecord,
    aggregate_feature_results,
    extract_sliding_window_features,
    feature_schema,
    summarize_feature_repeatability,
)
from pg_amcd.models import (
    PipelineResult,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    WindowResult,
)
from pg_amcd.optimization import optimize_cutoff
from pg_amcd.preprocessing import preprocess_signal_result
from pg_amcd.segmentation import select_max_energy_segment_indices
from pg_amcd.synthetic import generate_synthetic_signal
from pg_amcd.weighting import (
    PhysicsMetadata,
    analyze_physics_guided_weighting,
    calculate_maiw_weights,
    reconstruct_gated_signal,
    restore_physical_units,
    summarize_matched_gate_stability,
    validate_physics_metadata,
)


def _mapping_section(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    raw = config.get(name, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} configuration must be a mapping.")
    return dict(raw)


def _finite_positive(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {result}.")
    return result


def _validated_input_arrays(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate direct-array callers with the same invariants as file loading."""

    if np.iscomplexobj(time_values) or np.iscomplexobj(signal_values):
        raise ValueError("Time and signal arrays must be real-valued.")
    time_array = np.asarray(time_values, dtype=float)
    signal_array = np.asarray(signal_values, dtype=float)
    if time_array.ndim != 1 or signal_array.ndim != 1:
        raise ValueError("Time and signal arrays must be one-dimensional.")
    if time_array.size != signal_array.size:
        raise ValueError("Time and signal arrays must have identical lengths.")
    if signal_array.size < 3:
        raise ValueError("A recording must contain at least three samples.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(signal_array)):
        raise ValueError("Time and signal arrays must contain only finite values.")
    if float(np.std(signal_array)) <= np.finfo(float).tiny:
        raise ValueError("The recording is constant or numerically flat.")

    configured_fs = _finite_positive(config.get("sampling_rate"), "sampling_rate")
    differences = np.diff(time_array)
    if np.any(differences <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    median_step = float(np.median(differences))
    estimated_fs = 1.0 / median_step
    validation = _mapping_section(config, "validation")
    sampling_tolerance = float(validation.get("sampling_rate_tolerance", 0.05))
    jitter_tolerance = float(validation.get("timestamp_jitter_tolerance", 0.05))
    if not 0.0 <= sampling_tolerance < 1.0:
        raise ValueError("validation.sampling_rate_tolerance must be in [0, 1).")
    if not 0.0 <= jitter_tolerance < 1.0:
        raise ValueError("validation.timestamp_jitter_tolerance must be in [0, 1).")
    sampling_error = abs(estimated_fs - configured_fs) / configured_fs
    if sampling_error > sampling_tolerance:
        raise ValueError(
            "Timestamp-derived sampling rate differs from configured sampling_rate: "
            f"estimated={estimated_fs:.9g} Hz, configured={configured_fs:.9g} Hz, "
            f"relative_error={sampling_error:.3g}."
        )
    relative_jitter = float(np.max(np.abs(differences - median_step)) / median_step)
    if relative_jitter > jitter_tolerance:
        raise ValueError(
            "Timestamp jitter exceeds validation.timestamp_jitter_tolerance: "
            f"relative_jitter={relative_jitter:.3g}."
        )
    minimum_duration = float(validation.get("minimum_duration_seconds", 0.0))
    if not math.isfinite(minimum_duration) or minimum_duration < 0:
        raise ValueError("validation.minimum_duration_seconds must be finite and non-negative.")
    # Sample-count duration follows the acquisition convention used by the
    # file validator: N samples at fs represent N/fs seconds of acquisition.
    if signal_array.size / configured_fs < minimum_duration:
        raise ValueError(
            f"Recording duration {signal_array.size / configured_fs:.6g} s is shorter "
            f"than the required {minimum_duration:.6g} s."
        )
    return time_array, signal_array, configured_fs


def _resolved_preprocessing(config: Mapping[str, Any], fs: float) -> Dict[str, Any]:
    pre = _mapping_section(config, "preprocessing")
    lowpass_raw = pre.get(
        "low_pass_cutoff_hz",
        pre.get("lowpass_cutoff_hz", pre.get("upper_cutoff_hz", pre.get("high_cutoff_hz"))),
    )
    lowpass = min(4000.0, fs / 2.0 - 10.0) if lowpass_raw is None else float(lowpass_raw)
    if not math.isfinite(lowpass) or lowpass <= 0 or lowpass >= fs / 2.0:
        raise ValueError(
            "preprocessing.low_pass_cutoff_hz must satisfy 0 < cutoff < Nyquist; "
            f"resolved value was {lowpass}."
        )
    return {
        "lowpass_cutoff_hz": lowpass,
        "filter_order": pre.get("filter_order", 3),
        "detrend_type": str(pre.get("detrend_type", "linear")),
        "detrend_before_filter": bool(pre.get("detrend_before_filter", False)),
        "scale_percentile": float(pre.get("scale_percentile", 99.5)),
        "padtype": pre.get("padtype", "odd"),
        "padlen": pre.get("padlen"),
        "lowpass_resolution": "automatic min(4000 Hz, Nyquist - 10 Hz)"
        if lowpass_raw is None
        else "configured",
    }


def _preprocess(
    signal: np.ndarray,
    highpass_cutoff_hz: float,
    fs: float,
    settings: Mapping[str, Any],
):
    return preprocess_signal_result(
        signal,
        highpass_cutoff_hz,
        float(settings["lowpass_cutoff_hz"]),
        fs,
        order=settings["filter_order"],
        detrend_type=str(settings["detrend_type"]),
        detrend_before_filter=bool(settings["detrend_before_filter"]),
        scale_percentile=float(settings["scale_percentile"]),
        padtype=settings["padtype"],
        padlen=settings["padlen"],
    )


def _ceemdan_options(ceemdan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "parallel": bool(ceemdan.get("parallel", True)),
        "processes": ceemdan.get("processes"),
        "max_imf": int(ceemdan.get("max_imf", -1)),
        "noise_scale": float(ceemdan.get("noise_scale", 1.0)),
        "noise_kind": str(ceemdan.get("noise_kind", "normal")),
        "range_threshold": float(ceemdan.get("range_threshold", 0.01)),
        "total_power_threshold": float(ceemdan.get("total_power_threshold", 0.05)),
        "beta_progress": bool(ceemdan.get("beta_progress", True)),
        "residual_rtol": float(ceemdan.get("residual_rtol", 1e-7)),
        "residual_atol": float(ceemdan.get("residual_atol", 1e-10)),
    }


def _decompose(
    values: np.ndarray,
    ceemdan: Mapping[str, Any],
    *,
    seed: Optional[int] = None,
    trials: Optional[int] = None,
    sifting_iterations: Optional[int] = None,
) -> CEEMDANResult:
    return decompose_ceemdan(
        values,
        int(ceemdan.get("trials", 300) if trials is None else trials),
        float(ceemdan.get("epsilon", 0.02)),
        int(ceemdan.get("noise_seed", 42) if seed is None else seed),
        int(
            ceemdan.get("sifting_iterations", 16)
            if sifting_iterations is None
            else sifting_iterations
        ),
        **_ceemdan_options(ceemdan),
    )


def _safe_absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return abs(float(np.dot(left_centered, right_centered) / denominator))


def _legacy_weighting_metrics(
    source: np.ndarray,
    weighted: np.ndarray,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
    runtime_seconds: float,
) -> Dict[str, Any]:
    frequencies, source_psd = scipy.signal.welch(source, fs=fs, nperseg=min(source.size, 1024))
    _, weighted_psd = scipy.signal.welch(weighted, fs=fs, nperseg=min(weighted.size, 1024))
    chatter_low = max(0.0, chatter_center - chatter_spread)
    chatter_high = min(fs / 2.0, chatter_center + chatter_spread)
    chatter_mask = (frequencies >= chatter_low) & (frequencies <= chatter_high)
    outside_mask = ~chatter_mask

    def retention(mask: np.ndarray) -> float:
        before = float(np.sum(source_psd[mask]))
        after = float(np.sum(weighted_psd[mask]))
        return after / before if before > 0 else 0.0

    source_total = float(np.sum(source_psd))
    weighted_total = float(np.sum(weighted_psd))
    spectral_distortion = 0.0
    if source_total > 0 and weighted_total > 0:
        spectral_distortion = float(
            np.sum(np.abs(source_psd / source_total - weighted_psd / weighted_total))
        )
    return {
        "rms_before": float(np.sqrt(np.mean(np.square(source)))),
        "rms_after": float(np.sqrt(np.mean(np.square(weighted)))),
        "energy_before": float(np.sum(np.square(source))),
        "energy_after": float(np.sum(np.square(weighted))),
        "correlation_with_source": _safe_absolute_correlation(source, weighted),
        "chatter_band_retention": retention(chatter_mask),
        "spindle_harmonic_attenuation": None,
        "tooth_harmonic_attenuation": None,
        "out_of_band_attenuation": 1.0 - retention(outside_mask),
        "spectral_distortion": spectral_distortion,
        "reconstruction_runtime_seconds": float(runtime_seconds),
    }


def _legacy_weighting(
    decomposition: CEEMDANResult,
    source: np.ndarray,
    fs: float,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, list[Dict[str, Any]], Dict[str, Any]]:
    started = clock.perf_counter()
    weights, correlation, energy, kurtosis, proximity = calculate_maiw_weights(
        decomposition.components, source, fs, config
    )
    weighted = reconstruct_gated_signal(decomposition.components, weights)
    per_imf = calculate_imf_metrics(decomposition.imfs, fs)
    kurtosis_total = float(np.sum(np.maximum(kurtosis, 0.0)))
    rows: list[Dict[str, Any]] = []
    for index, metric in enumerate(per_imf):
        rows.append(
            {
                "imf_index": index + 1,
                "zero_based_imf_index": index,
                "correlation": float(correlation[index]),
                "relative_energy": float(energy[index]),
                "kurtosis": float(kurtosis[index]),
                "kurtosis_score": (
                    float(max(kurtosis[index], 0.0) / kurtosis_total) if kurtosis_total > 0 else 0.0
                ),
                "kurtosis_normalized": (
                    float(max(kurtosis[index], 0.0) / kurtosis_total) if kurtosis_total > 0 else 0.0
                ),
                # The historical fourth MAIW indicator is a Gaussian centre-
                # frequency proximity, not measured chatter-band energy.
                "chatter_band_energy_ratio": None,
                "spindle_harmonic_energy_ratio": None,
                "tooth_harmonic_energy_ratio": None,
                "forced_harmonic_energy_ratio": None,
                "frequency_proximity": float(proximity[index]),
                "centre_frequency_hz": float(metric.centre_frequency_hz),
                "bandwidth_hz": float(metric.bandwidth_hz),
                "spectral_entropy": float(metric.spectral_entropy),
                "gate": float(weights[index]),
                "weighting_method": "legacy_maiw_baseline",
            }
        )
    maiw = _mapping_section(config, "maiw")
    metrics = _legacy_weighting_metrics(
        source,
        weighted,
        fs,
        float(maiw.get("chatter_band_center", 1250.0)),
        float(maiw.get("chatter_band_spread", 500.0)),
        clock.perf_counter() - started,
    )
    return weights, weighted, rows, metrics


def _gate_stability_seeds(ceemdan: Mapping[str, Any]) -> list[int]:
    configured = ceemdan.get("gate_stability_seeds", ceemdan.get("stability_seeds"))
    if configured is None:
        configured = ceemdan.get("search_seed_values")
    if configured is not None:
        if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
            raise ValueError("CEEMDAN stability seeds must be a sequence of integers.")
        seeds = [int(value) for value in configured]
    else:
        count = int(ceemdan.get("search_seeds", 1))
        base = int(ceemdan.get("noise_seed", 42))
        seeds = [base + index for index in range(max(1, count))]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("CEEMDAN gate-stability seeds must be non-empty and unique.")
    return seeds


def _gate_stability(
    source: np.ndarray,
    fs: float,
    config: Dict[str, Any],
    ceemdan: Mapping[str, Any],
    use_physics: bool,
    physics_metadata: Optional[PhysicsMetadata],
    selection_threshold: float,
) -> Tuple[Dict[str, Any], Optional[str]]:
    seeds = _gate_stability_seeds(ceemdan)
    trials = int(
        ceemdan.get(
            "gate_stability_trials",
            ceemdan.get("search_trials", ceemdan.get("trials", 300)),
        )
    )
    sifting = int(
        ceemdan.get(
            "gate_stability_sifting_iterations",
            ceemdan.get("search_sifting_iterations", ceemdan.get("sifting_iterations", 16)),
        )
    )
    vectors: list[np.ndarray] = []
    decompositions: list[np.ndarray] = []
    component_counts: list[int] = []
    for seed in seeds:
        decomposition = _decompose(
            source,
            ceemdan,
            seed=seed,
            trials=trials,
            sifting_iterations=sifting,
        )
        component_counts.append(int(decomposition.num_imfs))
        decompositions.append(np.asarray(decomposition.imfs, dtype=float))
        if use_physics:
            if physics_metadata is None:  # guarded before pipeline work
                raise RuntimeError("Validated physics metadata is unavailable.")
            result = analyze_physics_guided_weighting(
                decomposition.components,
                source,
                fs,
                physics_metadata,
                config,
                residual_policy="last_row",
                strict_config=True,
            )
            vectors.append(result.gates)
        else:
            weights, _, _, _, _ = calculate_maiw_weights(
                decomposition.components, source, fs, config
            )
            vectors.append(weights)

    summary = summarize_matched_gate_stability(
        vectors,
        decompositions,
        fs,
        selection_threshold=selection_threshold,
        seed_values=seeds,
    )
    summary.update(
        {
            "available": True,
            "seeds": seeds,
            "physical_imf_counts": component_counts,
        }
    )
    return summary, None


def _threshold_rows(result: Any) -> list[Dict[str, Any]]:
    total_energy = float(sum(float(row.input_energy) for row in result.level_diagnostics))
    rows = []
    for diagnostic in result.level_diagnostics:
        row = diagnostic.to_dict()
        row.update(
            {
                "is_detail": not bool(diagnostic.is_approximation),
                "coefficient_energy": float(diagnostic.input_energy),
                "energy": float(diagnostic.input_energy),
                "energy_ratio": (
                    float(diagnostic.input_energy / total_energy) if total_energy > 0 else 0.0
                ),
                "chatter_band_overlap_fraction": float(diagnostic.chatter_overlap_fraction),
                "chatter_overlap": float(diagnostic.chatter_overlap_fraction),
            }
        )
        rows.append(row)
    return rows


def _synthetic_wavelet_self_check(
    fs: float,
    sample_count: int,
    wavelet: Mapping[str, Any],
    chatter_center: float,
    chatter_spread: float,
    seed: int,
    rpm: Optional[float],
    tooth_count: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    synthetic_samples = min(max(256, sample_count), 2048)
    duration = synthetic_samples / fs
    _, noisy, components = generate_synthetic_signal(
        fs=fs,
        duration=duration,
        seed=seed,
        rpm=600.0 if rpm is None else float(rpm),
        tooth_count=1 if tooth_count is None else int(tooth_count),
        chatter_freq=chatter_center,
        chatter_onset=0.4 * duration,
        snr_db=float(wavelet.get("synthetic_self_check_snr_db", 12.0)),
    )
    result = wavelet_denoise_with_diagnostics(
        noisy,
        wavelet_name=str(wavelet.get("wavelet_name", "db8")),
        level=int(wavelet.get("level", 4)),
        fs=fs,
        chatter_center=chatter_center,
        chatter_spread=chatter_spread,
        band_aware=bool(wavelet.get("band_aware", True)),
        chatter_threshold_scale=float(wavelet.get("chatter_threshold_scale", 0.5)),
        noise_threshold_scale=float(wavelet.get("noise_threshold_scale", 1.4)),
        threshold_mode=str(wavelet.get("threshold_mode", "soft")),
        minimum_noise_sigma=float(
            wavelet.get("minimum_noise_sigma", wavelet.get("min_noise_sigma", 1e-6))
        ),
        clean_reference=np.asarray(components["clean"], dtype=float),
    )
    metrics = result.metrics.to_dict()
    clean_signal = np.asarray(components["clean"], dtype=float)
    noise_signal = np.asarray(noisy - clean_signal, dtype=float)
    signal_var = float(np.var(clean_signal))
    noise_var = float(np.var(noise_signal))
    input_snr_db = (
        10.0 * np.log10(signal_var / noise_var) if noise_var > 0 else float("inf")
    )
    raw_output_snr = metrics.get("synthetic_reference_snr_db")
    output_snr_db = float(raw_output_snr) if raw_output_snr is not None else float("nan")
    metrics.update(
        {
            "sample_count": int(noisy.size),
            "duration_seconds": float(duration),
            "seed": int(seed),
            "known_clean_reference_used": True,
            "applied_level": int(result.applied_level),
            "input_snr_db": float(input_snr_db),
            "output_snr_db": output_snr_db,
            "delta_snr_db": float(output_snr_db - input_snr_db),
        }
    )
    signals = {
        "clean": np.asarray(components["clean"], dtype=float),
        "noisy": np.asarray(noisy, dtype=float),
        "recovered": np.asarray(result.denoised_signal, dtype=float),
    }
    return metrics, signals


def _feature_record_payload(
    record: WindowFeatureRecord,
    segment_time: np.ndarray,
    global_offset: int,
) -> Dict[str, Any]:
    payload = record.result.to_dict(include_traces=False)
    payload.update(
        {
            "window_index": int(record.window_index),
            "local_start_index": int(record.start_index),
            "local_end_index": int(record.end_index),
            "start_index": int(global_offset + record.start_index),
            "end_index": int(global_offset + record.end_index),
            "window_start_seconds": float(segment_time[record.start_index]),
            "window_end_seconds": float(segment_time[record.end_index - 1]),
        }
    )
    return payload


def process_recording(
    time: np.ndarray,
    signal: np.ndarray,
    config: Dict[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    mode: str = "exploratory",
) -> PipelineResult:
    """Process one recording through the complete, decision-free Stage 1--4 path.

    ``use_physics_gating=true`` is a strict mode: valid ``rpm`` and
    ``tooth_count`` metadata are required and no fallback machining parameters
    are substituted.  The optional legacy MAIW baseline is used only when the
    toggle is explicitly false.
    """

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping.")
    if mode not in {"exploratory", "sliding_window"}:
        raise ValueError("mode must be either 'exploratory' or 'sliding_window'.")
    time_array, raw_signal, fs = _validated_input_arrays(time, signal, config)
    metadata_dict: Dict[str, Any] = dict(metadata or {})
    use_physics = bool(config.get("use_physics_gating", True))
    physics_metadata: Optional[PhysicsMetadata] = None
    indicator_rows: list[Dict[str, Any]]
    weighting_metrics: Dict[str, Any]
    stage_2_metadata: Dict[str, Any]
    stage_2_config: Dict[str, Any]
    if use_physics:
        # Validate before any CEEMDAN work so a missing physical contract fails
        # fast rather than consuming an expensive run with invented defaults.
        physics_metadata = validate_physics_metadata(metadata_dict)

    warnings: list[str] = []
    if not use_physics:
        warnings.append(
            "Stage 2 used the optional legacy sum-normalised MAIW baseline because "
            "use_physics_gating=false."
        )

    # ---------------------------------------------------------------- Stage 1
    stage_1_started = clock.perf_counter()
    preprocessing = _resolved_preprocessing(config, fs)
    ceemdan = _mapping_section(config, "ceemdan")
    candidate_raw = ceemdan.get("search_cutoffs", [ceemdan.get("selected_cutoff", 100.0)])
    if isinstance(candidate_raw, (str, bytes)) or not isinstance(candidate_raw, Sequence):
        raise ValueError("ceemdan.search_cutoffs must be a non-empty sequence.")
    candidate_cutoffs = [float(value) for value in candidate_raw]
    if not candidate_cutoffs:
        raise ValueError("ceemdan.search_cutoffs must be non-empty.")

    preliminary = _preprocess(raw_signal, candidate_cutoffs[0], fs, preprocessing)
    segment_points_raw = config.get("segment_points", min(10000, raw_signal.size))
    if isinstance(segment_points_raw, bool):
        raise ValueError("segment_points must be a positive integer.")
    segment_points_float = float(segment_points_raw)
    if (
        not math.isfinite(segment_points_float)
        or not segment_points_float.is_integer()
        or segment_points_float < 3
    ):
        raise ValueError("segment_points must be an integer of at least 3.")
    segment_points = min(int(segment_points_float), int(raw_signal.size))
    start_index, end_index = select_max_energy_segment_indices(
        preliminary.physical_signal, segment_points
    )
    controlled_raw_segment = raw_signal[start_index:end_index].copy()

    cutoff_result = optimize_cutoff(
        controlled_raw_segment,
        candidate_cutoffs,
        config,
        fs,
        n_seeds=int(ceemdan.get("search_seeds", 2)),
    )
    selected_cutoff = float(cutoff_result.selected_cutoff)
    # The final full preprocessing pass establishes the one authoritative scale
    # factor.  The final CEEMDAN source is the same controlled raw segment by
    # index used during cutoff search, now under the selected full-pass filter.
    final_preprocessing = _preprocess(raw_signal, selected_cutoff, fs, preprocessing)
    segment_time = time_array[start_index:end_index].copy()
    segment_physical = final_preprocessing.physical_signal[start_index:end_index].copy()
    segment_scaled = final_preprocessing.scaled_signal[start_index:end_index].copy()
    decomposition = _decompose(segment_scaled, ceemdan)
    decomposition_metrics = calculate_decomposition_metrics(segment_scaled, decomposition, fs)
    imf_metrics = [row.as_dict() for row in calculate_imf_metrics(decomposition.imfs, fs)]
    selected_search_row = next(
        row for row in cutoff_result.per_cutoff_metrics if float(row["cutoff"]) == selected_cutoff
    )
    seed_stability = dict(selected_search_row["seed_stability"])
    scale_factor = float(final_preprocessing.scale_factor)
    imfs_physical = decomposition.imfs * scale_factor
    residual_physical = decomposition.residual * scale_factor
    residual_energy = float(np.sum(np.square(decomposition.residual)))
    source_energy = float(np.sum(np.square(segment_scaled)))
    stage_1_metrics = dict(decomposition_metrics)
    stage_1_metrics.update(
        {
            "residual_verified": bool(decomposition.residual_verified),
            "residual_verification_nrmse": float(decomposition.residual_verification_nrmse),
            "residual_source": decomposition.residual_source,
            "residual_energy_percentage_of_source": (
                100.0 * residual_energy / source_energy if source_energy > 0 else 0.0
            ),
            "selected_cutoff_objective": float(selected_search_row["final_score"]),
            "gap_to_second_best": float(cutoff_result.gap_to_second_best),
            "second_best_score": float(cutoff_result.second_best_score),
            "seed_consistency": (
                float(cutoff_result.seed_consistency)
                if cutoff_result.seed_consistency is not None
                else None
            ),
            "per_seed_best_cutoff": dict(cutoff_result.per_seed_best),
            "controlled_segment_start_index": int(start_index),
            "controlled_segment_end_index": int(end_index),
            "controlled_segment_samples": int(end_index - start_index),
            "preprocessing_parameters": final_preprocessing.parameters.as_dict(),
            "single_physical_scale_factor": scale_factor,
            "seed_stability_metric": "structural IMF stability (not reconstruction NRMSE variance)",
        }
    )
    stage_1_runtime = clock.perf_counter() - stage_1_started
    stage_1 = Stage1Output(
        time=time_array.copy(),
        raw_signal=raw_signal.copy(),
        preprocessed_physical=final_preprocessing.physical_signal.copy(),
        preprocessed_scaled=final_preprocessing.scaled_signal.copy(),
        segment_time=segment_time,
        segment_raw=controlled_raw_segment,
        segment_physical=segment_physical,
        segment_scaled=segment_scaled,
        imfs_scaled=decomposition.imfs.copy(),
        residual_scaled=decomposition.residual.copy(),
        imfs_physical=imfs_physical,
        residual_physical=residual_physical,
        start_index=int(start_index),
        end_index=int(end_index),
        sampling_rate=fs,
        scale_factor=scale_factor,
        selected_cutoff=selected_cutoff,
        random_seed=int(ceemdan.get("noise_seed", 42)),
        ceemdan_parameters={
            **decomposition.parameters,
            "residual_policy": "explicit_verified_final_row",
            "residual_included_in_later_stage_gates": False,
        },
        cutoff_search=list(cutoff_result.per_cutoff_metrics),
        imf_metrics=imf_metrics,
        seed_stability=seed_stability,
        metrics=stage_1_metrics,
        runtime_seconds=float(stage_1_runtime),
    )

    # ---------------------------------------------------------------- Stage 2
    stage_2_started = clock.perf_counter()
    maiw = _mapping_section(config, "maiw")
    chatter_center = _finite_positive(
        maiw.get("chatter_band_center", 1250.0), "maiw.chatter_band_center"
    )
    chatter_spread = _finite_positive(
        maiw.get("chatter_band_spread", 500.0), "maiw.chatter_band_spread"
    )
    physics_gating = _mapping_section(config, "physics_gating")
    selection_threshold = float(
        physics_gating.get(
            "selection_threshold",
            0.5 if use_physics else maiw.get("selection_threshold", 0.05),
        )
    )
    if not 0.0 <= selection_threshold <= 1.0:
        raise ValueError("Stage 2 selection_threshold must be in [0, 1].")
    if bool(physics_gating.get("include_residual", False)):
        warnings.append(
            "physics_gating.include_residual=true was ignored: the verified CEEMDAN "
            "residual is never treated as or gated like a physical IMF."
        )

    if use_physics:
        if physics_metadata is None:  # validated above, retained for type narrowing
            raise RuntimeError("Validated physics metadata is unavailable.")
        weighting = analyze_physics_guided_weighting(
            decomposition.components,
            segment_scaled,
            fs,
            physics_metadata,
            config,
            residual_policy="last_row",
            strict_config=True,
        )
        gates = weighting.gates.copy()
        weighted_scaled = weighting.reconstructed_scaled.copy()
        indicator_rows = []
        for indicator in weighting.indicators:
            row: Dict[str, Any] = dict(indicator.to_dict())
            zero_based_index = int(row["imf_index"])
            row.update(
                {
                    "zero_based_imf_index": zero_based_index,
                    "imf_index": zero_based_index + 1,
                    "kurtosis_normalized": float(row["kurtosis_score"]),
                    "weighting_method": "physics_guided_independent_gates",
                }
            )
            indicator_rows.append(row)
        weighting_metrics = dict(weighting.metrics.to_dict())
        stage_2_metadata = dict(physics_metadata.to_dict())
        stage_2_config = {
            **dict(weighting.coefficients),
            "method": "physics_guided_independent_gates",
            "selection_threshold": selection_threshold,
            "include_residual": False,
            "residual_policy": weighting.residual_policy,
            "chatter_band_center": chatter_center,
            "chatter_band_spread": chatter_spread,
            "harmonic_count": int(weighting.coefficients["harmonic_count"]),
        }
    else:
        gates, weighted_scaled, indicator_rows, weighting_metrics = _legacy_weighting(
            decomposition, segment_scaled, fs, config
        )
        stage_2_metadata = {
            key: metadata_dict.get(key)
            for key in (
                "rpm",
                "tooth_count",
                "stickout",
                "depth_of_cut",
                "feed_rate",
                "tool_identifier",
                "tool_id",
            )
            if key in metadata_dict
        }
        stage_2_metadata["physics_metadata_required"] = False
        stage_2_config = {
            **maiw,
            "method": "legacy_maiw_baseline",
            "selection_threshold": selection_threshold,
            "include_residual": False,
            "residual_policy": "last_row_explicitly_excluded",
            "chatter_band_center": chatter_center,
            "chatter_band_spread": chatter_spread,
            "harmonic_count": int(physics_gating.get("harmonic_count", 5)),
        }

    gate_stability, gate_warning = _gate_stability(
        segment_scaled,
        fs,
        config,
        ceemdan,
        use_physics,
        physics_metadata,
        selection_threshold,
    )
    if gate_warning is not None:
        warnings.append(gate_warning)
    weighting_metrics.update(
        {
            "method": stage_2_config["method"],
            "residual_excluded": True,
            "physical_imf_count": int(decomposition.num_imfs),
            "gate_vector_stability": gate_stability,
            "gate_sum": float(np.sum(gates)),
            "gate_normalisation": (
                "independent_not_sum_normalised"
                if use_physics
                else "legacy_sum_normalised_baseline"
            ),
        }
    )
    weighted_physical = restore_physical_units(weighted_scaled, scale_factor)
    stage_2_runtime = clock.perf_counter() - stage_2_started
    stage_2 = Stage2Output(
        indicators=indicator_rows,
        gates=gates,
        weighted_scaled=weighted_scaled,
        weighted_physical=weighted_physical,
        metadata=stage_2_metadata,
        metrics=weighting_metrics,
        config=stage_2_config,
        runtime_seconds=float(stage_2_runtime),
    )

    # ---------------------------------------------------------------- Stage 3
    stage_3_started = clock.perf_counter()
    wavelet = _mapping_section(config, "wavelet")
    wavelet_result = wavelet_denoise_with_diagnostics(
        weighted_scaled,
        wavelet_name=str(wavelet.get("wavelet_name", "db8")),
        level=int(wavelet.get("level", 4)),
        fs=fs,
        chatter_center=chatter_center,
        chatter_spread=chatter_spread,
        band_aware=bool(wavelet.get("band_aware", True)),
        chatter_threshold_scale=float(wavelet.get("chatter_threshold_scale", 0.5)),
        noise_threshold_scale=float(wavelet.get("noise_threshold_scale", 1.4)),
        threshold_mode=str(wavelet.get("threshold_mode", "soft")),
        minimum_noise_sigma=float(
            wavelet.get("minimum_noise_sigma", wavelet.get("min_noise_sigma", 1e-6))
        ),
    )
    denoised_scaled = wavelet_result.denoised_signal.copy()
    denoised_physical = restore_physical_units(denoised_scaled, scale_factor)
    synthetic_metrics, synthetic_signals = _synthetic_wavelet_self_check(
        fs,
        segment_scaled.size,
        wavelet,
        chatter_center,
        chatter_spread,
        int(ceemdan.get("noise_seed", 42)) + 10_000,
        None if physics_metadata is None else physics_metadata.rpm,
        None if physics_metadata is None else physics_metadata.tooth_count,
    )
    stage_3_metrics: Dict[str, Any] = dict(wavelet_result.metrics.to_dict())
    stage_3_metrics.update(
        {
            "resolved_level": int(wavelet_result.applied_level),
            "requested_level": int(wavelet_result.requested_level),
            "input_stage": "Stage_2 weighted reconstruction",
            "denoising_scope": "reconstruction_level",
            "single_scale_factor_restoration": scale_factor,
            "synthetic_self_check": synthetic_metrics,
        }
    )
    stage_3_config = {
        **wavelet,
        "wavelet_name": wavelet_result.wavelet_name,
        "level": int(wavelet_result.requested_level),
        "applied_level": int(wavelet_result.applied_level),
        "threshold_mode": wavelet_result.threshold_mode,
        "band_aware": bool(wavelet_result.band_aware),
        "chatter_center": chatter_center,
        "chatter_spread": chatter_spread,
        "chatter_band_center": chatter_center,
        "chatter_band_spread": chatter_spread,
        "minimum_noise_sigma": float(
            wavelet.get("minimum_noise_sigma", wavelet.get("min_noise_sigma", 1e-6))
        ),
    }
    coefficients = [wavelet_result.approximation_coefficients.copy()]
    coefficients.extend(
        coefficient.copy() for coefficient in wavelet_result.detail_coefficients.values()
    )
    stage_3_runtime = clock.perf_counter() - stage_3_started
    stage_3 = Stage3Output(
        coefficients=coefficients,
        threshold_rows=_threshold_rows(wavelet_result),
        denoised_scaled=denoised_scaled,
        denoised_physical=denoised_physical,
        metrics=stage_3_metrics,
        config=stage_3_config,
        runtime_seconds=float(stage_3_runtime),
        synthetic_signals=synthetic_signals,
    )

    # ---------------------------------------------------------------- Stage 4
    stage_4_started = clock.perf_counter()
    features_config = _mapping_section(config, "features")
    requested_window_seconds = float(features_config.get("window_seconds", 1.0))
    effective_window_seconds = requested_window_seconds
    segment_duration = segment_scaled.size / fs
    if requested_window_seconds > segment_duration:
        effective_window_seconds = segment_duration
        warnings.append(
            "features.window_seconds exceeded the controlled segment and was capped "
            f"from {requested_window_seconds:g} s to {effective_window_seconds:g} s."
        )
    overlap_ratio = float(features_config.get("overlap_ratio", 0.75))
    rpm_for_features: Optional[float]
    tooth_count_for_features: Optional[int]
    if physics_metadata is not None:
        rpm_for_features = physics_metadata.rpm
        tooth_count_for_features = physics_metadata.tooth_count
    else:
        rpm_for_features = metadata_dict.get("rpm")
        tooth_count_for_features = metadata_dict.get("tooth_count")
    band_ranges = features_config.get(
        "band_energy_ranges_hz", features_config.get("band_energy_ranges")
    )
    physical_components = np.vstack((imfs_physical, residual_physical))
    feature_call_kwargs: Dict[str, Any] = {
        "window_seconds": effective_window_seconds,
        "overlap_ratio": overlap_ratio,
        "imf_gates": gates,
        "residual_last_row": True,
        "selected_gate_threshold": selection_threshold,
        "wavelet_name": wavelet_result.wavelet_name,
        "wavelet_level": int(wavelet_result.applied_level),
        "chatter_center": chatter_center,
        "chatter_spread": chatter_spread,
        "harmonic_count": int(features_config.get("harmonic_count", 5)),
        "harmonic_tolerance_hz": float(features_config.get("harmonic_tolerance_hz", 15.0)),
        "sideband_tolerance_hz": float(features_config.get("sideband_tolerance_hz", 10.0)),
        "band_energy_ranges": band_ranges,
        "strict_chatter_band": True,
    }
    feature_records = extract_sliding_window_features(
        controlled_raw_segment,
        segment_physical,
        denoised_physical,
        physical_components,
        fs,
        rpm_for_features,
        tooth_count_for_features,
        **feature_call_kwargs,
    )
    repeated_feature_records = extract_sliding_window_features(
        controlled_raw_segment,
        segment_physical,
        denoised_physical,
        physical_components,
        fs,
        rpm_for_features,
        tooth_count_for_features,
        **feature_call_kwargs,
    )
    repeatability = summarize_feature_repeatability(feature_records, repeated_feature_records)
    recording_id = str(metadata_dict.get("recording_id") or "recording")
    aggregate = aggregate_feature_results(
        feature_records,
        recording_ids=[recording_id] * len(feature_records),
    )
    stage_4_rows: list[Dict[str, Any]] = []
    stage_4_records: list[Dict[str, Any]] = []
    total_defined = 0
    total_undefined = 0
    per_window_quality = []
    for aggregate_row, record in zip(aggregate.rows, feature_records):
        row = dict(aggregate_row)
        row.update(
            {
                "window_index": int(record.window_index),
                "start_index": int(start_index + record.start_index),
                "end_index": int(start_index + record.end_index),
                "window_start_seconds": float(segment_time[record.start_index]),
                "window_end_seconds": float(segment_time[record.end_index - 1]),
            }
        )
        stage_4_rows.append(row)
        stage_4_records.append(_feature_record_payload(record, segment_time, start_index))
        total_defined += int(record.result.quality["defined_feature_count"])
        total_undefined += int(record.result.quality["undefined_feature_count"])
        per_window_quality.append(
            {"window_index": int(record.window_index), **dict(record.result.quality)}
        )
    full_schema = feature_schema()
    full_schema["features"] = aggregate.schema["features"]
    full_schema["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    stage_4_quality = {
        "summary": aggregate.summary,
        "missingness": aggregate.missingness,
        "correlations": aggregate.correlations,
        "per_window": per_window_quality,
        "null_policy": (
            "Undefined canonical feature values remain null in feature_rows and "
            "feature_records, with per-feature reasons in feature_records."
        ),
        "legacy_window_result_policy": (
            "Only compatibility WindowResult.features replaces undefined values with 0.0."
        ),
    }
    stage_4_metrics = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "window_count": len(feature_records),
        "defined_feature_values": int(total_defined),
        "undefined_feature_values": int(total_undefined),
        "defined_fraction": (total_defined / max(1, total_defined + total_undefined)),
        "all_defined_values_finite": bool(
            all(record.result.quality["all_defined_values_finite"] for record in feature_records)
        ),
        "physics_metadata_valid_for_all_windows": bool(
            all(record.result.quality["physics_metadata_valid"] for record in feature_records)
        ),
        "repeat_extraction_stability": repeatability,
        "stage_scope": "feature_extraction_only",
        "feature_selection_performed": False,
        "model_training_performed": False,
        "probabilities_generated": False,
        "decisions_generated": False,
    }
    stage_4_config = {
        **features_config,
        "requested_window_seconds": requested_window_seconds,
        "window_seconds": effective_window_seconds,
        "overlap_ratio": overlap_ratio,
        "selected_gate_threshold": selection_threshold,
        "residual_handling": "explicit_last_row_excluded_from_physical_imf_features",
        "mode": mode,
    }
    stage_4_runtime = clock.perf_counter() - stage_4_started
    stage_4 = Stage4Output(
        feature_rows=stage_4_rows,
        feature_records=stage_4_records,
        feature_schema=full_schema,
        feature_quality=stage_4_quality,
        metrics=stage_4_metrics,
        config=stage_4_config,
        runtime_seconds=float(stage_4_runtime),
    )

    # Compatibility windows intentionally expose finite numeric features for
    # older evaluation code, while the canonical Stage 4 records above retain
    # every null and its reason.  No detector output is inferred here.
    window_results: list[WindowResult] = []
    selected_imfs = [int(index) for index, gate in enumerate(gates) if gate >= selection_threshold]
    for record in feature_records:
        local_start = int(record.start_index)
        local_end = int(record.end_index)
        compatibility_features = record.result.finite_values(undefined_fill=0.0)
        compatibility_features.update(
            {
                "mmi": float(stage_1_metrics["mean_adjacent_imf_correlation"]),
                "oi": float(stage_1_metrics["signed_orthogonality_index"]),
                "nrmse": float(stage_1_metrics["reconstruction_nrmse"]),
            }
        )
        window_results.append(
            WindowResult(
                time_segment=segment_time[local_start:local_end].copy(),
                start_time=float(segment_time[local_start]),
                end_time=float(segment_time[local_end - 1]),
                start_idx=int(start_index + local_start),
                end_idx=int(start_index + local_end),
                features=compatibility_features,
                chatter_probability=float("nan"),
                predicted_label="not_evaluated",
                selected_imfs=selected_imfs.copy(),
                confidence=float("nan"),
                imfs=decomposition.components[:, local_start:local_end].copy(),
                maiw_reconstructed=weighted_physical[local_start:local_end].copy(),
                denoised_clean=denoised_physical[local_start:local_end].copy(),
                gates=gates.copy(),
            )
        )

    selected_parameters = {
        "cutoff_frequency": selected_cutoff,
        "highpass_cutoff_hz": selected_cutoff,
        "lowpass_cutoff_hz": float(preprocessing["lowpass_cutoff_hz"]),
        "lowpass_resolution": preprocessing["lowpass_resolution"],
        "cutoff_search": list(cutoff_result.per_cutoff_metrics),
        "ceemdan_trials": int(ceemdan.get("trials", 300)),
        "ceemdan_epsilon": float(ceemdan.get("epsilon", 0.02)),
        "ceemdan_noise_seed": int(ceemdan.get("noise_seed", 42)),
        "sifting_iterations": int(ceemdan.get("sifting_iterations", 16)),
        "wavelet_name": wavelet_result.wavelet_name,
        "wavelet_level": int(wavelet_result.applied_level),
        "use_physics_gating": use_physics,
        "stage_2_method": stage_2_config["method"],
        "through_stage": 4,
        "detection_status": "not_evaluated",
    }
    input_path = str(metadata_dict.get("input_path", metadata_dict.get("file_path", "")) or "")
    return PipelineResult(
        raw_signal=raw_signal.copy(),
        physical_preprocessed_signal=final_preprocessing.physical_signal.copy(),
        scaled_preprocessed_signal=final_preprocessing.scaled_signal.copy(),
        window_results=window_results,
        sampling_rate=fs,
        scale_factors={
            "amplitude_995": scale_factor,
            "preprocessing_scale_factor": scale_factor,
        },
        selected_parameters=selected_parameters,
        warnings=warnings,
        recording_id=recording_id,
        input_path=input_path,
        metadata=metadata_dict,
        stage_1=stage_1,
        stage_2=stage_2,
        stage_3=stage_3,
        stage_4=stage_4,
    )
