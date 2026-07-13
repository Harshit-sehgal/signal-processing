import numpy as np
from typing import Dict, Any

from pg_amcd.models import WindowResult, PipelineResult

# Import local pg_amcd modules
from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.segmentation import select_max_energy_segment_indices, generate_sliding_windows
from pg_amcd.decomposition import run_ceemdan, calculate_adjacent_imf_correlation
from pg_amcd.weighting import calculate_maiw_weights, calculate_physics_gated_weights, reconstruct_gated_signal
from pg_amcd.denoising import wavelet_denoise
from pg_amcd.features import extract_window_features
from pg_amcd.optimization import optimize_cutoff
from pg_amcd.validation import orthogonality_index, reconstruction_nrmse


def process_recording(
    time: np.ndarray,
    signal: np.ndarray,
    config: Dict[str, Any],
    metadata: Dict[str, Any] = None,
    mode: str = "exploratory"  # "exploratory" or "sliding_window"
) -> PipelineResult:
    """The canonical entrypoint to process a single machining vibration recording.

    Preserves amplitude and runs input validation, preprocessing, segmentation,
    CEEMDAN, physics-aware gating, and band-aware Wavelet Denoising.
    """
    warnings = []
    fs = config["sampling_rate"]

    pipeline_cfg = config.get("pipeline", {})
    fallback_rpm = pipeline_cfg.get("fallback_rpm", 570.0)
    fallback_tooth_count = pipeline_cfg.get("fallback_tooth_count", 1)
    rpm = metadata.get("rpm", fallback_rpm) if metadata else fallback_rpm
    tooth_count = metadata.get("tooth_count", fallback_tooth_count) if metadata else fallback_tooth_count

    ceemdan_cfg = config["ceemdan"]
    high_cutoff = min(4000.0, fs / 2.0 - 10.0)
    candidate_cutoffs = ceemdan_cfg.get(
        "search_cutoffs", [ceemdan_cfg.get("selected_cutoff", 100.0)]
    )

    # 1. Locate ONE max-energy segment with a preliminary cutoff so that every
    #    candidate cutoff optimises the *same* raw segment (Goal 5.1).
    prelim_cutoff = float(candidate_cutoffs[0])
    phys_prelim, _, _ = preprocess_signal(signal, low_cutoff=prelim_cutoff, high_cutoff=high_cutoff, fs=fs)
    segment_points = config.get("segment_points", 10000)
    start_idx, end_idx = select_max_energy_segment_indices(phys_prelim, segment_points)
    raw_segment = signal[start_idx:end_idx]

    # 2. Adaptive cutoff optimisation over the identical raw segment.
    opt = optimize_cutoff(
        raw_segment,
        candidate_cutoffs,
        config,
        fs,
        n_seeds=max(1, ceemdan_cfg.get("search_seeds", 2)),
    )
    cutoff = opt.selected_cutoff
    cutoff_search = opt.per_cutoff_metrics

    # 3. Actual preprocessing with the selected cutoff.
    physical_preprocessed, scaled_preprocessed, scale_factor = preprocess_signal(
        signal,
        low_cutoff=cutoff,
        high_cutoff=high_cutoff,
        fs=fs
    )

    window_results = []

    # 4. Segment Generation
    if mode == "exploratory":
        windows = [{
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': float(time[start_idx]),
            'end_time': float(time[end_idx - 1] if end_idx <= len(time) else time[-1]),
            'time_segment': time[start_idx:end_idx],
            'signal_segment': scaled_preprocessed[start_idx:end_idx]
        }]
    else:
        windows = generate_sliding_windows(
            time,
            scaled_preprocessed,
            fs,
            window_seconds=1.0,
            overlap_ratio=0.75
        )

    # 3. Process Windows
    ceemdan_cfg = config["ceemdan"]
    trials = ceemdan_cfg["trials"]
    epsilon = ceemdan_cfg["epsilon"]
    seed = ceemdan_cfg["noise_seed"]
    sifting_iterations = ceemdan_cfg.get("sifting_iterations", 16)

    chatter_center = config["maiw"]["chatter_band_center"]
    chatter_spread = config["maiw"]["chatter_band_spread"]

    for win in windows:
        t_seg = win['time_segment']
        s_seg = win['signal_segment']

        # A. CEEMDAN Decomposition
        imfs = run_ceemdan(s_seg, trials, epsilon, seed, sifting_iterations)

        # B. Gating/Weighting (Dynamic toggle for standard vs. physics gating)
        use_physics = config.get("use_physics_gating", True)
        if use_physics:
            W, _, _, _, _ = calculate_physics_gated_weights(
                imfs, s_seg, fs, rpm, tooth_count, config
            )
            reconstructed_scaled = reconstruct_gated_signal(imfs, W)
        else:
            W, _, _, _, _ = calculate_maiw_weights(imfs, s_seg, fs, config)
            reconstructed_scaled = reconstruct_gated_signal(imfs, W)

        # C. Wavelet Denoising (Band-Aware)
        wavelet_cfg = config["wavelet"]
        denoised_scaled = wavelet_denoise(
            reconstructed_scaled,
            wavelet_name=wavelet_cfg["wavelet_name"],
            level=wavelet_cfg["level"],
            fs=fs,
            chatter_center=chatter_center,
            chatter_spread=chatter_spread,
            band_aware=wavelet_cfg.get("band_aware", True),
            chatter_threshold_scale=wavelet_cfg.get("chatter_threshold_scale", 0.5),
            noise_threshold_scale=wavelet_cfg.get("noise_threshold_scale", 1.4),
        )

        # D. Convert Denoised Output back to Physical Units
        denoised_physical = denoised_scaled * scale_factor
        maiw_reconstructed_physical = reconstructed_scaled * scale_factor

        # E. Calculate Features
        features = extract_window_features(
            raw_window=signal[win['start_idx']:win['end_idx']],
            prep_physical_window=physical_preprocessed[win['start_idx']:win['end_idx']],
            denoised_physical_window=denoised_physical,
            imfs=imfs,
            fs=fs,
            rpm=rpm,
            tooth_count=tooth_count,
            chatter_center=chatter_center,
            chatter_spread=chatter_spread
        )

        # Calculate diagnostics using canonical validation functions
        mmi, _ = calculate_adjacent_imf_correlation(imfs)
        oi = orthogonality_index(imfs)
        nrmse = reconstruction_nrmse(s_seg, imfs)

        features["mmi"] = mmi
        features["oi"] = oi
        features["nrmse"] = nrmse

        # Chatter detection is intentionally NOT evaluated in this commit.
        # The placeholder RMS heuristic and arbitrary confidence=0.9 were removed
        # until a validated detector exists (see Segment 6 of the roadmap).
        chatter_probability = float("nan")
        predicted_label = "not_evaluated"
        confidence = float("nan")

        window_results.append(WindowResult(
            time_segment=t_seg,
            start_time=win['start_time'],
            end_time=win['end_time'],
            start_idx=win['start_idx'],
            end_idx=win['end_idx'],
            features=features,
            chatter_probability=chatter_probability,
            predicted_label=predicted_label,
            selected_imfs=[i for i in range(len(W)) if W[i] > 0.05],
            confidence=confidence,
            imfs=imfs,
            maiw_reconstructed=maiw_reconstructed_physical,
            denoised_clean=denoised_physical
        ))

    selected_params = {
        "cutoff_frequency": cutoff,
        "cutoff_search": cutoff_search,
        "ceemdan_trials": trials,
        "ceemdan_epsilon": epsilon,
        "sifting_iterations": sifting_iterations,
        "wavelet_name": config["wavelet"]["wavelet_name"],
        "wavelet_level": config["wavelet"]["level"],
        "use_physics_gating": use_physics
    }

    return PipelineResult(
        raw_signal=signal,
        physical_preprocessed_signal=physical_preprocessed,
        scaled_preprocessed_signal=scaled_preprocessed,
        window_results=window_results,
        sampling_rate=fs,
        scale_factors={"amplitude_995": scale_factor},
        selected_parameters=selected_params,
        warnings=warnings
    )
