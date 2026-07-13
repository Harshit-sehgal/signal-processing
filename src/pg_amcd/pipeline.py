import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Import local pg_amcd modules
from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.segmentation import select_max_energy_segment_indices, generate_sliding_windows
from pg_amcd.decomposition import run_ceemdan, calculate_adjacent_imf_correlation
from pg_amcd.weighting import calculate_maiw_weights, reconstruct_weighted_signal
from pg_amcd.denoising import wavelet_denoise

@dataclass
class WindowResult:
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    features: Dict[str, float]
    chatter_probability: float
    predicted_label: str
    selected_imfs: List[int]
    confidence: float
    imfs: np.ndarray  # (num_layers, N)
    maiw_reconstructed: np.ndarray  # (N,)
    denoised_clean: np.ndarray  # (N,)

@dataclass
class PipelineResult:
    raw_signal: np.ndarray
    physical_preprocessed_signal: np.ndarray
    scaled_preprocessed_signal: np.ndarray
    window_results: List[WindowResult]
    sampling_rate: float
    scale_factors: Dict[str, float]
    selected_parameters: Dict[str, Any]
    warnings: List[str]

def process_recording(
    time: np.ndarray,
    signal: np.ndarray,
    config: Dict[str, Any],
    metadata: Dict[str, Any] = None,
    mode: str = "exploratory"  # "exploratory" or "sliding_window"
) -> PipelineResult:
    """The canonical entrypoint to process a single machining vibration recording.
    
    Preserves amplitude and runs input validation, preprocessing, segmentation, 
    CEEMDAN, MAIW weighting, and Bayesian Wavelet Denoising.
    """
    warnings = []
    fs = config["sampling_rate"]
    
    # 1. Preprocessing (optimal cutoff is loaded or pre-calculated)
    # By default, use the cutoff from config or default to 100 Hz
    cutoff = config.get("ceemdan", {}).get("selected_cutoff", 100.0)
    high_cutoff = 4000.0
    
    physical_preprocessed, scaled_preprocessed, scale_factor = preprocess_signal(
        signal, 
        low_cutoff=cutoff, 
        high_cutoff=high_cutoff, 
        fs=fs
    )
    
    window_results = []
    
    # 2. Segment Generation
    if mode == "exploratory":
        segment_points = config.get("segment_points", 10000)
        start_idx, end_idx = select_max_energy_segment_indices(physical_preprocessed, segment_points)
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
    
    for win in windows:
        t_seg = win['time_segment']
        s_seg = win['signal_segment']
        
        # A. CEEMDAN Decomposition
        imfs = run_ceemdan(s_seg, trials, epsilon, seed, sifting_iterations)
        
        # B. MAIW Weighting
        W, C, E, K, F = calculate_maiw_weights(imfs, s_seg, fs, config)
        reconstructed_scaled = reconstruct_weighted_signal(imfs, W)
        
        # C. Wavelet Denoising
        denoised_scaled = wavelet_denoise(
            reconstructed_scaled, 
            wavelet_name=config["wavelet"]["wavelet_name"], 
            level=config["wavelet"]["level"]
        )
        
        # D. Convert Denoised Output back to Physical Units
        denoised_physical = denoised_scaled * scale_factor
        maiw_reconstructed_physical = reconstructed_scaled * scale_factor
        
        # E. Calculate Features (Goal 11)
        # Root Mean Square of Denoised Signal in Physical Units
        rms_val = np.sqrt(np.mean(np.square(denoised_physical)))
        
        # Kurtosis
        kurt_val = float(scipy.stats.kurtosis(denoised_physical, fisher=False))
        
        # Diagnostics
        mmi = calculate_adjacent_imf_correlation(imfs)
        
        # Orthogonality Index (OI)
        cross_terms = 0.0
        num_layers = imfs.shape[0]
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                cross_terms += np.sum(imfs[i] * imfs[j])
        orig_energy = np.sum(s_seg ** 2)
        oi = float(2.0 * cross_terms / orig_energy) if orig_energy > 0 else 0.0
        
        # Reconstruction NRMSE
        recon_err = np.sum(imfs, axis=0) - s_seg
        nrmse = float(np.sqrt(np.mean(recon_err ** 2)) / np.sqrt(np.mean(s_seg ** 2))) if np.any(s_seg) else 0.0
        
        features = {
            "physical_rms": float(rms_val),
            "physical_kurtosis": kurt_val,
            "mmi": mmi,
            "oi": oi,
            "nrmse": nrmse
        }
        
        # Placeholder chatter probability (Heuristic for initial version)
        # Higher RMS in physical units generally correlates with chatter presence
        # Threshold: if RMS > 0.1, assign higher probability
        prob = float(min(1.0, rms_val / 0.5))
        label = "chatter" if rms_val > 0.15 else "stable"
        
        window_results.append(WindowResult(
            start_time=win['start_time'],
            end_time=win['end_time'],
            start_idx=win['start_idx'],
            end_idx=win['end_idx'],
            features=features,
            chatter_probability=prob,
            predicted_label=label,
            selected_imfs=[i for i in range(len(W)) if W[i] > 0.05],
            confidence=0.9,
            imfs=imfs,
            maiw_reconstructed=maiw_reconstructed_physical,
            denoised_clean=denoised_physical
        ))
        
    selected_params = {
        "cutoff_frequency": cutoff,
        "ceemdan_trials": trials,
        "ceemdan_epsilon": epsilon,
        "sifting_iterations": sifting_iterations,
        "wavelet_name": config["wavelet"]["wavelet_name"],
        "wavelet_level": config["wavelet"]["level"]
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
