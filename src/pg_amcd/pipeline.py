import os
import numpy as np
import scipy.stats
import scipy.signal
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Import local pg_amcd modules
from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.segmentation import select_max_energy_segment_indices, generate_sliding_windows
from pg_amcd.decomposition import run_ceemdan, calculate_adjacent_imf_correlation
from pg_amcd.weighting import calculate_maiw_weights, calculate_physics_gated_weights, reconstruct_weighted_signal
from pg_amcd.denoising import wavelet_denoise
from pg_amcd.features import extract_window_features

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

def apply_temporal_decision_logic(
    window_results: List[WindowResult],
    enter_chatter_threshold: float = 0.60,
    exit_chatter_threshold: float = 0.35,
    min_consecutive_windows: int = 2
):
    """Applies median probability filtering and a hysteresis state machine over adjacent windows.
    
    Prevents prediction flickering and ensures stable cutting-state transitions.
    """
    if not window_results:
        return
        
    probs = [res.chatter_probability for res in window_results]
    
    # 1. Median filter to smooth window-to-window flicker
    kernel_size = min(len(probs) | 1, 3) # ensure odd kernel size
    smoothed_probs = scipy.signal.medfilt(probs, kernel_size=kernel_size)
    
    # 2. Hysteresis State Machine
    state = "stable"
    for i, res in enumerate(window_results):
        sp = smoothed_probs[i]
        
        if state == "stable":
            # Require enter threshold to be exceeded for consecutive windows
            if sp >= enter_chatter_threshold:
                consecutive = True
                for k in range(min_consecutive_windows):
                    if i + k < len(smoothed_probs):
                        if smoothed_probs[i + k] < enter_chatter_threshold:
                            consecutive = False
                            break
                    else:
                        consecutive = False
                        break
                if consecutive:
                    state = "chatter"
        elif state == "chatter":
            # Require drop below exit threshold to transition back to stable
            if sp < exit_chatter_threshold:
                state = "stable"
                
        # Update predictions
        res.chatter_probability = float(sp)
        res.predicted_label = state

def process_recording(
    time: np.ndarray,
    signal: np.ndarray,
    config: Dict[str, Any],
    metadata: Dict[str, Any] = None,
    mode: str = "exploratory"  # "exploratory" or "sliding_window"
) -> PipelineResult:
    """The canonical entrypoint to process a single machining vibration recording.
    
    Preserves amplitude and runs input validation, preprocessing, segmentation, 
    CEEMDAN, physics-aware gating, band-aware Wavelet Denoising, and temporal decision logic.
    """
    warnings = []
    fs = config["sampling_rate"]
    
    rpm = metadata.get("rpm", 570.0) if metadata else 570.0
    tooth_count = metadata.get("tooth_count", 1) if metadata else 1
    
    # 1. Preprocessing (optimal cutoff is loaded or pre-calculated)
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
            reconstructed_scaled = reconstruct_weighted_signal(imfs, W)
        else:
            W, _, _, _, _ = calculate_maiw_weights(imfs, s_seg, fs, config)
            reconstructed_scaled = reconstruct_weighted_signal(imfs, W)
            
        # C. Wavelet Denoising (Band-Aware)
        denoised_scaled = wavelet_denoise(
            reconstructed_scaled, 
            wavelet_name=config["wavelet"]["wavelet_name"], 
            level=config["wavelet"]["level"],
            fs=fs,
            chatter_center=chatter_center,
            chatter_spread=chatter_spread,
            band_aware=config.get("wavelet", {}).get("band_aware", True)
        )
        
        # D. Convert Denoised Output back to Physical Units
        denoised_physical = denoised_scaled * scale_factor
        maiw_reconstructed_physical = reconstructed_scaled * scale_factor
        
        # E. Calculate Features (Goal 11)
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
        
        # Calculate diagnostics
        mmi, _ = calculate_adjacent_imf_correlation(imfs)
        
        cross_terms = 0.0
        num_layers = imfs.shape[0]
        for idx1 in range(num_layers):
            for idx2 in range(idx1 + 1, num_layers):
                cross_terms += np.sum(imfs[idx1] * imfs[idx2])
        orig_energy = np.sum(s_seg ** 2)
        oi = float(2.0 * cross_terms / orig_energy) if orig_energy > 0 else 0.0
        
        recon_err = np.sum(imfs, axis=0) - s_seg
        nrmse = float(np.sqrt(np.mean(recon_err ** 2)) / np.sqrt(np.mean(s_seg ** 2))) if np.any(s_seg) else 0.0
        
        features["mmi"] = mmi
        features["oi"] = oi
        features["nrmse"] = nrmse
        
        # Simple threshold classification based on RMS
        rms_val = features["time_rms"]
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
        
    # 4. Apply Hysteresis and Smoothing Temporal Logic (Goal 14)
    apply_temporal_decision_logic(
        window_results,
        enter_chatter_threshold=config.get("enter_chatter_threshold", 0.60),
        exit_chatter_threshold=config.get("exit_chatter_threshold", 0.35),
        min_consecutive_windows=config.get("min_consecutive_windows", 2)
    )
    
    selected_params = {
        "cutoff_frequency": cutoff,
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
