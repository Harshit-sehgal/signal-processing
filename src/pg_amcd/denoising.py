import numpy as np
import pywt
from pg_amcd.synthetic import evaluate_denoising_performance

def bayes_shrink_threshold(coeff: np.ndarray, noise_sigma: float) -> float:
    """Calculates the standard BayesShrink adaptive threshold for a subband."""
    var_y = np.mean(np.square(coeff))
    var_x = max(0.0, var_y - noise_sigma**2)
    if var_x == 0:
        return float(np.max(np.abs(coeff)))
    else:
        return float(noise_sigma**2 / np.sqrt(var_x))

def wavelet_denoise(
    signal: np.ndarray, 
    wavelet_name: str = "db8", 
    level: int = 4,
    fs: float = 10000.0,
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0,
    band_aware: bool = True
) -> np.ndarray:
    """Applies Bayesian Adaptive Wavelet Denoising.
    
    If band_aware is True, the threshold is scaled depending on whether the DWT
    subband overlaps with the expected chatter frequency band.
    """
    # 1. Validate wavelet and decomposition level
    wavelet = pywt.Wavelet(wavelet_name)  # raises ValueError if name is invalid
    if level < 1:
        raise ValueError(f"Wavelet decomposition level must be >= 1, got {level}")
    if len(signal) < 2:
        raise ValueError(f"Signal too short for wavelet denoising: {len(signal)} samples")
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    if max_level < 1:
        raise ValueError(
            f"Signal length {len(signal)} is too short for wavelet '{wavelet_name}' "
            f"(max decomposition level is 0)"
        )
    if level > max_level:
        level = max_level
        
    # 2. Multilevel decomposition
    # coeffs list: [cA_N, cD_N, cD_N-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # 3. Estimate noise standard deviation from finest detail coefficients (cD_1 at index -1)
    d1 = coeffs[-1]
    mad = np.median(np.abs(d1 - np.median(d1)))
    noise_sigma = mad / 0.6745
    if noise_sigma == 0:
        noise_sigma = 1e-6
        
    # Chatter frequency band limits
    chatter_min = chatter_center - chatter_spread
    chatter_max = chatter_center + chatter_spread
    
    # 4. Apply thresholding to all detail coefficients (index 1 and onwards)
    denoised_coeffs = [coeffs[0]] # approximation remains untouched
    for i in range(1, len(coeffs)):
        coeff = coeffs[i]
        
        # Calculate base threshold
        threshold = bayes_shrink_threshold(coeff, noise_sigma)
        
        if band_aware:
            # Determine DWT level of this subband (cD_j)
            # coeffs[1] is level N (coarsest), coeffs[-1] is level 1 (finest)
            subband_level = level - i + 1
            
            # Frequency range of this subband
            low_limit = fs * (2.0 ** -(subband_level + 1))
            high_limit = fs * (2.0 ** -subband_level)
            
            # Check overlap with chatter band
            overlap = (high_limit >= chatter_min) and (low_limit <= chatter_max)
            
            if overlap:
                # Halve threshold to preserve chatter components (Goal 8)
                threshold *= 0.5
            else:
                # Scale up threshold in noise-dominated bands
                threshold *= 1.4
                
        denoised_coeff = pywt.threshold(coeff, threshold, mode='soft')
        denoised_coeffs.append(denoised_coeff)
        
    # 5. Reconstruct
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet_name)
    
    # Trim or pad to match the exact original length
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='edge')
        
    return denoised_signal


def evaluate_wavelet_config(
    signal: np.ndarray,
    clean_reference: np.ndarray,
    wavelet_name: str,
    level: int,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
) -> dict:
    """Quantitative denoising quality for one wavelet configuration (Goal 5.6)."""
    denoised = wavelet_denoise(
        signal,
        wavelet_name=wavelet_name,
        level=level,
        fs=fs,
        chatter_center=chatter_center,
        chatter_spread=chatter_spread,
        band_aware=True,
    )
    return evaluate_denoising_performance(clean_reference, denoised, fs, chatter_center, chatter_spread)


def select_best_wavelet(
    signal: np.ndarray,
    clean_reference: np.ndarray,
    candidates: list,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
):
    """Evaluate candidate ``(wavelet_name, level)`` configs and return the best.

    Selection uses the highest SNR-improvement (dB) versus the clean reference.
    Returns ``(best_config_dict, all_results)``.
    """
    results = []
    for wavelet_name, level in candidates:
        metrics = evaluate_wavelet_config(
            signal, clean_reference, wavelet_name, level, fs, chatter_center, chatter_spread
        )
        results.append({"wavelet": wavelet_name, "level": level, **metrics})
    best = max(results, key=lambda d: d["snr_db"])
    return best, results
