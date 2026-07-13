import numpy as np
import pywt
from typing import Dict, Any

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
    # 1. Clamp decomposition level to prevent ValueError
    try:
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
        if level > max_level:
            level = max_level
    except Exception:
        level = min(level, 4)
        
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
