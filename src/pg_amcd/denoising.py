import numpy as np
import pywt
from typing import Dict, Any

def bayes_shrink_threshold(coeff: np.ndarray, noise_sigma: float) -> float:
    """Calculates the BayesShrink adaptive threshold for a subband."""
    var_y = np.mean(np.square(coeff))
    var_x = max(0.0, var_y - noise_sigma**2)
    if var_x == 0:
        return float(np.max(np.abs(coeff)))
    else:
        return float(noise_sigma**2 / np.sqrt(var_x))

def wavelet_denoise(
    signal: np.ndarray, 
    wavelet_name: str = "db8", 
    level: int = 4
) -> np.ndarray:
    """Applies Bayesian Adaptive Wavelet Denoising using BayesShrink.
    
    Noise standard deviation is estimated from the finest detail subband
    (which is the last element of the coefficients list returned by pywt.wavedec).
    """
    # 1. Clamp decomposition level to prevent ValueError
    try:
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
        if level > max_level:
            level = max_level
    except Exception:
        level = min(level, 4)
        
    # 2. Multilevel decomposition
    # coeffs list structure: [cA_N, cD_N, cD_N-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # 3. Estimate noise standard deviation from finest detail coefficients (cD_1 at index -1)
    d1 = coeffs[-1]
    mad = np.median(np.abs(d1 - np.median(d1)))
    noise_sigma = mad / 0.6745
    if noise_sigma == 0:
        noise_sigma = 1e-6
        
    # 4. Apply soft thresholding to all detail coefficients (index 1 and onwards)
    denoised_coeffs = [coeffs[0]] # approximation remains untouched
    for i in range(1, len(coeffs)):
        coeff = coeffs[i]
        threshold = bayes_shrink_threshold(coeff, noise_sigma)
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
