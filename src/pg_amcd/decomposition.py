import numpy as np
import scipy.signal
from PyEMD import CEEMDAN
from typing import Dict, Any, Tuple

def run_ceemdan(
    signal: np.ndarray, 
    trials: int, 
    epsilon: float, 
    noise_seed: int, 
    sifting_iterations: int = 16
) -> np.ndarray:
    """Decomposes a scaled signal into IMFs + 1 residual using CEEMDAN.
    
    Args:
        signal: Scaled signal array.
        trials: Number of ensemble trials.
        epsilon: Standard deviation of added noise.
        noise_seed: Seed for random white noise generation.
        sifting_iterations: Number of sifting iterations (forces FIXE value).
        
    Returns:
        imfs: 2D numpy array of shape (num_layers, N) where the last row is the residual.
    """
    ceemdan = CEEMDAN(
        trials=trials, 
        epsilon=epsilon, 
        FIXE=sifting_iterations, 
        parallel=True
    )
    ceemdan.noise_seed(noise_seed)
    
    # Run decomposition
    imfs = ceemdan(signal)
    return imfs

def calculate_adjacent_imf_correlation(imfs: np.ndarray) -> Tuple[float, float]:
    """Measures the Pearson correlation between adjacent IMFs (excluding residual).
    
    Returns:
        mean_adj_corr: Mean adjacent correlation (MMI).
        max_adj_corr: Maximum adjacent correlation.
    """
    num_layers = imfs.shape[0]
    if num_layers <= 2:
        return 0.0, 0.0
        
    num_imfs = num_layers - 1 # exclude residual
    corrs = []
    for i in range(num_imfs - 1):
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        if not np.isnan(corr):
            corrs.append(corr)
            
    mean_corr = float(np.mean(corrs)) if corrs else 1.0
    max_corr = float(np.max(corrs)) if corrs else 1.0
    return mean_corr, max_corr

def calculate_spectral_overlap(imfs: np.ndarray, fs: float) -> float:
    """Calculates the average frequency spectral overlap between adjacent IMFs.
    
    Spectral overlap is the intersection area of adjacent normalized PSDs.
    """
    num_layers = imfs.shape[0]
    num_imfs = num_layers - 1 # exclude residual
    if num_imfs <= 1:
        return 0.0
        
    overlaps = []
    for i in range(num_imfs - 1):
        # Calculate Welch PSD for both adjacent IMFs
        nperseg = min(len(imfs[i]), 1024)
        _, psd1 = scipy.signal.welch(imfs[i], fs, nperseg=nperseg)
        _, psd2 = scipy.signal.welch(imfs[i+1], fs, nperseg=nperseg)
        
        # Normalize PSDs so they act as probability density functions (sum to 1.0)
        sum1 = np.sum(psd1)
        sum2 = np.sum(psd2)
        if sum1 > 0:
            psd1 = psd1 / sum1
        if sum2 > 0:
            psd2 = psd2 / sum2
            
        # Calculate intersection area
        overlap = np.sum(np.minimum(psd1, psd2))
        overlaps.append(overlap)
        
    return float(np.mean(overlaps)) if overlaps else 0.0

def calculate_orthogonality_index(imfs: np.ndarray, original_signal: np.ndarray) -> float:
    """Calculates the global Orthogonality Index (OI) of the decomposition.
    
    OI = 2 * sum_{i < j} sum_t (c_i * c_j) / sum_t (X^2)
    """
    num_layers = imfs.shape[0]
    cross_terms = 0.0
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            cross_terms += np.sum(imfs[i] * imfs[j])
            
    orig_energy = np.sum(original_signal ** 2)
    return float(2.0 * cross_terms / orig_energy) if orig_energy > 0 else 0.0

def calculate_composite_cutoff_score(
    imfs: np.ndarray, 
    original_signal: np.ndarray, 
    fs: float
) -> float:
    """Computes a multi-factor optimization score for the decomposition.
    
    Score = 0.35 * Mean Spectral Overlap + 0.35 * Max Adjacent Correlation + 0.30 * |Orthogonality Index|
    """
    mean_corr, max_corr = calculate_adjacent_imf_correlation(imfs)
    spectral_overlap = calculate_spectral_overlap(imfs, fs)
    oi = calculate_orthogonality_index(imfs, original_signal)
    
    score = 0.35 * spectral_overlap + 0.35 * max_corr + 0.30 * abs(oi)
    return float(score)
