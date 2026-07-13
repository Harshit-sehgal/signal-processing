import numpy as np
import scipy.signal
import scipy.stats
from typing import Tuple, Dict, Any

def normalize_indicator(values: np.ndarray) -> np.ndarray:
    """Safely normalizes an indicator array across layers so it sums to 1.0."""
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(values)
    return values / total if total > 0 else np.ones_like(values) / len(values)

def calculate_maiw_weights(
    imfs: np.ndarray, 
    original_signal: np.ndarray, 
    fs: float,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Multi-Criteria Adaptive IMF weights based on Correlation, 
    Energy, Kurtosis, and Frequency Proximity.
    
    The residual (last IMF layer) is excluded from weighting.
    """
    num_layers = imfs.shape[0]
    num_weighted = num_layers - 1 # exclude residual
    
    C = np.zeros(num_weighted)
    E = np.zeros(num_weighted)
    K = np.zeros(num_weighted)
    F = np.zeros(num_weighted)
    
    maiw_cfg = config["maiw"]
    alpha = maiw_cfg.get("alpha", 0.25)
    beta = maiw_cfg.get("beta", 0.25)
    gamma = maiw_cfg.get("gamma", 0.25)
    delta = maiw_cfg.get("delta", 0.25)
    center = maiw_cfg.get("chatter_band_center", 1250.0)
    spread = maiw_cfg.get("chatter_band_spread", 500.0)
    
    # Pre-calculate energy of all physical IMFs
    total_energy = np.sum([np.sum(np.square(imfs[i])) for i in range(num_weighted)])
    if total_energy == 0:
        total_energy = 1.0
        
    # Pre-calculate kurtosis of all physical IMFs
    kurtoses = np.zeros(num_weighted)
    for i in range(num_weighted):
        kurtoses[i] = scipy.stats.kurtosis(imfs[i], fisher=False)
    total_kurtosis = np.sum(kurtoses)
    if total_kurtosis == 0:
        total_kurtosis = 1.0
        
    for i in range(num_weighted):
        imf = imfs[i]
        
        # 1. Correlation (C_i)
        corr_matrix = np.corrcoef(imf, original_signal)
        C[i] = np.abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # 2. Energy (E_i)
        E[i] = np.sum(np.square(imf)) / total_energy
        
        # 3. Kurtosis (K_i)
        K[i] = kurtoses[i] / total_kurtosis
        
        # 4. Frequency Proximity (F_i)
        # Compute Welch PSD to find dominant peak frequency
        freqs, psd = scipy.signal.welch(imf, fs, nperseg=min(len(imf), 1024))
        dom_freq = freqs[np.argmax(psd)]
        
        # Gaussian proximity score
        F[i] = np.exp(-((dom_freq - center) ** 2) / (2.0 * (spread ** 2)))
        
    # Normalize indicators so they are comparable
    nC = normalize_indicator(C)
    nE = normalize_indicator(E)
    nK = normalize_indicator(K)
    nF = normalize_indicator(F)
    
    # Compute weights
    W = alpha * nC + beta * nE + gamma * nK + delta * nF
    
    # Normalize final weights to sum to 1.0
    sum_W = np.sum(W)
    if sum_W > 0:
        W = W / sum_W
    else:
        W = np.ones(num_weighted) / num_weighted
        
    return W, C, E, K, F

def reconstruct_weighted_signal(imfs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reconstructs the vibration signal by summing physical IMFs weighted by W.
    The residual trend (last layer) is excluded from the reconstruction.
    """
    num_weighted = len(weights)
    reconstructed = np.zeros(imfs.shape[1])
    for i in range(num_weighted):
        reconstructed += weights[i] * imfs[i]
    return reconstructed
