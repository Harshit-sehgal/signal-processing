import numpy as np
import scipy.signal
import scipy.stats
import scipy.fftpack
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
    
    total_energy = np.sum([np.sum(np.square(imfs[i])) for i in range(num_weighted)])
    if total_energy == 0:
        total_energy = 1.0
        
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
        freqs, psd = scipy.signal.welch(imf, fs, nperseg=min(len(imf), 1024))
        dom_freq = freqs[np.argmax(psd)]
        F[i] = np.exp(-((dom_freq - center) ** 2) / (2.0 * (spread ** 2)))
        
    nC = normalize_indicator(C)
    nE = normalize_indicator(E)
    nK = normalize_indicator(K)
    nF = normalize_indicator(F)
    
    W = alpha * nC + beta * nE + gamma * nK + delta * nF
    
    sum_W = np.sum(W)
    if sum_W > 0:
        W = W / sum_W
    else:
        W = np.ones(num_weighted) / num_weighted
        
    return W, C, E, K, F

def calculate_physics_gated_weights(
    imfs: np.ndarray,
    original_signal: np.ndarray,
    fs: float,
    rpm: float,
    tooth_count: int,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates independent sigmoidal gates for each IMF layer based on machining physics.
    
    Sigmoid Gate = sigmoid(a1 * E_chatter + a2 * C + a3 * K - a4 * E_harmonics - offset)
    
    Returns:
        gates: Independent sigmoidal weights for each IMF (shape: (num_weighted,))
        C, E_chatter, K, E_harmonics: Diagnostic arrays for reporting.
    """
    num_layers = imfs.shape[0]
    num_weighted = num_layers - 1 # exclude residual
    
    gates = np.zeros(num_weighted)
    C = np.zeros(num_weighted)
    E_chatter = np.zeros(num_weighted)
    K = np.zeros(num_weighted)
    E_harmonics = np.zeros(num_weighted)
    
    maiw_cfg = config["maiw"]
    chatter_center = maiw_cfg.get("chatter_band_center", 1250.0)
    chatter_spread = maiw_cfg.get("chatter_band_spread", 500.0)
    
    # Machine physics parameters
    f_spindle = rpm / 60.0
    f_tooth = f_spindle * tooth_count
    harmonic_freqs = [k * f_tooth for k in range(1, 6)] # check up to 5th tooth harmonic
    
    for i in range(num_weighted):
        imf = imfs[i]
        
        # A. Pearson Correlation (C)
        corr_matrix = np.corrcoef(imf, original_signal)
        C[i] = np.abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # B. Kurtosis (K)
        # Standardize K to [0, 1] relative to nominal Gaussian kurtosis (3.0)
        k_val = float(scipy.stats.kurtosis(imf, fisher=False))
        K[i] = min(1.0, max(0.0, (k_val - 3.0) / 10.0))
        
        # C. Spectral analysis via FFT
        N = len(imf)
        fft_vals = np.abs(scipy.fftpack.fft(imf))
        freqs = scipy.fftpack.fftfreq(N, 1.0 / fs)
        pos_idx = freqs >= 0
        pos_freqs = freqs[pos_idx]
        psd = (fft_vals[pos_idx]) ** 2
        total_psd = np.sum(psd)
        
        if total_psd > 0:
            # D. Chatter Band Energy (E_chatter)
            chatter_mask = (pos_freqs >= (chatter_center - chatter_spread)) & (pos_freqs <= (chatter_center + chatter_spread))
            E_chatter[i] = np.sum(psd[chatter_mask]) / total_psd
            
            # E. Harmonic forced vibration energy (E_harmonics)
            harmonic_mask = np.zeros_like(pos_freqs, dtype=bool)
            for h_freq in harmonic_freqs:
                # 15 Hz sideband tolerance window
                harmonic_mask |= (np.abs(pos_freqs - h_freq) <= 15.0)
            E_harmonics[i] = np.sum(psd[harmonic_mask]) / total_psd
        else:
            E_chatter[i] = 0.0
            E_harmonics[i] = 0.0
            
        # Sigmoid function parameters
        a1 = 4.0   # Chatter energy coefficient
        a2 = 2.0   # Correlation coefficient
        a3 = 1.0   # Kurtosis coefficient
        a4 = 5.0   # Harmonics penalty coefficient (high penalty for forced harmonics)
        offset = 1.5
        
        score = a1 * E_chatter[i] + a2 * C[i] + a3 * K[i] - a4 * E_harmonics[i] - offset
        gates[i] = 1.0 / (1.0 + np.exp(-score))
        
    return gates, C, E_chatter, K, E_harmonics

def reconstruct_weighted_signal(imfs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reconstructs the vibration signal by summing physical IMFs weighted by weights."""
    num_weighted = len(weights)
    reconstructed = np.zeros(imfs.shape[1])
    for i in range(num_weighted):
        reconstructed += weights[i] * imfs[i]
    return reconstructed
