import numpy as np
import scipy.stats
import scipy.signal
import scipy.fftpack
from typing import Dict, Any

def extract_window_features(
    raw_window: np.ndarray,
    prep_physical_window: np.ndarray,
    denoised_physical_window: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    rpm: float,
    tooth_count: int,
    chatter_center: float = 1250.0,
    chatter_spread: float = 500.0
) -> Dict[str, float]:
    """Extracts time, frequency, time-frequency, and EMD features from a single window.
    
    Returns:
        A dictionary containing all calculated feature names mapped to float values.
    """
    features = {}
    N = len(denoised_physical_window)
    
    # ----------------------------------------------------
    # 1. Time-Domain Features (Denoised Physical Signal)
    # ----------------------------------------------------
    rms = np.sqrt(np.mean(np.square(denoised_physical_window)))
    var = np.var(denoised_physical_window)
    peak_to_peak = np.max(denoised_physical_window) - np.min(denoised_physical_window)
    
    # Crest Factor
    max_abs = np.max(np.abs(denoised_physical_window))
    crest_factor = max_abs / rms if rms > 0 else 0.0
    
    kurtosis = float(scipy.stats.kurtosis(denoised_physical_window, fisher=False))
    skewness = float(scipy.stats.skew(denoised_physical_window))
    
    features["time_rms"] = rms
    features["time_variance"] = var
    features["time_peak_to_peak"] = peak_to_peak
    features["time_crest_factor"] = crest_factor
    features["time_kurtosis"] = kurtosis
    features["time_skewness"] = skewness
    
    # ----------------------------------------------------
    # 2. Frequency-Domain Features
    # ----------------------------------------------------
    # Welch PSD
    nperseg = min(N, 1024)
    freqs, psd = scipy.signal.welch(denoised_physical_window, fs, nperseg=nperseg)
    total_psd = np.sum(psd)
    
    if total_psd > 0:
        # Normalized PSD as probability density
        psd_norm = psd / total_psd
        
        # Spectral Centroid
        centroid = np.sum(freqs * psd_norm)
        # Spectral Spread
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
        # Spectral Entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    else:
        centroid = 0.0
        spread = 0.0
        entropy = 0.0
        
    features["freq_centroid"] = centroid
    features["freq_spread"] = spread
    features["freq_entropy"] = entropy
    
    # Spindle and Tooth Harmonic Energies
    f_spindle = rpm / 60.0
    f_tooth = f_spindle * tooth_count
    harmonic_freqs = [k * f_tooth for k in range(1, 6)]
    
    if total_psd > 0:
        # Energy inside expected chatter band
        chatter_min = chatter_center - chatter_spread
        chatter_max = chatter_center + chatter_spread
        chatter_mask = (freqs >= chatter_min) & (freqs <= chatter_max)
        features["freq_chatter_band_ratio"] = float(np.sum(psd[chatter_mask]) / total_psd)
        
        # Energy in tooth passing harmonics
        harmonic_mask = np.zeros_like(freqs, dtype=bool)
        for h_freq in harmonic_freqs:
            harmonic_mask |= (np.abs(freqs - h_freq) <= 15.0)
        features["freq_harmonics_ratio"] = float(np.sum(psd[harmonic_mask]) / total_psd)
    else:
        features["freq_chatter_band_ratio"] = 0.0
        features["freq_harmonics_ratio"] = 0.0
        
    # Peak Frequency
    if len(psd) > 0:
        features["freq_peak"] = float(freqs[np.argmax(psd)])
    else:
        features["freq_peak"] = 0.0
        
    # ----------------------------------------------------
    # 3. IMF Features (EMD Domain)
    # ----------------------------------------------------
    num_layers = imfs.shape[0]
    num_imfs = num_layers - 1 # exclude residual
    
    # IMF Energy ratios
    imf_energies = [np.sum(np.square(imfs[i])) for i in range(num_imfs)]
    total_imf_energy = np.sum(imf_energies)
    
    if total_imf_energy > 0:
        # Max IMF energy ratio
        features["imf_max_energy_ratio"] = float(np.max(imf_energies) / total_imf_energy)
    else:
        features["imf_max_energy_ratio"] = 0.0
        
    # Correlation between first IMF and preprocessed signal
    if num_imfs > 0:
        corr_val = np.abs(np.corrcoef(imfs[0], prep_physical_window)[0, 1])
        features["imf1_correlation"] = float(corr_val) if not np.isnan(corr_val) else 0.0
    else:
        features["imf1_correlation"] = 0.0
        
    # ----------------------------------------------------
    # 4. Time-Frequency (Wavelet Energy) Features
    # ----------------------------------------------------
    import pywt
    coeffs = pywt.wavedec(denoised_physical_window, "db8", level=4)
    wavelet_energies = [np.sum(c ** 2) for c in coeffs]
    total_wavelet_energy = np.sum(wavelet_energies)
    
    if total_wavelet_energy > 0:
        # Ratio of high-frequency detail coefficients (level 1 & 2)
        features["wavelet_high_freq_ratio"] = float((wavelet_energies[-1] + wavelet_energies[-2]) / total_wavelet_energy)
    else:
        features["wavelet_high_freq_ratio"] = 0.0
        
    return features
