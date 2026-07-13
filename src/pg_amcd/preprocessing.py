import numpy as np
import scipy.signal
from typing import Tuple

def butter_bandpass_filter_sos(
    signal: np.ndarray, 
    low_cutoff: float, 
    high_cutoff: float, 
    fs: float, 
    order: int = 3
) -> np.ndarray:
    """Applies a Butterworth bandpass filter using Second-Order Sections (SOS) 
    for numerical robustness, avoiding float overflow/underflow issues.
    """
    nyquist = 0.5 * fs
    # Validate cutoffs
    if low_cutoff <= 0 or high_cutoff >= nyquist or low_cutoff >= high_cutoff:
        raise ValueError(
            f"Invalid filter cutoffs: [{low_cutoff}, {high_cutoff}] Hz for Nyquist {nyquist} Hz"
        )
        
    sos = scipy.signal.butter(
        order,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    return scipy.signal.sosfiltfilt(sos, signal)

def preprocess_signal(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Filters, detrends, and scales the raw signal.
    
    Returns:
        physical_preprocessed: Detrended and bandpass filtered signal in physical units.
        scaled_preprocessed: Scaled signal in range roughly [-1, 1] for numerical algorithm execution.
        scale_factor: The 99.5th percentile amplitude scale factor used.
    """
    # 1. Bandpass filter
    filtered = butter_bandpass_filter_sos(signal, low_cutoff, high_cutoff, fs, order=order)
    
    # 2. Detrend (remove linear drift)
    physical_preprocessed = scipy.signal.detrend(filtered)
    
    # 3. Robust scaling based on 99.5th percentile amplitude (preserves physical scale factor)
    scale_factor = np.percentile(np.abs(physical_preprocessed), 99.5)
    if scale_factor == 0:
        scale_factor = 1e-12
        
    scaled_preprocessed = physical_preprocessed / scale_factor
    
    return physical_preprocessed, scaled_preprocessed, scale_factor
