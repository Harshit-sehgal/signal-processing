import numpy as np
from PyEMD import CEEMDAN
from typing import Dict, Any

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

def calculate_adjacent_imf_correlation(imfs: np.ndarray) -> float:
    """Measures the Mode-Mixing Index (MMI) by calculating the mean Pearson 
    correlation coefficient between adjacent IMFs (excluding the residual).
    """
    num_layers = imfs.shape[0]
    if num_layers <= 2:
        return 0.0
        
    num_imfs = num_layers - 1 # exclude residual
    corrs = []
    for i in range(num_imfs - 1):
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        if not np.isnan(corr):
            corrs.append(corr)
            
    return float(np.mean(corrs)) if corrs else 1.0
