import os
import sys
import numpy as np
import scipy.io

# Add current directory and src/ to path so we can import pg_amcd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pg_amcd.config import load_pipeline_config
from pg_amcd.weighting import calculate_maiw_weights, reconstruct_weighted_signal

def process_file_maiw(npz_path, output_mat_path):
    print(f"MAIW Weighting on: {os.path.basename(npz_path)}")
    config = load_pipeline_config()
    fs = config["sampling_rate"]
    
    try:
        data = np.load(npz_path)
        time = data['time']
        original_signal = data['original_signal']
        imfs = data['imfs']
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return
        
    W, C, E, K, F = calculate_maiw_weights(imfs, original_signal, fs, config)
    reconstructed = reconstruct_weighted_signal(imfs, W)
    
    # Scale reconstructed signal to maintain normalized range
    max_val = np.max(np.abs(reconstructed))
    if max_val > 0:
        reconstructed = reconstructed / max_val
        
    # Save to mat file
    reconstructed_tsDS = np.column_stack((time, reconstructed))
    scipy.io.savemat(output_mat_path, {
        'tsDS': reconstructed_tsDS,
        'weights': W,
        'correlation': C,
        'energy': E,
        'kurtosis': K,
        'frequency_proximity': F
    })
    print(f"Saved MAIW reconstructed signal to {output_mat_path}\n")

if __name__ == "__main__":
    from pg_amcd.cli import main
    main()
