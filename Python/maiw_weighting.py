import os
import sys
import glob
import numpy as np
import scipy.io

# Add current directory to path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    # Resolve paths dynamically relative to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imf_dir = os.path.join(root_dir, f"Vibration_IMFs{suffix}")
    output_dir = os.path.join(root_dir, f"Vibration_MAIW{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_npz_files = glob.glob(os.path.join(imf_dir, "**/*_IMFs.npz"), recursive=True)
    print(f"Found {len(all_npz_files)} IMF files to process.")
    
    for npz_path in all_npz_files:
        folder_name = os.path.basename(os.path.dirname(npz_path))
        base_name = os.path.basename(npz_path).replace("_IMFs.npz", ".mat")
        
        target_folder = os.path.join(output_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        output_mat_path = os.path.join(target_folder, base_name)
        if os.path.exists(output_mat_path):
            print(f"Skipping {base_name} (MAIW already computed!)")
            continue
        process_file_maiw(npz_path, output_mat_path)
        
    print("MAIW Weighting Phase Complete!")
