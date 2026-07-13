import os
import sys
import glob
import numpy as np
import scipy.io

# Add current directory to path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pg_amcd.config import load_pipeline_config
from pg_amcd.denoising import wavelet_denoise

def process_file_denoise(mat_path, output_mat_path):
    print(f"Wavelet Denoising on: {os.path.basename(mat_path)}")
    config = load_pipeline_config()
    wav_cfg = config["wavelet"]
    
    try:
        data = scipy.io.loadmat(mat_path)['tsDS']
        time = data[:, 0]
        maiw_signal = data[:, 1]
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return
        
    clean_signal = wavelet_denoise(
        maiw_signal, 
        wavelet_name=wav_cfg["wavelet_name"], 
        level=wav_cfg["level"]
    )
    
    # Scale clean signal to preserve normalized amplitude range
    max_val = np.max(np.abs(clean_signal))
    if max_val > 0:
        clean_signal = clean_signal / max_val
        
    # Save to mat file
    clean_tsDS = np.column_stack((time, clean_signal))
    scipy.io.savemat(output_mat_path, {'tsDS': clean_tsDS})
    print(f"Saved clean signal to {output_mat_path}\n")

if __name__ == "__main__":
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    # Resolve paths dynamically relative to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maiw_dir = os.path.join(root_dir, f"Vibration_MAIW{suffix}")
    output_dir = os.path.join(root_dir, f"Vibration_Clean{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_mat_files = glob.glob(os.path.join(maiw_dir, "**/*.mat"), recursive=True)
    print(f"Found {len(all_mat_files)} MAIW files to denoise.")
    
    for mat_path in all_mat_files:
        folder_name = os.path.basename(os.path.dirname(mat_path))
        file_name = os.path.basename(mat_path)
        
        target_folder = os.path.join(output_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        output_mat_path = os.path.join(target_folder, file_name)
        if os.path.exists(output_mat_path):
            print(f"Skipping {file_name} (Wavelet Denoising already computed!)")
            continue
        process_file_denoise(mat_path, output_mat_path)
        
    print("Wavelet Denoising Phase Complete!")
