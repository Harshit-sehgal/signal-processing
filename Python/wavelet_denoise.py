import os
import sys
import numpy as np
import scipy.io

# Add current directory and src/ to path so we can import pg_amcd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pg_amcd.config import load_pipeline_config
from pg_amcd.denoising import wavelet_denoise

def process_file_denoise(maiw_mat_path, clean_mat_path):
    print(f"Wavelet Denoising on: {os.path.basename(maiw_mat_path)}")
    config = load_pipeline_config()
    fs = config["sampling_rate"]
    
    try:
        data = scipy.io.loadmat(maiw_mat_path)
        tsDS = data['tsDS']
        time = tsDS[:, 0]
        reconstructed = tsDS[:, 1]
    except Exception as e:
        print(f"Error loading {maiw_mat_path}: {e}")
        return
        
    denoised = wavelet_denoise(
        reconstructed,
        wavelet_name=config["wavelet"]["wavelet_name"],
        level=config["wavelet"]["level"],
        fs=fs
    )
    
    # Save to mat file
    clean_tsDS = np.column_stack((time, denoised))
    scipy.io.savemat(clean_mat_path, {'tsDS': clean_tsDS})
    print(f"Saved clean signal to {clean_mat_path}\n")

if __name__ == "__main__":
    from pg_amcd.cli import main
    main()
