import os
import sys
import glob
import numpy as np
import scipy.io

# Add current directory and src/ to path so we can import pg_amcd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.preprocessing import preprocess_signal

def preprocess_signal_legacy(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    config = load_pipeline_config()
    fs = config["sampling_rate"]
    low_cutoff = config["preprocessing"]["low_cutoff"]
    high_cutoff = config["preprocessing"]["high_cutoff"]
    order = config["preprocessing"]["order"]
    
    # Dynamically cap high_cutoff if Nyquist is too low
    high_cutoff_val = min(high_cutoff, fs / 2.0 - 10.0)
    
    try:
        time, raw_vibration, _ = validate_and_load_signal(file_path, fs)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
        
    # Reuse standard robust preprocessing library (SOS filtering + detrending)
    physical_prep, _, _ = preprocess_signal(
        raw_vibration,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff_val,
        fs=fs,
        order=order
    )
    
    # Normalized scaling expected by legacy script
    max_val = np.max(np.abs(physical_prep))
    if max_val > 0:
        normalized = physical_prep / max_val
    else:
        normalized = physical_prep
        
    return time, normalized

if __name__ == "__main__":
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    # Resolve paths dynamically relative to repository root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root_dir, "Vibration - ML")
    output_dir = os.path.join(root_dir, f"Vibration - ML_Preprocessed{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_mat_files = glob.glob(os.path.join(base_dir, "**/*.mat"), recursive=True)
    print(f"Found {len(all_mat_files)} total .mat data files to process.")
    
    for file_path in all_mat_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        
        target_folder = os.path.join(output_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        target_mat_path = os.path.join(target_folder, file_name)
        
        if os.path.exists(target_mat_path):
            print(f"Skipping {file_name} in {folder_name} (Already preprocessed!)")
            continue
            
        print(f"\n[{folder_name}] -> Generating {file_name}...")
        time, preprocessed_vibration = preprocess_signal_legacy(file_path)
        
        if time is not None and preprocessed_vibration is not None:
            reconstructed_tsDS = np.column_stack((time, preprocessed_vibration))
            scipy.io.savemat(target_mat_path, {'tsDS': reconstructed_tsDS})
            print(f"Success! Saved preprocessed data to {target_mat_path}")
            
    print("\nAll files have been successfully preprocessed, denoised, detrended, and normalized!")
