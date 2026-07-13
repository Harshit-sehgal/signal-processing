import os
import glob
import numpy as np
import scipy.io
from scipy import signal
from scipy import signal

def preprocess_signal(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    print("1. Loading Data...")
    try:
        data = scipy.io.loadmat(file_path)['tsDS']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
        
    time = data[:, 0]
    raw_vibration = data[:, 1]
    
    print("2. Denoising (Band-Pass Filter)... starts")
    # The sensor samples at 10,000 Hz. We apply a 3rd-order Butterworth band-pass filter
    # from 100 Hz to 4,000 Hz to remove low-frequency spindle forced vibrations and
    # ultra-high frequency electrical 'hiss'.
    nyquist = 0.5 * 10000 
    low_cutoff = 100
    high_cutoff = 4000         
    b, a = signal.butter(3, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
    denoised = signal.filtfilt(b, a, raw_vibration)
    print("2. Denoising... finished")
    
    print("3. Detrending (Centering)... starts")
    # Removes any "slope" or DC offset (sensor drift) from the data so it sits perfectly around zero.
    # Doing this AFTER denoising works beautifully, especially if the high-frequency noise was slightly unbalancing the average.
    detrended = signal.detrend(denoised)
    print("3. Detrending... finished")
    
    print("4. Normalization (Scaling)... starts")
    max_val = np.max(np.abs(detrended))
    normalized = detrended / max_val
    print("4. Normalization... finished")
    
    
    return time, normalized

if __name__ == "__main__":

    base_dir = "/home/harshit/Documents/Research/Vibration - ML"
    
    output_dir = "/home/harshit/Documents/Research/Vibration - ML_Preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    
    all_mat_files = glob.glob(os.path.join(base_dir, "**/*.mat"), recursive=True) #for loading all the files, glob utility searches for the similar file path and fetch it
    
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
        
        time, preprocessed_vibration = preprocess_signal(file_path)
        
        if time is not None and preprocessed_vibration is not None:
            reconstructed_tsDS = np.column_stack((time, preprocessed_vibration))
            
            scipy.io.savemat(target_mat_path, {'tsDS': reconstructed_tsDS})
            print(f"Success! Saved preprocessed data to {target_mat_path}")
            
    print("\nAll files have been successfully preprocessed, denoised, detrended, and normalized!")
