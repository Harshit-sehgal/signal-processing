import os
import glob
import numpy as np
import scipy.io
import pywt
from config_utils import load_pipeline_config

def bayes_shrink_threshold(coeff, noise_sigma):
    """Calculate BayesShrink threshold for a set of wavelet coefficients."""
    var_y = np.mean(np.square(coeff))
    var_x = max(0.0, var_y - noise_sigma**2)
    if var_x == 0:
        return np.max(np.abs(coeff))
    else:
        return noise_sigma**2 / np.sqrt(var_x)

def wavelet_denoise(signal, wavelet=None, level=None):
    """Perform Bayesian Adaptive Wavelet Denoising using BayesShrink."""
    config = load_pipeline_config()
    if wavelet is None:
        wavelet = config["wavelet"]["wavelet_name"]
    if level is None:
        level = config["wavelet"]["level"]
        
    # Clamp decomposition level to max level allowed to prevent ValueError
    try:
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        if level > max_level:
            print(f"Warning: level {level} exceeds max level {max_level}. Clamping to {max_level}.")
            level = max_level
    except Exception as e:
        print(f"Error checking max level: {e}")
        
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Detail coefficients at the finest level (d1) are at index -1
    d1 = coeffs[-1]
    
    # Estimate noise standard deviation using MAD
    mad = np.median(np.abs(d1 - np.median(d1)))
    noise_sigma = mad / 0.6745
    if noise_sigma == 0:
        noise_sigma = 1e-6
        
    # Apply soft thresholding to all detail coefficients (coeffs[1:])
    denoised_coeffs = [coeffs[0]] # Keep approximation coefficients untouched
    
    for i in range(1, len(coeffs)):
        coeff = coeffs[i]
        threshold = bayes_shrink_threshold(coeff, noise_sigma)
        denoised_coeff = pywt.threshold(coeff, threshold, mode='soft')
        denoised_coeffs.append(denoised_coeff)
        
    # Reconstruct signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # Trim to original size if needed due to padding/decomposition length differences
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='edge')
        
    return denoised_signal

def process_file_denoise(mat_path, output_mat_path):
    print(f"Wavelet Denoising on: {os.path.basename(mat_path)}")
    try:
        data = scipy.io.loadmat(mat_path)['tsDS']
        time = data[:, 0]
        maiw_signal = data[:, 1]
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return
        
    clean_signal = wavelet_denoise(maiw_signal)
    
    # Scale clean signal to preserve normalized amplitude range
    max_val = np.max(np.abs(clean_signal))
    if max_val > 0:
        clean_signal = clean_signal / max_val
        
    # Save to mat file
    clean_tsDS = np.column_stack((time, clean_signal))
    scipy.io.savemat(output_mat_path, {'tsDS': clean_tsDS})
    print(f"Saved clean signal to {output_mat_path}\n")

if __name__ == "__main__":
    from config_utils import load_pipeline_config
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    maiw_dir = f"/home/harshit/Documents/Research/Vibration_MAIW{suffix}"
    output_dir = f"/home/harshit/Documents/Research/Vibration_Clean{suffix}"
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
