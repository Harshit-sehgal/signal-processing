import os
import glob
import numpy as np
import scipy.io
import scipy.signal
import scipy.stats
from config_utils import load_pipeline_config

def calculate_maiw_weights(imfs, original_signal, fs=None):
    config = load_pipeline_config()
    if fs is None:
        fs = config["sampling_rate"]
        
    num_imfs = imfs.shape[0] - 1 # exclude residual
    
    C = np.zeros(num_imfs)
    E = np.zeros(num_imfs)
    K = np.zeros(num_imfs)
    F = np.zeros(num_imfs)
    
    # Calculate energy of all IMFs for normalization
    total_energy = np.sum([np.sum(np.square(imfs[i])) for i in range(num_imfs)])
    if total_energy == 0:
        total_energy = 1.0
        
    # Calculate kurtosis of all IMFs for normalization
    kurtoses = np.zeros(num_imfs)
    for i in range(num_imfs):
        kurtoses[i] = scipy.stats.kurtosis(imfs[i], fisher=False)
    total_kurtosis = np.sum(kurtoses)
    if total_kurtosis == 0:
        total_kurtosis = 1.0
        
    # Load parameters from config
    maiw_cfg = config["maiw"]
    alpha = maiw_cfg.get("alpha", 0.25)
    beta = maiw_cfg.get("beta", 0.25)
    gamma = maiw_cfg.get("gamma", 0.25)
    delta = maiw_cfg.get("delta", 0.25)
    center = maiw_cfg.get("chatter_band_center", 1250.0)
    spread = maiw_cfg.get("chatter_band_spread", 500.0)
    
    for i in range(num_imfs):
        imf = imfs[i]
        
        # 1. Correlation (C_k)
        corr_matrix = np.corrcoef(imf, original_signal)
        C[i] = np.abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # 2. Energy (E_k)
        E[i] = np.sum(np.square(imf)) / total_energy
        
        # 3. Kurtosis (K_k)
        K[i] = kurtoses[i] / total_kurtosis
        
        # 4. Frequency Proximity (F_k)
        # Compute Welch PSD to find dominant frequency
        freqs, psd = scipy.signal.welch(imf, fs, nperseg=1024)
        dom_freq = freqs[np.argmax(psd)]
        
        # Chatter band center and spread from config
        F[i] = np.exp(-((dom_freq - center) ** 2) / (2.0 * (spread ** 2)))
        
    W = alpha * C + beta * E + gamma * K + delta * F
    
    # Normalize weights so they sum to 1.0
    sum_W = np.sum(W)
    if sum_W > 0:
        W = W / sum_W
    else:
        W = np.ones(num_imfs) / num_imfs
        
    return W, C, E, K, F

def process_file_maiw(npz_path, output_mat_path):
    print(f"MAIW Weighting on: {os.path.basename(npz_path)}")
    try:
        data = np.load(npz_path)
        time = data['time']
        original_signal = data['original_signal']
        imfs = data['imfs']
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return
        
    W, C, E, K, F = calculate_maiw_weights(imfs, original_signal)
    
    num_imfs = imfs.shape[0] - 1
    reconstructed = np.zeros_like(original_signal)
    
    print("IMF weights:")
    for i in range(num_imfs):
        reconstructed += W[i] * imfs[i]
        print(f"  IMF {i+1}: Weight = {W[i]:.4f}")
        
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
    from config_utils import load_pipeline_config
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    imf_dir = f"/home/harshit/Documents/Research/Vibration_IMFs{suffix}"
    output_dir = f"/home/harshit/Documents/Research/Vibration_MAIW{suffix}"
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
