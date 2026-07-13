import os
import glob
import numpy as np
import scipy.io
import scipy.signal
from PyEMD import CEEMDAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config_utils import load_pipeline_config

def preprocess_raw_signal(raw_vibration, low_cutoff, fs=None):
    config = load_pipeline_config()
    if fs is None:
        fs = config["sampling_rate"]
        
    nyquist = 0.5 * fs
    high_cutoff = 4000.0
    b, a = scipy.signal.butter(3, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
    denoised = scipy.signal.filtfilt(b, a, raw_vibration)
    detrended = scipy.signal.detrend(denoised)
    max_val = np.max(np.abs(detrended))
    normalized = detrended / (max_val if max_val > 0 else 1.0)
    return normalized

def select_max_energy_segment(signal, time, segment_points=None):
    config = load_pipeline_config()
    if segment_points is None:
        segment_points = config["segment_points"]
        
    if len(signal) > segment_points:
        squared_signal = np.square(signal)
        window = np.ones(segment_points)
        rolling_energy = np.convolve(squared_signal, window, mode='valid')
        start_idx = np.argmax(rolling_energy)
        end_idx = start_idx + segment_points
        return signal[start_idx:end_idx], time[start_idx:end_idx]
    else:
        return signal, time

def evaluate_mode_mixing(s_seg, low_cutoff, trials=None, epsilon=None):
    config = load_pipeline_config()
    ceemdan_cfg = config["ceemdan"]
    if trials is None:
        trials = ceemdan_cfg["search_trials"]
    if epsilon is None:
        epsilon = ceemdan_cfg["epsilon"]
        
    fxe = ceemdan_cfg.get("sifting_iterations", 0)
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, FIXE=fxe, parallel=True)
    ceemdan.noise_seed(ceemdan_cfg["noise_seed"])
    imfs = ceemdan(s_seg)
    num_imfs = imfs.shape[0]
    
    corrs = []
    for i in range(num_imfs - 2): # exclude residual
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        if not np.isnan(corr):
            corrs.append(corr)
            
    avg_corr = np.mean(corrs) if corrs else 1.0
    return avg_corr, imfs

def perform_optimized_decomposition(raw_path, preprocessed_path, npz_path, plot_path):
    print(f"\nDecomposing & Optimizing: {os.path.basename(raw_path)}")
    config = load_pipeline_config()
    ceemdan_cfg = config["ceemdan"]
    
    try:
        raw_data = scipy.io.loadmat(raw_path)['tsDS']
    except Exception as e:
        print(f"Error loading {raw_path}: {e}")
        return
        
    time_arr = raw_data[:, 0]
    raw_vibration = raw_data[:, 1]
    
    # 1. Parameter loop over low cutoffs to find best preprocessing
    cutoffs = ceemdan_cfg["search_cutoffs"]
    best_score = float('inf')
    best_cutoff = 100
    
    print("Looping cutoffs to find optimal preprocessing...")
    for cut in cutoffs:
        normalized = preprocess_raw_signal(raw_vibration, cut)
        s_seg, _ = select_max_energy_segment(normalized, time_arr)
        
        # Fast evaluation
        score, _ = evaluate_mode_mixing(s_seg, cut)
        print(f"  Cutoff {cut} Hz -> Avg Adj IMF Corr: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_cutoff = cut
            
    print(f"Optimal low cutoff selected: {best_cutoff} Hz (score: {best_score:.4f})")
    
    # 2. Final preprocessing with optimal cutoff and save
    opt_normalized = preprocess_raw_signal(raw_vibration, best_cutoff)
    reconstructed_tsDS = np.column_stack((time_arr, opt_normalized))
    scipy.io.savemat(preprocessed_path, {'tsDS': reconstructed_tsDS})
    print(f"Saved optimized preprocessed data to {preprocessed_path}")
    
    # 3. Final decomposition with trials from config
    opt_s_seg, opt_t_seg = select_max_energy_segment(opt_normalized, time_arr)
    trials = ceemdan_cfg["trials"]
    epsilon = ceemdan_cfg["epsilon"]
    seed = ceemdan_cfg["noise_seed"]
    fxe = ceemdan_cfg.get("sifting_iterations", 0)
    print(f"Running final CEEMDAN (trials={trials}, epsilon={epsilon}, FIXE={fxe})...")
    
    final_ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, FIXE=fxe, parallel=True)
    final_ceemdan.noise_seed(seed)
    imfs = final_ceemdan(opt_s_seg)
    
    # Save IMFs
    np.savez_compressed(npz_path, time=opt_t_seg, original_signal=opt_s_seg, imfs=imfs)
    print(f"Saved IMFs mathematically to {npz_path}")
    
    # Save Plot
    num_imfs = imfs.shape[0]
    plt.figure(figsize=(14, 2 * (num_imfs + 1)))
    plt.subplot(num_imfs + 1, 1, 1)
    plt.plot(opt_t_seg, opt_s_seg, color='black')
    plt.title(f"Optimized Preprocessed Input Signal (Cutoff={best_cutoff}Hz) - {os.path.basename(raw_path)}")
    plt.ylabel("Input")
    
    for i in range(num_imfs):
        plt.subplot(num_imfs + 1, 1, i + 2)
        if i == num_imfs - 1:
            plt.plot(opt_t_seg, imfs[i], color='red')
            plt.ylabel("Residual")
        else:
            plt.plot(opt_t_seg, imfs[i], color='blue')
            plt.ylabel(f"IMF {i+1}")
            
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved visual decomposition to {plot_path}")

if __name__ == "__main__":
    from config_utils import load_pipeline_config
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    raw_dir = "/home/harshit/Documents/Research/Vibration - ML"
    preprocessed_dir = "/home/harshit/Documents/Research/Vibration - ML_Preprocessed"
    imf_output_dir = f"/home/harshit/Documents/Research/Vibration_IMFs{suffix}"
    
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(imf_output_dir, exist_ok=True)
    
    all_raw_files = glob.glob(os.path.join(raw_dir, "**/*.mat"), recursive=True)
    print(f"Found {len(all_raw_files)} raw data files to process.")
    
    for raw_path in all_raw_files:
        if raw_path.endswith("combinations.xlsx") or "~lock" in raw_path:
            continue
            
        folder_name = os.path.basename(os.path.dirname(raw_path))
        base_name = os.path.basename(raw_path).replace(".mat", "")
        
        preprocessed_folder = os.path.join(preprocessed_dir, folder_name)
        imf_folder = os.path.join(imf_output_dir, folder_name)
        
        os.makedirs(preprocessed_folder, exist_ok=True)
        os.makedirs(imf_folder, exist_ok=True)
        
        preprocessed_path = os.path.join(preprocessed_folder, f"{base_name}.mat")
        npz_path = os.path.join(imf_folder, f"{base_name}_IMFs.npz")
        plot_path = os.path.join(imf_folder, f"{base_name}_IMFs_plot.png")
        
        if os.path.exists(npz_path):
            print(f"Skipping {base_name} (CEEMDAN already computed and saved!)")
            continue
            
        try:
            perform_optimized_decomposition(raw_path, preprocessed_path, npz_path, plot_path)
        except Exception as e:
            print(f"❌ Error processing file {base_name}: {e}")
            print("Skipping to the next file...")
        
    print("\nAll files successfully processed with optimized parameter search!")
