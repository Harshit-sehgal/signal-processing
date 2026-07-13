import os
import sys
import glob
import numpy as np
import scipy.io
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.segmentation import select_max_energy_segment_indices
from pg_amcd.decomposition import run_ceemdan, calculate_composite_cutoff_score

def perform_optimized_decomposition(raw_path, preprocessed_path, npz_path, plot_path):
    print(f"\nDecomposing & Optimizing: {os.path.basename(raw_path)}")
    config = load_pipeline_config()
    ceemdan_cfg = config["ceemdan"]
    fs = config["sampling_rate"]
    high_cutoff_val = min(4000.0, fs / 2.0 - 10.0)
    
    try:
        # Load and validate signal
        time_arr, raw_vibration, _ = validate_and_load_signal(raw_path, fs)
    except Exception as e:
        print(f"Error validating {raw_path}: {e}")
        return
        
    # 1. Parameter loop over low cutoffs to find best preprocessing
    cutoffs = ceemdan_cfg["search_cutoffs"]
    best_score = float('inf')
    best_cutoff = 100
    
    print("Looping cutoffs to find optimal preprocessing...")
    for cut in cutoffs:
        # Preprocess
        _, scaled, _ = preprocess_signal(raw_vibration, cut, high_cutoff_val, fs)
        
        # Segment indices for max energy
        start_idx, end_idx = select_max_energy_segment_indices(scaled, config["segment_points"])
        s_seg = scaled[start_idx:end_idx]
        
        # Fast evaluation using search_trials
        trials = ceemdan_cfg["search_trials"]
        epsilon = ceemdan_cfg["epsilon"]
        seed = ceemdan_cfg["noise_seed"]
        sifting_iterations = ceemdan_cfg.get("sifting_iterations", 16)
        
        imfs = run_ceemdan(s_seg, trials, epsilon, seed, sifting_iterations)
        score = calculate_composite_cutoff_score(imfs, s_seg, fs)
        print(f"  Cutoff {cut} Hz -> Composite Score: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_cutoff = cut
            
    print(f"Optimal low cutoff selected: {best_cutoff} Hz (score: {best_score:.4f})")
    
    # 2. Final preprocessing with optimal cutoff and save
    physical_prep, scaled_prep, scale_factor = preprocess_signal(raw_vibration, best_cutoff, high_cutoff_val, fs)
    reconstructed_tsDS = np.column_stack((time_arr, physical_prep))
    scipy.io.savemat(preprocessed_path, {'tsDS': reconstructed_tsDS})
    print(f"Saved optimized preprocessed data to {preprocessed_path}")
    
    # 3. Final decomposition with trials from config
    start_idx, end_idx = select_max_energy_segment_indices(physical_prep, config["segment_points"])
    opt_t_seg = time_arr[start_idx:end_idx]
    opt_s_seg = scaled_prep[start_idx:end_idx]
    
    trials = ceemdan_cfg["trials"]
    epsilon = ceemdan_cfg["epsilon"]
    seed = ceemdan_cfg["noise_seed"]
    sifting_iterations = ceemdan_cfg.get("sifting_iterations", 16)
    
    print(f"Running final CEEMDAN (trials={trials}, epsilon={epsilon}, sifting_iterations={sifting_iterations})...")
    imfs = run_ceemdan(opt_s_seg, trials, epsilon, seed, sifting_iterations)
    
    # Save IMFs
    np.savez_compressed(npz_path, time=opt_t_seg, original_signal=opt_s_seg, imfs=imfs, start_index=start_idx)
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
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    # Resolve paths relative to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "Vibration - ML")
    preprocessed_dir = os.path.join(root_dir, "Vibration - ML_Preprocessed")
    imf_output_dir = os.path.join(root_dir, f"Vibration_IMFs{suffix}")
    
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
