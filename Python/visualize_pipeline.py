import os
import sys
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_comparison_plot(stickout_folder="2p5inch_stickout", file_name="u_570_005.mat"):
    from config_utils import load_pipeline_config
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    base_name = file_name.replace(".mat", "")
    
    # File paths
    raw_path = f"/home/harshit/Documents/Research/Vibration - ML/{stickout_folder}/{file_name}"
    prep_path = f"/home/harshit/Documents/Research/Vibration - ML_Preprocessed/{stickout_folder}/{file_name}"
    maiw_path = f"/home/harshit/Documents/Research/Vibration_MAIW{suffix}/{stickout_folder}/{file_name}"
    clean_path = f"/home/harshit/Documents/Research/Vibration_Clean{suffix}/{stickout_folder}/{file_name}"
    
    print(f"Generating comparison plot for: {stickout_folder}/{file_name}")
    
    # Load data
    try:
        raw_data = scipy.io.loadmat(raw_path)['tsDS']
        raw_vibration = raw_data[:, 1]
        time = raw_data[:, 0]
    except Exception as e:
        print(f"Error loading raw data from {raw_path}: {e}")
        return
        
    try:
        prep_vibration = scipy.io.loadmat(prep_path)['tsDS'][:, 1]
    except Exception as e:
        print(f"Error loading preprocessed data from {prep_path}: {e}")
        return
        
    try:
        maiw_vibration = scipy.io.loadmat(maiw_path)['tsDS'][:, 1]
    except Exception as e:
        print(f"Error loading MAIW data from {maiw_path}: {e}")
        return
        
    try:
        clean_vibration = scipy.io.loadmat(clean_path)['tsDS'][:, 1]
    except Exception as e:
        print(f"Error loading clean data from {clean_path}: {e}")
        return
        
    # Standardize lengths
    N = min(len(raw_vibration), len(prep_vibration), len(maiw_vibration), len(clean_vibration))
    time = time[:N]
    raw_vibration = raw_vibration[:N]
    prep_vibration = prep_vibration[:N]
    maiw_vibration = maiw_vibration[:N]
    clean_vibration = clean_vibration[:N]
    
    # Crop to a 0.2 second window for detailed wave shape view
    # 0.2 seconds at 10kHz = 2000 points
    # Let's take the middle 2000 points
    start_idx = N // 2 - 1000
    end_idx = N // 2 + 1000
    t_crop = time[start_idx:end_idx]
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Raw Signal
    plt.subplot(4, 1, 1)
    plt.plot(t_crop, raw_vibration[start_idx:end_idx], color='gray', alpha=0.8)
    plt.title(f"Stage 0: Raw Vibration Signal (Unprocessed) - {file_name}", fontsize=12, fontweight='bold')
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Preprocessed Signal
    plt.subplot(4, 1, 2)
    plt.plot(t_crop, prep_vibration[start_idx:end_idx], color='orange')
    plt.title("Stage 1: Bandpass Preprocessed Signal (Optimized Cutoff)", fontsize=12, fontweight='bold')
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: MAIW Signal
    plt.subplot(4, 1, 3)
    plt.plot(t_crop, maiw_vibration[start_idx:end_idx], color='blue')
    plt.title("Stage 2: Multi-Criteria Adaptive IMF Weighting (Reconstructed)", fontsize=12, fontweight='bold')
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 4: Clean Signal
    plt.subplot(4, 1, 4)
    plt.plot(t_crop, clean_vibration[start_idx:end_idx], color='green', linewidth=1.5)
    plt.title("Stage 3: Bayesian Adaptive Wavelet Denoised (Final Clean Output)", fontsize=12, fontweight='bold')
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_plot_path = f"/home/harshit/Documents/Research/Vibration_Clean{suffix}/{stickout_folder}/{base_name}_pipeline_comparison.png"
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Generated comparison plot: {output_plot_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        stickout_folder = sys.argv[1]
        file_name = sys.argv[2]
        if not file_name.endswith(".mat"):
            file_name += ".mat"
        generate_comparison_plot(stickout_folder, file_name)
    else:
        print("Usage: python visualize_pipeline.py <stickout_folder> <file_name>")
        print("Falling back to default: 2p5inch_stickout u_570_005.mat")
        generate_comparison_plot()
