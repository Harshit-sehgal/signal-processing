import os
import sys
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_visual(stickout_folder="2inch_stickout", file_name="c_320_005.mat"):
    base_name = file_name.replace(".mat", "")
    
    raw_file = f"/home/harshit/Documents/Research/Vibration - ML/{stickout_folder}/{file_name}"
    processed_file = f"/home/harshit/Documents/Research/Vibration - ML_Preprocessed/{stickout_folder}/{file_name}"
    
    print(f"Generating preprocessing visual for: {stickout_folder}/{file_name}")
    
    # Load raw
    try:
        raw_data = scipy.io.loadmat(raw_file)['tsDS']
        time = raw_data[:, 0]
        raw_vib = raw_data[:, 1]
    except Exception as e:
        print(f"Error loading raw file {raw_file}: {e}")
        return
        
    # Load processed
    try:
        proc_data = scipy.io.loadmat(processed_file)['tsDS']
        proc_vib = proc_data[:, 1]
    except Exception as e:
        print(f"Error loading preprocessed file {processed_file}: {e}")
        return
        
    # We will zoom in on a small 0.1 second window to actually see the noise and the smoothing
    # 0.1 seconds at 10,000 Hz = 1000 data points. Let's take points from the middle of the signal
    N = min(len(raw_vib), len(proc_vib))
    start_idx = N // 3
    end_idx = start_idx + 1000
    
    zoom_time = time[start_idx:end_idx]
    zoom_raw = raw_vib[start_idx:end_idx]
    zoom_proc = proc_vib[start_idx:end_idx]
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Full raw signal
    plt.subplot(3, 1, 1)
    plt.plot(time, raw_vib, color='gray', alpha=0.7)
    plt.title(f"Entire Raw Sensor Signal (Chatter State - {base_name})", fontsize=14)
    plt.ylabel("Raw Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Zoomed-in Raw
    plt.subplot(3, 1, 2)
    plt.plot(zoom_time, zoom_raw, color='red', label="Raw (Noisy & Drifting)")
    plt.title("Zoomed-in Raw Signal (0.1 Second Window) - Notice the high-frequency jagged noise", fontsize=14)
    plt.ylabel("Raw Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Zoomed-in Processed
    plt.subplot(3, 1, 3)
    plt.plot(zoom_time, zoom_proc, color='green', linewidth=2, label="Processed (Filtered, Centered, Scaled)")
    plt.title("Zoomed-in Preprocessed Signal - Notice the smooth physical vibration curve", fontsize=14)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude (-1 to 1)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save inside preprocessed folder
    plot_path = f"/home/harshit/Documents/Research/Vibration - ML_Preprocessed/{stickout_folder}/{base_name}_preprocessing_visual.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved visual plot to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        stickout_folder = sys.argv[1]
        file_name = sys.argv[2]
        if not file_name.endswith(".mat"):
            file_name += ".mat"
        create_visual(stickout_folder, file_name)
    else:
        print("Usage: python visualize_preprocess.py <stickout_folder> <file_name>")
        print("Falling back to default: 2inch_stickout c_320_005.mat")
        create_visual()
