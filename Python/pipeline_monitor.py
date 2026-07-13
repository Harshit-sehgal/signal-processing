import os
import sys
import time
import glob
import subprocess

# Add current directory to path so we can import packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Clean imports from renamed modules
try:
    from maiw_weighting import process_file_maiw
    from wavelet_denoise import process_file_denoise
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def check_active_processes():
    try:
        # Check if iceemdan.py is running
        res = subprocess.run(["pgrep", "-f", "iceemdan.py"], capture_output=True, text=True)
        pids = res.stdout.strip().split()
        return len(pids) > 0
    except Exception:
        return False

def main():
    from config_utils import load_pipeline_config
    config = load_pipeline_config()
    suffix = config.get("output_suffix", "")
    
    # Resolve paths dynamically relative to repository root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imf_dir = os.path.join(root_dir, f"Vibration_IMFs{suffix}")
    maiw_dir = os.path.join(root_dir, f"Vibration_MAIW{suffix}")
    clean_dir = os.path.join(root_dir, f"Vibration_Clean{suffix}")
    
    os.makedirs(maiw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    print("==================================================")
    print("🔍 PG-AMCD Incremental Pipeline Monitor Started 🔍")
    print("==================================================")
    print("This script will monitor Vibration_IMFs for new outputs")
    print("and run Stage 2 (MAIW) and Stage 3 (Wavelet Denoising) on them.")
    print("==================================================\n")
    
    # Loop as long as 02_iceemdan.py is running, or run at least once
    first_run = True
    while first_run or check_active_processes():
        first_run = False
        
        # Find all .npz files in Vibration_IMFs
        all_npz_files = glob.glob(os.path.join(imf_dir, "**/*_IMFs.npz"), recursive=True)
        new_processed = 0
        
        for npz_path in all_npz_files:
            folder_name = os.path.basename(os.path.dirname(npz_path))
            base_name = os.path.basename(npz_path).replace("_IMFs.npz", ".mat")
            
            # Target output paths
            maiw_folder = os.path.join(maiw_dir, folder_name)
            clean_folder = os.path.join(clean_dir, folder_name)
            
            os.makedirs(maiw_folder, exist_ok=True)
            os.makedirs(clean_folder, exist_ok=True)
            
            maiw_path = os.path.join(maiw_folder, base_name)
            clean_path = os.path.join(clean_folder, base_name)
            
            # Check if denoised file already exists
            if os.path.exists(clean_path):
                continue
                
            print(f"\n✨ Found new IMF output: {folder_name}/{base_name}")
            
            # Run Stage 2: MAIW Weighting
            try:
                if not os.path.exists(maiw_path):
                    process_file_maiw(npz_path, maiw_path)
                else:
                    print(f"  MAIW file already exists, skipping to denoising.")
            except Exception as e:
                print(f"❌ Error in MAIW stage for {base_name}: {e}")
                continue
                
            # Run Stage 3: Wavelet Denoising
            try:
                process_file_denoise(maiw_path, clean_path)
                new_processed += 1
                print(f"✅ Successfully completed pipeline for {folder_name}/{base_name}!")
            except Exception as e:
                print(f"❌ Error in Wavelet Denoising stage for {base_name}: {e}")
                continue
                
        if new_processed > 0:
            print(f"\nProcessed {new_processed} new file(s) in this iteration.")
            
        # If CEEMDAN is still running, wait before next check
        if check_active_processes():
            print("⏳ iceemdan.py is still running. Sleeping for 15 seconds...", end="\r")
            time.sleep(15)
        else:
            print("\n🏁 iceemdan.py has finished execution or is not running.")
            print("Running final sweep...")
            break
            
    print("\n==================================================")
    print("🎉 Pipeline Monitor Finished! All ready files processed.")
    print("==================================================")

if __name__ == "__main__":
    main()
