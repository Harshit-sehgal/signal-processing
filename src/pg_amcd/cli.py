import os
import sys
import argparse
import glob
import time
import json
import hashlib
import subprocess
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local pg_amcd modules
from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.pipeline import process_recording, PipelineResult

def get_md5_checksum(file_path: str) -> str:
    """Calculates the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_git_commit_sha() -> str:
    """Retrieves the current git commit SHA of the repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "Unknown"

def get_environment_info() -> dict:
    """Collects python, packages, and OS information."""
    import platform
    import scipy
    import pywt
    info = {
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywavelets": pywt.__version__
        }
    }
    try:
        import PyEMD
        info["packages"]["PyEMD"] = PyEMD.__version__
    except Exception:
        pass
    return info

def run_pipeline_on_dataset(args):
    # 1. Load config
    config = load_pipeline_config(args.config)
    fs = config["sampling_rate"]
    
    # 2. Get list of files
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
        
    mat_files = glob.glob(os.path.join(args.input_dir, "**/*.mat"), recursive=True)
    mat_files = [f for f in mat_files if not f.endswith("combinations.xlsx") and "~lock" not in f]
    
    if not mat_files:
        print(f"No MAT files found in: {args.input_dir}")
        sys.exit(0)
        
    print("=" * 65)
    print(f"🚀 Running PG-AMCD CLI Pipeline on {len(mat_files)} files 🚀")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 65)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect run metadata
    run_metadata = {
        "start_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit_sha(),
        "env_info": get_environment_info(),
        "resolved_config": config,
        "files_processed": [],
        "failures": []
    }
    
    success_count = 0
    failure_count = 0
    
    for raw_path in mat_files:
        folder_name = os.path.basename(os.path.dirname(raw_path))
        base_name = os.path.basename(raw_path)
        rel_path = os.path.join(folder_name, base_name)
        
        print(f"\nProcessing: {rel_path}")
        
        try:
            # A. Validate and load signal (Strict Goal 3 Input Checks)
            t_arr, sig_arr, fs_est = validate_and_load_signal(
                raw_path, 
                configured_fs=fs, 
                tolerance=0.05, 
                min_duration_seconds=1.0
            )
            
            # Compute file checksum
            file_md5 = get_md5_checksum(raw_path)
            
            # B. Run the canonical pipeline
            res: PipelineResult = process_recording(t_arr, sig_arr, config, mode="exploratory")
            
            # Prepare file output directories
            target_folder = os.path.join(args.output_dir, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            # Save stage results
            # Save IMFs (Stage 1)
            # Take the first window result for exploratory mode
            win_res = res.window_results[0]
            
            npz_name = base_name.replace(".mat", "_IMFs.npz")
            npz_path = os.path.join(target_folder, npz_name)
            np.savez_compressed(
                npz_path, 
                time=win_res.time_segment, 
                original_signal=res.scaled_preprocessed_signal[win_res.start_idx:win_res.end_idx],
                imfs=win_res.imfs,
                start_index=win_res.start_idx
            )
            
            # Save MAIW (Stage 2)
            maiw_name = base_name
            maiw_path = os.path.join(target_folder, maiw_name)
            scipy.io.savemat(maiw_path, {
                'tsDS': np.column_stack((win_res.time_segment, win_res.maiw_reconstructed))
            })
            
            # To match the monitor and pipeline, let's write to Vibration_Clean
            # In the new structure, we can just save all outputs under the output directory:
            # outputs/Vibration_IMFs/, outputs/Vibration_MAIW/, outputs/Vibration_Clean/
            # Or save directly to target folder
            clean_mat_path = os.path.join(target_folder, base_name.replace(".mat", "_Clean.mat"))
            scipy.io.savemat(clean_mat_path, {
                'tsDS': np.column_stack((win_res.time_segment, win_res.denoised_clean))
            })
            
            # Save IMF plot
            num_imfs = win_res.imfs.shape[0]
            plt.figure(figsize=(12, 1.5 * (num_imfs + 1)))
            plt.subplot(num_imfs + 1, 1, 1)
            plt.plot(win_res.time_segment, win_res.maiw_reconstructed, color='black')
            plt.title(f"CEEMDAN & Denoised Decomposition - {base_name}")
            plt.ylabel("Weighted Reconstruct")
            for i in range(num_imfs):
                plt.subplot(num_imfs + 1, 1, i + 2)
                plt.plot(win_res.time_segment, win_res.imfs[i], color='blue' if i < num_imfs - 1 else 'red')
                plt.ylabel(f"IMF {i+1}" if i < num_imfs - 1 else "Residual")
            plt.xlabel("Time (seconds)")
            plt.tight_layout()
            plt.savefig(os.path.join(target_folder, base_name.replace(".mat", "_IMFs_plot.png")), dpi=120)
            plt.close()
            
            success_count += 1
            run_metadata["files_processed"].append({
                "path": rel_path,
                "md5": file_md5,
                "cutoff": res.selected_parameters["cutoff_frequency"],
                "diagnostics": win_res.features
            })
            print(f"✅ Success: NRMSE={win_res.features['nrmse']:.2e}, MMI={win_res.features['mmi']:.4f}, OI={win_res.features['oi']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            failure_count += 1
            run_metadata["failures"].append({
                "path": rel_path,
                "error": str(e)
            })
            if not args.continue_on_error:
                # If strict error handling, abort immediately!
                print("\n❌ Pipeline aborted due to failure. (Use --continue-on-error to skip failures)")
                sys.exit(1)
                
    run_metadata["end_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_metadata["success_count"] = success_count
    run_metadata["failure_count"] = failure_count
    
    # Save provenance metadata
    prov_path = os.path.join(args.output_dir, "provenance.json")
    with open(prov_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
        
    print("\n" + "=" * 65)
    print(f"🏁 RUN COMPLETE: {success_count} succeeded, {failure_count} failed")
    print(f"Provenance saved to: {prov_path}")
    print("=" * 65)
    
    if failure_count > 0:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PG-AMCD Signal Processing CLI Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run signal processing pipeline on a dataset")
    run_parser.add_argument("--input-dir", required=True, help="Path to Vibration - ML raw data directory")
    run_parser.add_argument("--metadata", required=False, help="Path to combination spreadsheet")
    run_parser.add_argument("--output-dir", required=True, help="Path to output processed results")
    run_parser.add_argument("--config", required=False, help="Path to config.json file")
    run_parser.add_argument("--continue-on-error", action="store_true", help="Continue processing on file failure")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_pipeline_on_dataset(args)

if __name__ == "__main__":
    main()
