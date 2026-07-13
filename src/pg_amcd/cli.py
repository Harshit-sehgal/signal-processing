import os
import sys
import argparse
import glob
import time
import json
import hashlib
import subprocess
import csv
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local pg_amcd modules
from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.pipeline import process_recording
from pg_amcd.models import PipelineResult
from pg_amcd.validation import validate_decomposition

from pg_amcd.provenance import compute_file_sha256, is_output_stale, compute_run_id

def get_git_commit_sha() -> str:
    """Retrieves the current git commit SHA of the repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "Unknown"


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def _sha256_of_config(config: dict, config_path: str) -> str:
    """SHA-256 of the resolved configuration (Goal 4.3)."""
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    if config_path and os.path.exists(config_path):
        h.update(compute_file_sha256(config_path).encode("utf-8"))
    return h.hexdigest()

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
    # Goal 4.4: deterministic run id from stable inputs (config + git + file
    # contents), known before any CEEMDAN work. Outputs are written under
    # <output_dir>/<run_id>/ and never reused unless the id matches.
    _config_sha = _sha256_of_config(config, args.config)
    _git_commit = get_git_commit_sha()
    _input_checksums = [compute_file_sha256(f) for f in mat_files]
    run_id = compute_run_id(_config_sha, _git_commit, _input_checksums)
    run_output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run output dir: {run_output_dir}")
    
    # Collect run metadata
    run_start = time.time()
    run_metadata = {
        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_line": " ".join(sys.argv),
        "git_commit": get_git_commit_sha(),
        "git_dirty": _git_is_dirty(),
        "config_sha256": _sha256_of_config(config, args.config),
        "random_seeds": config.get("ceemdan", {}),
        "env_info": get_environment_info(),
        "resolved_config": config,
        "per_file_runtime": {},
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
            file_start = time.time()
            # 0. Stale-output detection: skip if all outputs exist and are
            #    newer than the input (Sprint 3 reproducibility requirement).
            target_folder = os.path.join(run_output_dir, folder_name)
            npz_path = os.path.join(target_folder, base_name.replace(".mat", "_IMFs.npz"))
            maiw_path = os.path.join(target_folder, base_name)
            clean_path = os.path.join(target_folder, base_name.replace(".mat", "_Clean.mat"))
            if not is_output_stale(raw_path, [npz_path, maiw_path, clean_path]):
                print(f"⏭ Skipping {rel_path} (outputs up to date)")
                continue

            # A. Validate and load signal (Strict Goal 3 Input Checks)
            t_arr, sig_arr, fs_est = validate_and_load_signal(
                raw_path,
                configured_fs=fs,
                tolerance=0.05,
                min_duration_seconds=1.0
            )

            # Compute file checksum (SHA-256 provenance)
            file_sha256 = compute_file_sha256(raw_path)
            
            # B. Run the canonical pipeline
            res: PipelineResult = process_recording(t_arr, sig_arr, config, mode="exploratory")
            
            # Prepare file output directories
            target_folder = os.path.join(run_output_dir, folder_name)
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
            
            run_metadata["per_file_runtime"][rel_path] = time.time() - file_start
            output_checksums = {
                "imfs_npz": compute_file_sha256(npz_path),
                "maiw_mat": compute_file_sha256(maiw_path),
                "clean_mat": compute_file_sha256(clean_mat_path),
            }
            success_count += 1
            validation_metrics = validate_decomposition(
                res.scaled_preprocessed_signal[win_res.start_idx:win_res.end_idx],
                win_res.imfs,
                fs,
            )
            run_metadata["files_processed"].append({
                "path": rel_path,
                "sha256": file_sha256,
                "cutoff": res.selected_parameters["cutoff_frequency"],
                "diagnostics": win_res.features,
                "validation": validation_metrics,
                "output_checksums": output_checksums,
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
                
    # run_id was computed from all input file checksums before processing, so
    # the provenance record matches the on-disk run directory exactly.
    run_metadata["run_id"] = run_id
    run_metadata["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_metadata["total_runtime"] = time.time() - run_start
    run_metadata["success_count"] = success_count
    run_metadata["failure_count"] = failure_count
    
    # Save provenance metadata
    prov_path = os.path.join(run_output_dir, "provenance.json")
    with open(prov_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
        
    print("\n" + "=" * 65)
    print(f"🏁 RUN COMPLETE: {success_count} succeeded, {failure_count} failed")
    print(f"Provenance saved to: {prov_path}")
    print("=" * 65)
    
    if failure_count > 0:
        sys.exit(1)




def _load_metadata_index(path):
    """Load a dataset metadata spreadsheet (CSV or XLSX) into a list of row dicts."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading Excel metadata requires pandas+openpyxl; supply a CSV instead."
            ) from exc
        df = pd.read_excel(path)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    raise ValueError(f"Unsupported metadata format: {ext}")
def run_validation_on_dataset(args):
    """Validate a dataset's raw signals against the configured input contract.

    Implements the ``pg-amcd validate`` command (Goal 3: strict input
    validation surfaced as a first-class, non-destructive operation).
    """
    config = load_pipeline_config(args.config)
    fs = config["sampling_rate"]
    vcfg = config.get("validation", {})
    tolerance = vcfg.get("tolerance", 0.05)
    min_duration = vcfg.get("min_duration_seconds", 1.0)
    meta_index = {}
    duplicate_meta = 0
    missing_meta = 0
    missing_label = 0
    sampling_mismatch = 0
    if getattr(args, "metadata", None):
        if not os.path.exists(args.metadata):
            print(f"Error: Metadata file does not exist: {args.metadata}")
            sys.exit(1)
        meta_rows = _load_metadata_index(args.metadata)
        def _meta_key(r):
            return os.path.basename(str(r.get("file_path") or r.get("recording_id") or ""))
        for _row in meta_rows:
            _k = _meta_key(_row)
            if _k in meta_index:
                duplicate_meta += 1
            else:
                meta_index[_k] = _row
        missing_label = sum(1 for _r in meta_rows if not str(_r.get("label", "") or "").strip())

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    mat_files = glob.glob(os.path.join(args.input_dir, "**/*.mat"), recursive=True)
    mat_files = [f for f in mat_files if not f.endswith("combinations.xlsx") and "~lock" not in f]

    if not mat_files:
        print(f"No MAT files found in: {args.input_dir}")
        sys.exit(0)

    report = {
        "sampling_rate": fs,
        "tolerance": tolerance,
        "min_duration_seconds": min_duration,
        "files": [],
    }
    n_valid = 0
    print("=" * 65)
    print(f"🔎 Validating {len(mat_files)} files (fs={fs} Hz, tol={tolerance*100:.0f}%, min_dur={min_duration}s)")
    print("=" * 65)
    for raw_path in mat_files:
        rel_path = os.path.relpath(raw_path, args.input_dir)
        entry = {"path": rel_path, "valid": False}
        base = os.path.basename(raw_path)
        if meta_index and base not in meta_index:
            missing_meta += 1
        try:
            t_arr, sig_arr, fs_est = validate_and_load_signal(
                raw_path, configured_fs=fs, tolerance=tolerance, min_duration_seconds=min_duration
            )
            entry["valid"] = True
            entry["fs_estimated"] = float(fs_est)
            entry["n_samples"] = int(len(sig_arr))
            entry["duration_seconds"] = float(len(sig_arr) / fs_est)
            n_valid += 1
            print(f"✅ {rel_path}  fs={fs_est:.1f}Hz  N={len(sig_arr)}  dur={entry['duration_seconds']:.3f}s")
        except Exception as e:
            emsg = str(e).lower()
            if "sampling rate" in emsg or "deviates" in emsg:
                sampling_mismatch += 1
            entry["error"] = str(e)
            print(f"❌ {rel_path}  {e}")
        report["files"].append(entry)

    report["n_files"] = len(mat_files)
    report["n_valid"] = n_valid
    report["n_invalid"] = len(mat_files) - n_valid
    if meta_index:
        report["metadata"] = {
            "n_rows": len(meta_index) + duplicate_meta,
            "missing_metadata": missing_meta,
            "duplicate_metadata_entries": duplicate_meta,
            "missing_chatter_label": missing_label,
            "sampling_rate_mismatches": sampling_mismatch,
        }
        print("\n--- Metadata validation ---")
        print(f"Files discovered:           {len(mat_files)}")
        print(f"Valid:                      {n_valid}")
        print(f"Invalid:                    {report['n_invalid']}")
        print(f"Missing metadata:           {missing_meta}")
        print(f"Duplicate metadata entries: {duplicate_meta}")
        print(f"Missing chatter label:      {missing_label}")
        print(f"Sampling-rate mismatches:   {sampling_mismatch}")

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nValidation report written to: {args.output}")

    print("\n" + "=" * 65)
    print(f"🏁 VALIDATION: {n_valid}/{len(mat_files)} files passed")
    print("=" * 65)
    if report["n_invalid"] > 0:
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

    # 'validate' subcommand
    validate_parser = subparsers.add_parser(
        "validate", help="Validate raw signals against the input contract without processing"
    )
    validate_parser.add_argument("--input-dir", required=True, help="Path to Vibration - ML raw data directory")
    validate_parser.add_argument("--config", required=False, help="Path to config.json file")
    validate_parser.add_argument("--output", required=False, help="Path to write the JSON validation report")
    validate_parser.add_argument("--metadata", required=False, help="Path to metadata CSV/XLSX for dataset validation reporting")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_pipeline_on_dataset(args)
    elif args.command == "validate":
        run_validation_on_dataset(args)

if __name__ == "__main__":
    main()
