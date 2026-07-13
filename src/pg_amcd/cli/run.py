"""Implementation of the ``pg-amcd run`` command."""

import glob
import json
import os
import sys
import time

# Set non-interactive matplotlib backend before any pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.pipeline import process_recording
from pg_amcd.provenance import compute_file_sha256, is_output_stale, compute_run_id
from pg_amcd.validation import validate_decomposition
from pg_amcd.cli.utils import (
    get_case_insensitive,
    get_environment_info,
    get_git_commit_sha,
    _git_is_dirty,
    _load_metadata_index,
    _sha256_of_config,
)


def run_pipeline_on_dataset(args):
    """Run the PG-AMCD pipeline on every MAT file in ``args.input_dir``."""
    config = load_pipeline_config(args.config)
    fs = config["sampling_rate"]

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    mat_files = glob.glob(os.path.join(args.input_dir, "**/*.mat"), recursive=True)
    mat_files = [f for f in mat_files if not f.endswith("combinations.xlsx") and "~lock" not in f]

    if not mat_files:
        print(f"No MAT files found in: {args.input_dir}")
        sys.exit(0)

    use_physics = config.get("use_physics_gating", True)
    if use_physics and not args.metadata:
        print(
            "Error: Physics gating is enabled (use_physics_gating=true), "
            "but no metadata file was provided via --metadata."
        )
        sys.exit(1)

    meta_index = {}
    if args.metadata:
        if not os.path.exists(args.metadata):
            print(f"Error: Metadata file does not exist: {args.metadata}")
            sys.exit(1)
        meta_rows = _load_metadata_index(args.metadata)

        def _meta_key(r):
            return os.path.basename(str(r.get("file_path") or r.get("recording_id") or ""))

        for _row in meta_rows:
            _k = _meta_key(_row)
            if _k not in meta_index:
                meta_index[_k] = _row

    print("=" * 65)
    print(f"🚀 Running PG-AMCD CLI Pipeline on {len(mat_files)} files 🚀")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    _config_sha = _sha256_of_config(config, args.config)
    _git_commit = get_git_commit_sha()
    _input_checksums = [compute_file_sha256(f) for f in mat_files]
    run_id = compute_run_id(_config_sha, _git_commit, _input_checksums)
    run_output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run output dir: {run_output_dir}")

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
        "failures": [],
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
            target_folder = os.path.join(run_output_dir, folder_name)
            npz_path = os.path.join(target_folder, base_name.replace(".mat", "_IMFs.npz"))
            maiw_path = os.path.join(target_folder, base_name)
            clean_path = os.path.join(target_folder, base_name.replace(".mat", "_Clean.mat"))

            if not is_output_stale(raw_path, [npz_path, maiw_path, clean_path]):
                print(f"⏭ Skipping {rel_path} (outputs up to date)")
                run_metadata["files_processed"].append(
                    {
                        "path": rel_path,
                        "sha256": compute_file_sha256(raw_path),
                        "status": "skipped_up_to_date",
                    }
                )
                continue

            row = meta_index.get(base_name)
            if use_physics and not row:
                raise ValueError(
                    f"Recording {base_name} has no matching metadata row, "
                    "required for physics gating."
                )

            metadata_dict = None
            if row:
                if use_physics:
                    rpm_raw = get_case_insensitive(row, ["rpm", "RPM"])
                    if rpm_raw is None or str(rpm_raw).strip() == "":
                        raise ValueError(
                            f"Recording {base_name} metadata row is missing required 'rpm' field."
                        )
                    tc_raw = get_case_insensitive(row, ["tooth_count", "toothcount", "ToothCount"])
                    if tc_raw is None or str(tc_raw).strip() == "":
                        raise ValueError(
                            f"Recording {base_name} metadata row is missing required 'tooth_count' field."
                        )

                try:
                    rpm = float(get_case_insensitive(row, ["rpm", "RPM"], 570.0))
                except ValueError:
                    rpm = 570.0

                try:
                    tooth_count = int(get_case_insensitive(row, ["tooth_count", "toothcount", "ToothCount"], 1))
                except ValueError:
                    tooth_count = 1

                metadata_dict = {
                    "rpm": rpm,
                    "tooth_count": tooth_count,
                    "stickout": get_case_insensitive(row, ["stickout", "Stickout"]),
                    "depth_of_cut": get_case_insensitive(
                        row,
                        ["depth_of_cut", "depthofcut", "depth_of_cut_mm", "DepthOfCut"],
                    ),
                    "recording_id": get_case_insensitive(row, ["recording_id", "recordingid", "RecordingID"]),
                    "label": get_case_insensitive(row, ["label", "Label"]),
                }

            t_arr, sig_arr, fs_est = validate_and_load_signal(
                raw_path,
                configured_fs=fs,
                tolerance=0.05,
                min_duration_seconds=1.0,
            )

            file_sha256 = compute_file_sha256(raw_path)

            res = process_recording(t_arr, sig_arr, config, metadata=metadata_dict, mode="exploratory")

            os.makedirs(target_folder, exist_ok=True)
            win_res = res.window_results[0]

            np.savez_compressed(
                npz_path,
                time=win_res.time_segment,
                original_signal=res.scaled_preprocessed_signal[win_res.start_idx : win_res.end_idx],
                imfs=win_res.imfs,
                start_index=win_res.start_idx,
            )

            scipy.io.savemat(maiw_path, {"tsDS": np.column_stack((win_res.time_segment, win_res.maiw_reconstructed))})

            scipy.io.savemat(
                clean_path,
                {"tsDS": np.column_stack((win_res.time_segment, win_res.denoised_clean))},
            )

            num_imfs = win_res.imfs.shape[0]
            plt.figure(figsize=(12, 1.5 * (num_imfs + 1)))
            plt.subplot(num_imfs + 1, 1, 1)
            plt.plot(win_res.time_segment, win_res.maiw_reconstructed, color="black")
            plt.title(f"CEEMDAN & Denoised Decomposition - {base_name}")
            plt.ylabel("Weighted Reconstruct")
            for i in range(num_imfs):
                plt.subplot(num_imfs + 1, 1, i + 2)
                plt.plot(
                    win_res.time_segment,
                    win_res.imfs[i],
                    color="blue" if i < num_imfs - 1 else "red",
                )
                plt.ylabel(f"IMF {i + 1}" if i < num_imfs - 1 else "Residual")
            plt.xlabel("Time (seconds)")
            plt.tight_layout()
            plot_path = os.path.join(target_folder, base_name.replace(".mat", "_IMFs_plot.png"))
            plt.savefig(plot_path, dpi=120)
            plt.close()

            run_metadata["per_file_runtime"][rel_path] = time.time() - file_start
            output_checksums = {
                "imfs_npz": compute_file_sha256(npz_path),
                "maiw_mat": compute_file_sha256(maiw_path),
                "clean_mat": compute_file_sha256(clean_path),
            }
            success_count += 1

            validation_metrics = validate_decomposition(
                res.scaled_preprocessed_signal[win_res.start_idx : win_res.end_idx],
                win_res.imfs,
                fs,
            )
            run_metadata["files_processed"].append(
                {
                    "path": rel_path,
                    "sha256": file_sha256,
                    "cutoff": res.selected_parameters["cutoff_frequency"],
                    "diagnostics": win_res.features,
                    "validation": validation_metrics,
                    "output_checksums": output_checksums,
                }
            )
            print(
                f"✅ Success: NRMSE={win_res.features['nrmse']:.2e}, "
                f"MMI={win_res.features['mmi']:.4f}, OI={win_res.features['oi']:.4f}"
            )

        except Exception as e:
            print(f"❌ Failed: {e}")
            failure_count += 1
            run_metadata["failures"].append({"path": rel_path, "error": str(e)})
            if not args.continue_on_error:
                print(
                    "\n❌ Pipeline aborted due to failure. "
                    "(Use --continue-on-error to skip failures)"
                )
                sys.exit(1)

    run_metadata["run_id"] = run_id
    run_metadata["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_metadata["total_runtime"] = time.time() - run_start
    run_metadata["success_count"] = success_count
    run_metadata["failure_count"] = failure_count

    prov_path = os.path.join(run_output_dir, "provenance.json")
    with open(prov_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    print("\n" + "=" * 65)
    print(f"🏁 RUN COMPLETE: {success_count} succeeded, {failure_count} failed")
    print(f"Provenance saved to: {prov_path}")
    print("=" * 65)

    if failure_count > 0:
        sys.exit(1)
