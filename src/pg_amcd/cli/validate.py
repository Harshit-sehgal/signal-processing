"""Implementation of the ``pg-amcd validate`` command."""

import glob
import json
import os
import sys

from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.cli.utils import get_case_insensitive, _load_metadata_index


def run_validation_on_dataset(args):
    """Validate a dataset's raw signals against the configured input contract."""
    config = load_pipeline_config(args.config)
    fs = config["sampling_rate"]
    vcfg = config.get("validation", {})
    tolerance = vcfg.get("tolerance", 0.05)
    min_duration = vcfg.get("min_duration_seconds", 1.0)

    use_physics = config.get("use_physics_gating", True)
    if use_physics and not getattr(args, "metadata", None):
        print("Error: Physics gating is enabled in configuration, but no metadata file was provided.")
        sys.exit(1)

    meta_index = {}
    duplicate_meta = 0
    missing_meta = 0
    missing_label = 0
    invalid_rpm = 0
    invalid_tooth = 0
    metadata_row_no_file = 0
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
            if not _k:
                continue
            if _k in meta_index:
                duplicate_meta += 1
            else:
                meta_index[_k] = _row

            lbl = get_case_insensitive(_row, ["label"])
            if lbl is None or str(lbl).strip() == "":
                missing_label += 1

            rpm_val = get_case_insensitive(_row, ["rpm"])
            if rpm_val is None or str(rpm_val).strip() == "":
                invalid_rpm += 1
            else:
                try:
                    if float(rpm_val) <= 0:
                        invalid_rpm += 1
                except ValueError:
                    invalid_rpm += 1

            tc_val = get_case_insensitive(_row, ["tooth_count", "toothcount", "ToothCount"])
            if tc_val is None or str(tc_val).strip() == "":
                invalid_tooth += 1
            else:
                try:
                    if int(tc_val) < 1:
                        invalid_tooth += 1
                except ValueError:
                    invalid_tooth += 1

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
    discovered_files = set()
    print("=" * 65)
    print(
        f"🔎 Validating {len(mat_files)} files (fs={fs} Hz, "
        f"tol={tolerance * 100:.0f}%, min_dur={min_duration}s)"
    )
    print("=" * 65)

    for raw_path in mat_files:
        rel_path = os.path.relpath(raw_path, args.input_dir)
        entry = {"path": rel_path, "valid": False}
        base = os.path.basename(raw_path)
        discovered_files.add(base)

        if getattr(args, "metadata", None) and base not in meta_index:
            missing_meta += 1

        try:
            t_arr, sig_arr, fs_est = validate_and_load_signal(
                raw_path,
                configured_fs=fs,
                tolerance=tolerance,
                min_duration_seconds=min_duration,
            )
            entry["valid"] = True
            entry["fs_estimated"] = float(fs_est)
            entry["n_samples"] = int(len(sig_arr))
            entry["duration_seconds"] = float(len(sig_arr) / fs_est)
            n_valid += 1
            print(
                f"✅ {rel_path}  fs={fs_est:.1f}Hz  N={len(sig_arr)}  "
                f"dur={entry['duration_seconds']:.3f}s"
            )
        except Exception as e:
            emsg = str(e).lower()
            if "sampling rate" in emsg or "deviates" in emsg:
                sampling_mismatch += 1
            entry["error"] = str(e)
            print(f"❌ {rel_path}  {e}")
        report["files"].append(entry)

    if getattr(args, "metadata", None):
        for _k in meta_index:
            if _k not in discovered_files:
                metadata_row_no_file += 1

    report["n_files"] = len(mat_files)
    report["n_valid"] = n_valid
    report["n_invalid"] = len(mat_files) - n_valid

    if getattr(args, "metadata", None):
        report["metadata"] = {
            "n_rows": len(meta_index) + duplicate_meta,
            "missing_metadata": missing_meta,
            "duplicate_metadata_entries": duplicate_meta,
            "missing_chatter_label": missing_label,
            "invalid_rpm_values": invalid_rpm,
            "invalid_tooth_values": invalid_tooth,
            "metadata_row_no_file": metadata_row_no_file,
            "sampling_rate_mismatches": sampling_mismatch,
        }
        print("\n--- Metadata validation ---")
        print(f"Files discovered:           {len(mat_files)}")
        print(f"Valid signals:              {n_valid}")
        print(f"Invalid signals:            {report['n_invalid']}")
        print(f"Missing metadata:           {missing_meta}")
        print(f"Duplicate metadata:         {duplicate_meta}")
        print(f"Missing labels:             {missing_label}")
        print(f"Invalid RPM values:         {invalid_rpm}")
        print(f"Invalid tooth values:       {invalid_tooth}")
        print(f"Metadata rows without files: {metadata_row_no_file}")
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

    has_errors = (
        (report["n_invalid"] > 0)
        or (duplicate_meta > 0)
        or (missing_label > 0)
        or (invalid_rpm > 0)
        or (invalid_tooth > 0)
        or (missing_meta > 0)
        or (metadata_row_no_file > 0)
    )
    if has_errors:
        sys.exit(1)
