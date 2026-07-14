"""Implementation of the ``pg-amcd evaluate`` command."""

import json
import os
import sys

from pg_amcd.config import load_pipeline_config
from pg_amcd.evaluation import (
    evaluate_directory,
    evaluate_real_dataset,
    evaluate_real_dataset_temporal,
)


def run_evaluation_on_dataset(args):
    """Evaluate chatter detection on a dataset and write results to JSON.

    Supports three modes:

    * Raw MAT files (default): ``--input-dir`` -> ``evaluate_real_dataset``
    * Raw MAT files with temporal smoothing: ``--input-dir --temporal`` ->
      ``evaluate_real_dataset_temporal``
    * Precomputed NPZ artifacts: ``--npz-dir`` -> ``evaluate_directory``

    The output JSON contains per-model metrics, calibration info, and (for
    real-data modes) feature ablations / cross-condition splits.
    """
    input_dir = getattr(args, "input_dir", None)
    npz_dir = getattr(args, "npz_dir", None)
    output_path = getattr(args, "output", None)
    temporal = getattr(args, "temporal", False)
    label_filter = getattr(args, "label_filter", "c,i,s")
    random_state = getattr(args, "random_state", 42)

    if not output_path:
        print("Error: --output is required for evaluate command.")
        sys.exit(1)

    if input_dir and npz_dir:
        print("Error: --input-dir and --npz-dir are mutually exclusive.")
        sys.exit(1)

    if not input_dir and not npz_dir:
        print("Error: Either --input-dir or --npz-dir must be provided.")
        sys.exit(1)

    if npz_dir and temporal:
        print("Error: --temporal is not supported with --npz-dir.")
        sys.exit(1)

    config = load_pipeline_config(getattr(args, "config", None))
    label_filter_tuple = tuple(label_filter.split(","))

    try:
        if npz_dir:
            fs = getattr(args, "fs", 10_000.0)
            rpm = getattr(args, "rpm", 600.0)
            tooth_count = getattr(args, "tooth_count", 1)
            results = evaluate_directory(
                npz_dir,
                fs=fs,
                rpm=rpm,
                tooth_count=tooth_count,
                random_state=random_state,
            )
        elif temporal:
            results = evaluate_real_dataset_temporal(
                input_dir,
                config,
                label_filter=label_filter_tuple,
                random_state=random_state,
            )
        else:
            results = evaluate_real_dataset(
                input_dir,
                config,
                label_filter=label_filter_tuple,
                random_state=random_state,
            )
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"Error during evaluation: {exc}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"✅ Evaluation complete. Results written to: {output_path}")
