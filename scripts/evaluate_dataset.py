"""Dataset evaluation orchestrator (Segment 7 / Goals 6.1-6.5).

Implements the evaluation half of ``make reproduce``:

1. Build the dataset index from raw signals + metadata (Goal 6.1).
2. Extract window-level features per recording (Goal 6.2).
3. Train baseline classifiers with leakage-proof ``GroupKFold`` on
   ``recording_id`` (Goals 6.3-6.4).
4. Evaluate with calibrated detection metrics (Goal 6.5).
5. Write ``outputs/evaluation_results.json``.

This script requires a real labelled chatter dataset. It exits with a clear
message when the dataset directory is absent rather than fabricating results
(there is no placeholder detector in this repository).
"""
import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from pg_amcd.config import load_pipeline_config
from pg_amcd.evaluation import (
    build_dataset_index,
    evaluate_directory,
    evaluate_real_dataset,
)
from pg_amcd.io import validate_and_load_signal
from pg_amcd.pipeline import process_recording
from pg_amcd.features import extract_window_features
from pg_amcd.detection import (
    train_baseline_classifiers,
    predict_window_probabilities,
    evaluate_detector,
    temporal_smooth_probabilities,
)


def _label_to_int(label) -> int:
    if label is None:
        return 0
    s = str(label).strip().lower()
    if s in ("chatter", "1", "true", "yes", "unstable"):
        return 1
    return 0


def _collect_features(index_rows, config, fs):
    """Extract window features for every indexed recording."""
    X, y, groups = [], [], []
    for row in index_rows:
        path = row.get("file_path") or row.get("path")
        if not path or not os.path.exists(path):
            continue
        try:
            t, sig, fs_est = validate_and_load_signal(
                path, configured_fs=fs, tolerance=0.05, min_duration_seconds=1.0
            )
        except Exception as exc:  # skip unreadable / invalid files
            print(f"  skip {path}: {exc}")
            continue
        res = process_recording(t, sig, config, mode="exploratory")
        for wr in res.window_results:
            feats = extract_window_features(
                sig[wr.start_idx : wr.end_idx],
                res.scaled_preprocessed_signal[wr.start_idx : wr.end_idx],
                res.denoised_clean[wr.start_idx : wr.end_idx],
                wr.imfs,
                fs,
                rpm=float(row.get("rpm", 0.0) or 0.0),
                tooth_count=int(row.get("tooth_count", 1) or 1),
            )
            vec = [v for v in feats.values() if isinstance(v, (int, float))]
            X.append(vec)
            y.append(_label_to_int(row.get("label")))
            groups.append(row.get("recording_id") or path)
    return np.array(X, dtype=float), np.array(y, dtype=int), groups


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD dataset evaluation")
    parser.add_argument(
        "--input-dir", default=os.path.join(ROOT, "Vibration - ML"),
        help="Directory of raw tsDS MAT files.",
    )
    parser.add_argument("--metadata", default=None, help="Metadata spreadsheet.")
    parser.add_argument(
        "--npz-dir",
        default=None,
        help="Directory of *_IMFs.npz recordings (real-data mode, filename labels).",
    )
    parser.add_argument(
        "--mat-dir",
        default=None,
        help="Directory of raw *.mat recordings (real-data mode, filename labels).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "outputs", "evaluation_results.json"),
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    # Real-data npz directory mode: uses bundled testing/t1-style artifacts with
    # filename-derived chatter/stable labels; no metadata workbook required.
    if args.npz_dir and os.path.isdir(args.npz_dir):
        print(f"Evaluating real-data npz directory: {args.npz_dir}")
        result = evaluate_directory(args.npz_dir, fs=10_000.0)
        print(
            f"  recordings={result['n_recordings']} "
            f"holdout_balanced_acc={result['holdout_metrics']['balanced_accuracy']:.4f}"
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote evaluation results to {args.output}")
        return 0
    # Real-data raw .mat directory mode: processes every recording through the
    # full PG-AMCD pipeline and evaluates chatter detection with grouped
    # cross-validation (leave-one-recording / stickout / rpm). Labels come from
    # the real <label>_<rpm>_<feed>.mat filename convention; no labels fabricated.
    if args.mat_dir and os.path.isdir(args.mat_dir):
        print(f"Evaluating real-data MAT directory: {args.mat_dir}")
        config_path = args.config or os.path.join(ROOT, "configs", "research_fast.json")
        config = load_pipeline_config(config_path)
        result = evaluate_real_dataset(args.mat_dir, config)
        print(
            f"  recordings={result['n_recordings']} windows={result['n_windows']} "
            f"chatter={result['label_counts']['chatter']} stable={result['label_counts']['stable']}"
        )
        best = result["calibration"].get("best_model")
        if best and best in result["leave_one_recording_out"]:
            m = result["leave_one_recording_out"][best]
            print(
                f"  best={best} LOO balanced_acc={m['balanced_accuracy']:.4f} "
                f"roc_auc={m['roc_auc']:.4f}"
            )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote evaluation results to {args.output}")
        return 0

    if not os.path.isdir(args.input_dir):
        print(
            f"Dataset directory not found: {args.input_dir}\n"
            "Provide a labelled chatter dataset to run evaluation (Segments 6/7)."
        )
        return 1

    config = load_pipeline_config(args.config)
    fs = config["sampling_rate"]

    if args.metadata and os.path.exists(args.metadata):
        index = build_dataset_index(args.input_dir, args.metadata)
    else:
        index = [
            {
                "file_path": f,
                "label": "unknown",
                "recording_id": os.path.basename(f),
            }
            for f in glob.glob(
                os.path.join(args.input_dir, "**", "*.mat"), recursive=True
            )
            if not f.endswith("combinations.xlsx") and "~lock" not in f
        ]

    print(f"Indexed {len(index)} recordings; extracting features...")
    X, y, groups = _collect_features(index, config, fs)
    if X.shape[0] == 0:
        print("No usable feature vectors extracted; cannot train.")
        return 1
    print(f"Feature matrix: {X.shape}, class distribution: {np.bincount(y)}")

    results = train_baseline_classifiers(X, y, groups=np.array(groups))
    evaluation = {}
    for name, payload in results.items():
        model = payload["model"]
        proba = predict_window_probabilities(X, model)
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_detector(y, pred, proba)
        labels, _ = temporal_smooth_probabilities(proba)
        smoothed_metrics = evaluate_detector(y, labels, proba)
        evaluation[name] = {
            "mean_metrics": payload.get("mean_metrics"),
            "raw_metrics": metrics,
            "temporal_smoothed_metrics": smoothed_metrics,
        }
        print(
            f"{name:<20} balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=2)
    print(f"Wrote evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
