"""End-to-end ablation and baseline runner harness (P0 #9).

This script combines two independent acceptance checks into one reproducible
entry point:

1. **Synthetic denoising baselines** (Segment 5 / Goal 5.4-5.6): the full
   proposed PG-AMCD pipeline is compared against the eight required baselines
   on synthetic signals with known ground truth.

2. **Real-data feature and config ablations** (Goal 7): the chatter-detection
   performance of the full pipeline is compared against ablated variants:
   - without frequency/spectral features
   - without IMF features
   - time-domain-only baseline
   - physics gating disabled

The script writes a single JSON report and a Markdown summary, plus a bar-chart
visualisation of the ablation results.

Usage:
    python scripts/run_ablation_and_baselines.py
    python scripts/run_ablation_and_baselines.py --mat-dir data/raw_mats --out outputs/ablations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pg_amcd.baselines import benchmark_denoising, METRIC_KEYS, METHODS
from pg_amcd.config import load_pipeline_config
from pg_amcd.evaluation import evaluate_real_dataset
from pg_amcd.visualization import plot_ablation_results

CHEAP_CEEMDAN = {
    "trials": 2,
    "epsilon": 0.02,
    "noise_seed": 42,
    "sifting_iterations": 2,
    "search_cutoffs": [100.0],
    "search_seeds": 1,
}


def run_baselines(n_signals: int, fs: float, duration: float, snr_db: float, seed: int) -> Dict[str, Any]:
    """Run synthetic denoising baselines and return aggregated metrics."""
    agg = benchmark_denoising(
        n_signals=n_signals,
        fs=fs,
        duration=duration,
        seed=seed,
        snr_db=snr_db,
        ceemdan_cfg=CHEAP_CEEMDAN,
    )
    best = min(METHODS, key=lambda m: agg[m]["rmse"])
    return {
        "config": {
            "n_signals": n_signals,
            "fs": fs,
            "duration": duration,
            "snr_db": snr_db,
            "seed": seed,
            "ceemdan_cfg": CHEAP_CEEMDAN,
        },
        "methods": METHODS,
        "metric_keys": METRIC_KEYS,
        "aggregated": agg,
        "best_rmse_method": best,
    }


def _best_auc_metric(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract the best-model ROC-AUC and F1 from an evaluate_real_dataset result."""
    if not result or "leave_one_recording_out" not in result:
        return {}
    loo = result["leave_one_recording_out"]
    if not loo:
        return {}
    best_name = max(loo, key=lambda n: (loo[n] or {}).get("roc_auc", -1.0))
    metrics = loo.get(best_name, {})
    return {
        "best_model": best_name,
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "f1": metrics.get("f1", float("nan")),
        "accuracy": metrics.get("accuracy", float("nan")),
    }


def run_ablations(mat_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run real-data evaluation with the full config and an ablated (no-physics) config."""
    if not mat_dir or not Path(mat_dir).is_dir():
        raise ValueError(f"Real-data directory does not exist: {mat_dir}")

    full = evaluate_real_dataset(mat_dir, config)
    ablated_config = deepcopy(config)
    ablated_config["use_physics_gating"] = False
    no_physics = evaluate_real_dataset(mat_dir, ablated_config)

    return {
        "full": {
            "metrics": _best_auc_metric(full),
            "feature_ablations": full.get("feature_ablations", {}),
        },
        "no_physics_gating": {
            "metrics": _best_auc_metric(no_physics),
            "feature_ablations": no_physics.get("feature_ablations", {}),
        },
    }


def _build_ablation_plot_data(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Prepare data for plot_ablation_results from the real-data ablations."""
    data: Dict[str, Dict[str, float]] = {}
    full_metrics = results.get("full", {}).get("metrics", {})
    data["full_pipeline"] = {
        "roc_auc": float(full_metrics.get("roc_auc", 0.0)),
        "f1": float(full_metrics.get("f1", 0.0)),
    }
    no_physics_metrics = results.get("no_physics_gating", {}).get("metrics", {})
    data["no_physics_gating"] = {
        "roc_auc": float(no_physics_metrics.get("roc_auc", 0.0)),
        "f1": float(no_physics_metrics.get("f1", 0.0)),
    }
    feature_ablations = results.get("full", {}).get("feature_ablations", {})
    for name, split_results in feature_ablations.items():
        loo = split_results.get("leave_one_recording_out", {})
        if not loo:
            continue
        best_name = max(loo, key=lambda n: (loo[n] or {}).get("roc_auc", -1.0))
        metrics = loo.get(best_name, {})
        data[name] = {
            "roc_auc": float(metrics.get("roc_auc", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
        }
    return data


def _write_report(baselines: Dict[str, Any], ablations: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "synthetic_baselines": baselines,
        "real_data_ablations": ablations,
    }
    json_path = out_dir / "ablation_and_baseline_results.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    plot_data = _build_ablation_plot_data(ablations)
    plot_path = str(out_dir / "ablation_summary.png")
    plot_ablation_results(plot_data, output_path=plot_path, metric="roc_auc")

    lines: List[str] = [
        "# Ablation and Baseline Runner Report (P0 #9)",
        "",
        "## Synthetic Denoising Baselines",
        "",
        f"Signals: {baselines['config']['n_signals']} | SNR: {baselines['config']['snr_db']} dB | "
        f"Duration: {baselines['config']['duration']} s | fs: {baselines['config']['fs']} Hz",
        "",
        "| method | rmse | snr_db | chatter_ret | noise_att |",
        "|---|---|---|---|---|",
    ]
    for method in baselines["methods"]:
        m = baselines["aggregated"][method]
        lines.append(
            f"| {method} | {m['rmse']:.4f} | {m['snr_db']:.2f} | "
            f"{m['chatter_band_retention']:.3f} | {m['noise_band_attenuation']:.3f} |"
        )
    lines.append("")
    lines.append(f"Best method by RMSE: **{baselines['best_rmse_method']}**.")
    lines.append("")

    if "error" in ablations:
        lines.extend([
            "## Real-Data Ablation Summary",
            "",
            f"_Real-data ablation failed: {ablations['error']}_",
            "",
        ])
    else:
        full_metrics = ablations.get("full", {}).get("metrics", {})
        no_physics_metrics = ablations.get("no_physics_gating", {}).get("metrics", {})
        lines.extend([
            "## Real-Data Ablation Summary",
            "",
            "| configuration | best_model | roc_auc | f1 | accuracy |",
            "|---|---|---|---|---|",
            f"| full_pipeline | {full_metrics.get('best_model', 'n/a')} | "
            f"{full_metrics.get('roc_auc', float('nan')):.4f} | "
            f"{full_metrics.get('f1', float('nan')):.4f} | "
            f"{full_metrics.get('accuracy', float('nan')):.4f} |",
            f"| no_physics_gating | {no_physics_metrics.get('best_model', 'n/a')} | "
            f"{no_physics_metrics.get('roc_auc', float('nan')):.4f} | "
            f"{no_physics_metrics.get('f1', float('nan')):.4f} | "
            f"{no_physics_metrics.get('accuracy', float('nan')):.4f} |",
            "",
            "### Feature Ablations",
            "",
            "| configuration | roc_auc | f1 |",
            "|---|---|---|",
        ])
        for name, metrics in plot_data.items():
            lines.append(
                f"| {name} | {metrics['roc_auc']:.4f} | {metrics['f1']:.4f} |"
            )
        lines.append("")

    md_path = out_dir / "ablation_and_baseline_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {plot_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end ablation and baseline runner harness")
    parser.add_argument("--mat-dir", type=str, default=None, help="Directory with real .mat recordings")
    parser.add_argument("--out-dir", type=str, default=os.path.join(ROOT, "outputs", "ablations"), help="Output directory")
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "research_fast.json"), help="Pipeline config JSON")
    parser.add_argument("--n-signals", type=int, default=3)
    parser.add_argument("--fs", type=float, default=10_000.0)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-real-data", action="store_true", help="Skip real-data ablations even if --mat-dir is provided")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()

    baselines = run_baselines(
        n_signals=args.n_signals,
        fs=args.fs,
        duration=args.duration,
        snr_db=args.snr_db,
        seed=args.seed,
    )

    ablations: Dict[str, Any] = {"full": {}, "no_physics_gating": {}}
    ablation_failed = False
    if args.mat_dir and not args.skip_real_data:
        try:
            config = load_pipeline_config(args.config)
            ablations = run_ablations(args.mat_dir, config)
        except Exception as exc:
            print(f"Warning: real-data ablation failed: {exc}")
            ablations = {"full": {}, "no_physics_gating": {}, "error": str(exc)}
            ablation_failed = True

    _write_report(baselines, ablations, out_dir)
    return 1 if ablation_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
