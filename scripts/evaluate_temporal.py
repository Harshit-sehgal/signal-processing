"""Multi-window temporal-smoothing evaluation on real labelled data (Goal 6.5).

Runs :func:`pg_amcd.evaluation.evaluate_real_dataset_temporal` over a directory
of raw ``*.mat`` recordings and writes ``outputs/temporal_results.json``. Kept
separate from ``evaluate_dataset.py`` because segmenting each recording into
several windows is materially slower than the single-window evaluation.
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD temporal-smoothing eval")
    parser.add_argument("--mat-dir", default=os.path.join(ROOT, "Vibration_Clean"))
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "research_fast.json"),
    )
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "temporal_results.json"))
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--window-points", type=int, default=2048)
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from pg_amcd.config import load_pipeline_config
    from pg_amcd.evaluation import evaluate_real_dataset_temporal

    cfg = load_pipeline_config(args.config)
    res = evaluate_real_dataset_temporal(
        args.mat_dir, cfg,
        n_windows=args.n_windows, window_points=args.window_points,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    best = res["best_model"]
    rf = res["per_window_metrics"].get("random_forest", {})
    sm = res["smoothed_metrics"].get(best, {})
    print(f"recordings={res['n_recordings']} windows={res['n_windows']} "
          f"chatter={res['label_counts']['chatter']} stable={res['label_counts']['stable']}")
    print(f"best={best} per_window bal_acc={rf.get('balanced_accuracy'):.4f} "
          f"roc_auc={rf.get('roc_auc'):.4f}")
    print(f"smoothed  bal_acc={sm.get('balanced_accuracy'):.4f} "
          f"roc_auc={sm.get('roc_auc'):.4f}")
    print(f"Wrote temporal results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
