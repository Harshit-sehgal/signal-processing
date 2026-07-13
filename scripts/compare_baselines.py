"""Synthetic denoising baseline comparison (Segment 5 / Goal 5.4-5.6).

Runs :func:`pg_amcd.baselines.benchmark_denoising` (no real dataset needed) and
reports the full proposed pipeline against the required baselines:

    raw, butterworth_only, wavelet_only, ceemdan_only, ceemdan_simple_selection,
    current_maiw, full_proposed, stft_baseline

Writes ``outputs/baselines/results.json`` and a Markdown summary. This is the
Segment 5 mathematical acceptance check: the proposed pipeline must outperform
the baselines on chatter reconstruction.
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pg_amcd.baselines import benchmark_denoising, METHODS, METRIC_KEYS  # noqa: E402

CHEAP_CEEMDAN = {
    "trials": 2,
    "epsilon": 0.02,
    "noise_seed": 42,
    "sifting_iterations": 2,
    "search_cutoffs": [100.0],
    "search_seeds": 1,
}


def _format_row(name, m):
    return "  {:<26} {:>9.4f} {:>8.2f} {:>10.3f} {:>12.3f} {:>10.3f} {:>10.3f}".format(
        name,
        m["rmse"],
        m["snr_db"],
        m["spectral_distortion"],
        m["chatter_band_retention"],
        m["noise_band_attenuation"],
        m["onset_detection_error"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run denoising baseline comparison")
    parser.add_argument("--n-signals", type=int, default=5)
    parser.add_argument("--fs", type=float, default=10_000.0)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--out", type=str, default=os.path.join(ROOT, "outputs", "baselines"))
    args = parser.parse_args()

    agg = benchmark_denoising(
        n_signals=args.n_signals,
        fs=args.fs,
        duration=args.duration,
        seed=args.seed,
        snr_db=args.snr_db,
        ceemdan_cfg=CHEAP_CEEMDAN,
    )

    header = "  {:<26} {:>9} {:>8} {:>10} {:>12} {:>10} {:>10}".format(
        "method", "rmse", "snr_db", "spec_dist", "chatter_ret", "noise_att", "onset_err"
    )
    print("\nDenoising baseline comparison (Segment 5)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in METHODS:
        print(_format_row(m, agg[m]))

    best = min(METHODS, key=lambda m: agg[m]["rmse"])
    print("\nLowest chatter-reconstruction RMSE: {} ({:.4f})".format(best, agg[best]["rmse"]))
    if best != "full_proposed":
        print("WARNING: full_proposed is not the best method on RMSE.")

    os.makedirs(args.out, exist_ok=True)
    results = {
        "config": {
            "n_signals": args.n_signals,
            "fs": args.fs,
            "duration": args.duration,
            "seed": args.seed,
            "snr_db": args.snr_db,
            "ceemdan_cfg": CHEAP_CEEMDAN,
        },
        "methods": METHODS,
        "metric_keys": METRIC_KEYS,
        "aggregated": agg,
        "best_rmse_method": best,
    }
    json_path = os.path.join(args.out, "results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote {}".format(json_path))

    md_path = os.path.join(args.out, "README.md")
    lines = ["# Denoising Baseline Comparison (Segment 5)", ""]
    lines.append("Config: n_signals={}, fs={}, duration={}, seed={}, snr_db={}".format(
        args.n_signals, args.fs, args.duration, args.seed, args.snr_db))
    lines.append("")
    lines.append("| method | rmse | snr_db | spec_dist | chatter_ret | noise_att | onset_err |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in METHODS:
        a = agg[m]
        lines.append("| {} | {:.4f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            m, a["rmse"], a["snr_db"], a["spectral_distortion"],
            a["chatter_band_retention"], a["noise_band_attenuation"], a["onset_detection_error"]))
    lines.append("")
    lines.append("Best method by chatter-reconstruction RMSE: **{}**.".format(best))
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    print("Wrote {}".format(md_path))


if __name__ == "__main__":
    main()
