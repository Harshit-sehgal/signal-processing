"""Statistical analysis for the chatter-detection evaluation (Segment 7).

Reads ``outputs/evaluation_results.json`` (out-of-fold predictions from the
leave-one-recording-out split) and reports:

* Bootstrap 95% confidence intervals for the strongest model's ROC-AUC and
  balanced accuracy (resampling the OOF predictions with replacement),
* a McNemar test comparing the two strongest models' predictions, to check
  whether their error patterns differ significantly.

Writes ``outputs/statistics.json`` and a Markdown summary. Degrades gracefully
when OOF predictions are absent.
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy import stats

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _auc(x, y):
    order = np.argsort(x)
    return float(_trapz(y[order], x[order]))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bootstrap_ci(y_true, proba, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= 0.5).astype(int)
    n = len(y_true)
    aucs, bas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr, pd = y_true[idx], proba[idx], pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        # balanced accuracy
        bas.append(_balanced_acc(yt, pd))
        # roc-auc (only if both classes present)
        order = np.argsort(pr)[::-1]
        tp = np.cumsum(yt[order] == 1)
        fp = np.cumsum(yt[order] == 0)
        if fp[-1] == 0 or tp[-1] == 0:
            continue
        tp = np.concatenate([[0], tp])
        fp = np.concatenate([[0], fp])
        aucs.append(_auc(fp / fp[-1], tp / tp[-1]))
    return {
        "roc_auc_95ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "balanced_accuracy_95ci": [float(np.percentile(bas, 2.5)), float(np.percentile(bas, 97.5))],
        "n_bootstrap": len(aucs),
    }


def _balanced_acc(y_true, pred):
    sens = np.mean(pred[y_true == 1] == 1) if np.any(y_true == 1) else 0.0
    spec = np.mean(pred[y_true == 0] == 0) if np.any(y_true == 0) else 0.0
    return 0.5 * (sens + spec)


def _mcnemar(y_true, pred_a, pred_b):
    """McNemar test on two models' predictions; returns chi2 and p-value."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    # McNemar with continuity correction.
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": p}


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD statistics")
    parser.add_argument(
        "--input", default=os.path.join(ROOT, "outputs", "evaluation_results.json")
    )
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "outputs", "statistics.json")
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"No evaluation results at {args.input}.")
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    oof = data.get("oof", {})
    y_true = oof.get("y_true")
    proba = oof.get("proba", {})
    if not y_true or not proba:
        print("No out-of-fold predictions in results; run evaluation first.")
        return 1

    y_true = np.asarray(y_true, dtype=int)
    # Rank models by ROC-AUC from the stored per-model metrics.
    loi = data.get("leave_one_recording_out", {})
    ranked = sorted(
        ((m, v.get("roc_auc", -1.0)) for m, v in loi.items() if v),
        key=lambda kv: kv[1], reverse=True,
    )
    best, second = ranked[0][0], (ranked[1][0] if len(ranked) > 1 else None)

    best_proba = np.asarray(proba[best], dtype=float)
    ci = _bootstrap_ci(y_true, best_proba)
    result = {
        "best_model": best,
        "bootstrap": ci,
    }
    lines = ["# PG-AMCD Detection Statistics", ""]
    lines.append(f"Best model (by ROC-AUC): **{best}**")
    lines.append("")
    lines.append("## Bootstrap 95% confidence intervals (leave-one-recording-out)")
    lines.append("")
    lines.append(f"* ROC-AUC: [{ci['roc_auc_95ci'][0]:.3f}, "
                 f"{ci['roc_auc_95ci'][1]:.3f}]")
    lines.append(f"* Balanced accuracy: [{ci['balanced_accuracy_95ci'][0]:.3f}, "
                 f"{ci['balanced_accuracy_95ci'][1]:.3f}]")
    lines.append(f"  (n_bootstrap = {ci['n_bootstrap']})")
    lines.append("")

    if second:
        pred_best = (best_proba >= 0.5).astype(int)
        pred_second = (np.asarray(proba[second], dtype=float) >= 0.5).astype(int)
        mc = _mcnemar(y_true, pred_best, pred_second)
        result["mcnemar"] = {"vs_model": second, **mc}
        lines.append("## McNemar test (best vs second-best)")
        lines.append("")
        lines.append(f"* {best} vs {second}: chi2={mc['chi2']:.3f}, "
                     f"p={mc['p_value']:.4f} (b={mc['b']}, c={mc['c']})")
        sig = "significant" if mc["p_value"] < 0.05 else "not significant"
        lines.append(f"  Difference in errors is **{sig}** at alpha=0.05.")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    md_path = os.path.join(ROOT, "outputs", "statistics_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote statistics to {args.output}")
    print(f"Wrote statistics report to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
