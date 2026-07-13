"""Generate a Markdown evaluation report (Segment 7).

Reads ``outputs/evaluation_results.json`` (produced by
``scripts/evaluate_dataset.py``) and emits a human-readable Markdown table of
per-model detection metrics. Degrades gracefully when no results are present.
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD report generator")
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "outputs", "evaluation_results.json"),
    )
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "outputs", "evaluation_report.md")
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(
            f"No evaluation results at {args.input}.\n"
            "Run `make evaluate` (or scripts/evaluate_dataset.py) on a labelled "
            "dataset first."
        )
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    metric_keys = [
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    lines = ["# PG-AMCD Evaluation Report", ""]
    lines.append(
        "Metrics are computed with leakage-proof GroupKFold on recording id. "
        "Raw metrics use direct window probabilities; temporal-smoothed metrics "
        "apply hysteresis (Goal 6.5)."
    )
    lines.append("")

    for section, key in (
        ("Raw window probabilities", "raw_metrics"),
        ("Temporal-smoothed (hysteresis)", "temporal_smoothed_metrics"),
    ):
        lines.append(f"## {section}")
        lines.append("")
        header = "| Model | " + " | ".join(metric_keys) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(metric_keys) + 1)) + "|")
        for model, payload in data.items():
            metrics = payload.get(key, {})
            row = "| " + model + " | " + " | ".join(
                _fmt(metrics.get(mk)) for mk in metric_keys
            ) + " |"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
