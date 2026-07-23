"""Report generation for PG-AMCD runs and run comparisons.

Produces:

* an interactive HTML dashboard,
* a Markdown summary,
* CSV summaries,
* comparison reports between two runs.

All reports read from the JSON/CSV artifacts written by the pipeline so that
the saved files remain the source of truth.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from pg_amcd.tracking import (
    ProjectScorecard,
    calculate_scorecard,
    save_scorecard,
    append_score_history,
)
from pg_amcd.visualization import (
    plot_project_scorecard,
    plot_validation_summary,
    plot_decomposition_metrics,
)


# --------------------------------------------------------------------------- #
# HTML dashboard
# --------------------------------------------------------------------------- #
def _scorecard_html(scores: Dict[str, float]) -> str:
    rows = ""
    for name, value in scores.items():
        label = name.replace("_", " ").title()
        color = "green" if value >= 80 else "orange" if value >= 50 else "red"
        rows += (
            f"<tr><td>{label}</td>"
            f"<td style='color:{color};font-weight:bold'>{value:.1f}</td></tr>"
        )
    return f"<table>{rows}</table>"


def generate_html_dashboard(
    run_dir: str,
    output_path: Optional[str] = None,
    scorecard_path: Optional[str] = None,
    validation_report_path: Optional[str] = None,
    provenance_path: Optional[str] = None,
) -> str:
    """Generate an interactive HTML report for a single run."""
    if output_path is None:
        output_path = os.path.join(run_dir, "report.html")

    scorecard_data: Dict[str, Any] = {}
    if scorecard_path and os.path.exists(scorecard_path):
        with open(scorecard_path, "r", encoding="utf-8") as fh:
            scorecard_data = json.load(fh)

    provenance: Dict[str, Any] = {}
    if provenance_path and os.path.exists(provenance_path):
        with open(provenance_path, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)

    validation: Dict[str, Any] = {}
    if validation_report_path and os.path.exists(validation_report_path):
        with open(validation_report_path, "r", encoding="utf-8") as fh:
            validation = json.load(fh)

    scores = scorecard_data.get("scorecard", {})
    overall = scorecard_data.get("overall", 0.0)
    commit = scorecard_data.get("commit", "unknown")
    timestamp = scorecard_data.get("timestamp", "")

    # Load stage scorecard if it exists to show engineering/scientific split.
    stage_scorecard: Dict[str, Any] = {}
    stage_scorecard_path = os.path.join(run_dir, "stage_scorecard.json")
    if os.path.exists(stage_scorecard_path):
        with open(stage_scorecard_path, "r", encoding="utf-8") as fh:
            stage_scorecard = json.load(fh)

    def _stage_score_html(stage: str) -> str:
        data = stage_scorecard.get(stage, {})
        if not data:
            return ""
        eng = data.get("engineering_score", 0.0)
        sci = data.get("scientific_score", 0.0)
        total = data.get("score", 0.0)
        return (
            f"<tr><td>{stage}</td>"
            f"<td>{total:.1f}</td>"
            f"<td>{eng:.1f}</td>"
            f"<td>{sci:.1f}</td></tr>"
        )

    def _img_tag(rel: str) -> str:
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            return f'<img src="{rel}" alt="{rel}">'
        return f"<p><em>{rel} not yet generated</em></p>"

    stage_score_rows = "".join(_stage_score_html(stage) for stage in ("Stage_1", "Stage_2", "Stage_3", "Stage_4"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PG-AMCD Run Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2em; background: #f8f9fa; }}
        h1 {{ color: #1a5276; }}
        h2 {{ color: #2874a6; border-bottom: 2px solid #2874a6; padding-bottom: .2em; }}
        table {{ border-collapse: collapse; margin: 1em 0; width: 100%; max-width: 800px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #2874a6; color: white; }}
        .metric {{ display: inline-block; width: 23%; margin: 1%; padding: 1em;
                   background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #1a5276; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 1em 0; }}
        .section {{ margin-bottom: 2em; }}
    </style>
</head>
<body>
    <h1>PG-AMCD Run Report</h1>
    <p><strong>Run ID:</strong> {provenance.get('run_id', 'N/A')}</p>
    <p><strong>Git commit:</strong> {commit}</p>
    <p><strong>Timestamp:</strong> {timestamp}</p>

    <div class="section">
        <h2>Overall Score (Pipeline Completeness)</h2>
        <div class="metric">
            <div class="metric-value">{overall:.1f}</div>
            <div>Overall</div>
        </div>
    </div>

    <div class="section">
        <h2>Engineering vs Scientific Scores (per stage)</h2>
        <table>
            <tr><th>Stage</th><th>Completeness</th><th>Engineering</th><th>Scientific</th></tr>
            {stage_score_rows}
        </table>
        {_img_tag('stage_scorecard.png')}
        {_img_tag('stage_progress.png')}
    </div>

    <div class="section">
        <h2>Project Scorecard</h2>
        {_scorecard_html(scores)}
        {_img_tag('project_scorecard.png')}
    </div>

    <div class="section">
        <h2>Dataset Validation</h2>
        <p>Valid files: {validation.get('n_valid', 'N/A')} / {validation.get('n_files', 'N/A')}</p>
        {_img_tag('validation_summary.png')}
    </div>

    <div class="section">
        <h2>Decomposition Metrics</h2>
        <p>Files processed: {provenance.get('success_count', 'N/A')}</p>
        <p>Failures: {provenance.get('failure_count', 'N/A')}</p>
        {_img_tag('decomposition_metrics.png')}
    </div>

    <footer>
        <p><em>Generated by pg-amcd reporting on {datetime.now().isoformat()}</em></p>
    </footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def generate_markdown_report(
    run_dir: str,
    output_path: Optional[str] = None,
    scorecard_path: Optional[str] = None,
    validation_report_path: Optional[str] = None,
    provenance_path: Optional[str] = None,
) -> str:
    """Generate a Markdown summary for a single run."""
    if output_path is None:
        output_path = os.path.join(run_dir, "report.md")

    scorecard_data: Dict[str, Any] = {}
    if scorecard_path and os.path.exists(scorecard_path):
        with open(scorecard_path, "r", encoding="utf-8") as fh:
            scorecard_data = json.load(fh)

    provenance: Dict[str, Any] = {}
    if provenance_path and os.path.exists(provenance_path):
        with open(provenance_path, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)

    validation: Dict[str, Any] = {}
    if validation_report_path and os.path.exists(validation_report_path):
        with open(validation_report_path, "r", encoding="utf-8") as fh:
            validation = json.load(fh)

    scores = scorecard_data.get("scorecard", {})
    overall = scorecard_data.get("overall", 0.0)

    lines = [
        "# PG-AMCD Run Report",
        "",
        f"* **Run ID:** {provenance.get('run_id', 'N/A')}",
        f"* **Git commit:** {scorecard_data.get('commit', 'unknown')}",
        f"* **Timestamp:** {scorecard_data.get('timestamp', '')}",
        f"* **Overall score:** {overall:.1f}",
        "",
        "## Project Scorecard",
        "",
        "| Segment | Score |",
        "| --- | --- |",
    ]
    for name, value in scores.items():
        lines.append(f"| {name.replace('_', ' ').title()} | {value:.1f} |")
    lines.append("")

    lines.extend(
        [
            "## Dataset Validation",
            "",
            f"* Valid files: {validation.get('n_valid', 'N/A')} / {validation.get('n_files', 'N/A')}",
            f"* Invalid files: {validation.get('n_invalid', 'N/A')}",
            "",
            "## Run Summary",
            "",
            f"* Files processed: {provenance.get('success_count', 'N/A')}",
            f"* Failures: {provenance.get('failure_count', 'N/A')}",
            f"* Total runtime (s): {provenance.get('total_runtime', 'N/A')}",
            "",
            "## Figures",
            "",
            "* Project scorecard: `project_scorecard.png`",
            "* Validation summary: `validation_summary.png`",
            "* Decomposition metrics: `decomposition_metrics.png`",
            "",
        ]
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return output_path


# --------------------------------------------------------------------------- #
# CSV summaries
# --------------------------------------------------------------------------- #
def generate_csv_summaries(
    run_dir: str,
    output_dir: Optional[str] = None,
    provenance_path: Optional[str] = None,
) -> Dict[str, str]:
    """Generate CSV summaries from provenance data."""
    if output_dir is None:
        output_dir = os.path.join(run_dir, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    provenance: Dict[str, Any] = {}
    if provenance_path and os.path.exists(provenance_path):
        with open(provenance_path, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)

    paths: Dict[str, str] = {}

    # validation_metrics.csv
    files = provenance.get("files_processed", [])
    validation_path = os.path.join(output_dir, "validation_metrics.csv")
    with open(validation_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "nrmse", "mode_mixing_index", "frequency_ordering_index"])
        for f in files:
            val = f.get("validation", {})
            writer.writerow(
                [
                    f.get("path", ""),
                    val.get("nrmse", ""),
                    val.get("mode_mixing_index", ""),
                    val.get("frequency_ordering_index", ""),
                ]
            )
    paths["validation_metrics"] = validation_path

    # feature_summary.csv
    feature_path = os.path.join(output_dir, "feature_summary.csv")
    with open(feature_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if files:
            diag = files[0].get("diagnostics", {})
            keys = sorted(diag.keys())
            writer.writerow(["path"] + keys)
            for f in files:
                diag = f.get("diagnostics", {})
                writer.writerow([f.get("path", "")] + [diag.get(k, "") for k in keys])
    paths["feature_summary"] = feature_path

    return paths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_run_reports(
    run_dir: str,
    validation_report_path: Optional[str] = None,
    evaluation_results_path: Optional[str] = None,
    baseline_results_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Dict[str, str]:
    """Generate all reports and figures for a single run.

    ``history_path`` controls whether the scorecard is appended to the
    historical record.  Pass ``None`` (the default) to skip this side effect.

    Returns a dict of generated artifact paths.
    """
    provenance_path = os.path.join(run_dir, "provenance.json")
    scorecard_path = os.path.join(run_dir, "project_scorecard.json")

    # Calculate scorecard from available artifacts.
    run_metadata: Dict[str, Any] = {}
    if os.path.exists(provenance_path):
        with open(provenance_path, "r", encoding="utf-8") as fh:
            run_metadata = json.load(fh)

    validation_report: Optional[Dict[str, Any]] = None
    if validation_report_path and os.path.exists(validation_report_path):
        with open(validation_report_path, "r", encoding="utf-8") as fh:
            validation_report = json.load(fh)

    evaluation_results: Optional[Dict[str, Any]] = None
    if evaluation_results_path and os.path.exists(evaluation_results_path):
        with open(evaluation_results_path, "r", encoding="utf-8") as fh:
            evaluation_results = json.load(fh)

    baseline_results: Optional[Dict[str, Any]] = None
    if baseline_results_path and os.path.exists(baseline_results_path):
        with open(baseline_results_path, "r", encoding="utf-8") as fh:
            baseline_results = json.load(fh)

    card, details = calculate_scorecard(
        run_metadata=run_metadata,
        validation_report=validation_report,
        evaluation_results=evaluation_results,
        baseline_results=baseline_results,
        run_dir=run_dir,
    )

    scorecard_path = save_scorecard(card, run_dir, details)
    if history_path is not None:
        append_score_history(card, details, history_path)

    # Figures
    plot_project_scorecard(scorecard_path, os.path.join(run_dir, "project_scorecard.png"))
    if validation_report_path:
        plot_validation_summary(
            validation_report_path, os.path.join(run_dir, "validation_summary.png")
        )
    plot_decomposition_metrics(provenance_path, os.path.join(run_dir, "decomposition_metrics.png"))

    # Reports
    html_path = generate_html_dashboard(
        run_dir,
        scorecard_path=scorecard_path,
        validation_report_path=validation_report_path,
        provenance_path=provenance_path,
    )
    md_path = generate_markdown_report(
        run_dir,
        scorecard_path=scorecard_path,
        validation_report_path=validation_report_path,
        provenance_path=provenance_path,
    )
    csv_paths = generate_csv_summaries(run_dir, provenance_path=provenance_path)

    return {
        "scorecard": scorecard_path,
        "html": html_path,
        "markdown": md_path,
        **csv_paths,
    }


# --------------------------------------------------------------------------- #
# Run comparison
# --------------------------------------------------------------------------- #
def compare_runs(
    baseline_dir: str,
    candidate_dir: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate an HTML comparison between two runs.

    If ``output_path`` is omitted, the file is written to the parent of
    ``candidate_dir`` and the path is printed.
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(candidate_dir) or ".",
            f"comparison_{os.path.basename(baseline_dir)}_vs_{os.path.basename(candidate_dir)}.html",
        )

    def _load(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    base = _load(os.path.join(baseline_dir, "project_scorecard.json"))
    cand = _load(os.path.join(candidate_dir, "project_scorecard.json"))
    base_scores = base.get("scorecard", {})
    cand_scores = cand.get("scorecard", {})

    rows = ""
    field_names = list(ProjectScorecard.__dataclass_fields__.keys())
    for key in field_names:
        b = base_scores.get(key, 0.0)
        c = cand_scores.get(key, 0.0)
        delta = c - b
        color = "green" if delta >= 0 else "red"
        rows += (
            f"<tr><td>{key.replace('_', ' ').title()}</td>"
            f"<td>{b:.1f}</td><td>{c:.1f}</td>"
            f"<td style='color:{color}'>{delta:+.1f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PG-AMCD Run Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2em; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #2874a6; color: white; }}
    </style>
</head>
<body>
    <h1>Run Comparison</h1>
    <p><strong>Baseline:</strong> {baseline_dir}</p>
    <p><strong>Candidate:</strong> {candidate_dir}</p>
    <table>
        <tr><th>Segment</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>
        {rows}
    </table>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote comparison report to: {output_path}")
    return output_path
