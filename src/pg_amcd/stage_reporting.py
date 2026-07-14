"""Manifest-driven Markdown and HTML reports for completed Stage 1--4 runs."""

from __future__ import annotations

import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from pg_amcd.stage_scoring import STAGES, STAGE_METRICS_FILE, generate_stage_scorecard


SECTION_TITLES = (
    "Run overview",
    "Input validation",
    "Preprocessing summary",
    "Stage 1 decomposition",
    "Stage 2 IMF gating",
    "Stage 3 wavelet denoising",
    "Stage 4 feature extraction",
    "Stage scorecard",
    "Warnings and failures",
    "Limitations",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _recording_dirs(stage_dir: Path) -> list[Path]:
    if not stage_dir.is_dir():
        return []
    return sorted(
        path
        for path in stage_dir.iterdir()
        if path.is_dir() and path.name not in {"aggregate", "figures", "report"}
    )


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_scalars(child, path))
    elif isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(item, (Mapping, list)) for item in value):
            result[prefix] = value
        else:
            result[f"{prefix}.count"] = len(value)
    elif value is not None:
        result[prefix] = value
    return result


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _stage_metrics(run_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    return {
        record.name: _read_json(record / STAGE_METRICS_FILE[stage])
        for record in _recording_dirs(run_dir / stage)
    }


def _csv_inventory(run_dir: Path, stage: str) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    stage_dir = run_dir / stage
    if not stage_dir.is_dir():
        return inventory
    for path in sorted(stage_dir.rglob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
        except (OSError, csv.Error):
            header, rows = [], 0
        inventory.append(
            {
                "path": str(path.relative_to(run_dir)),
                "rows": rows,
                "columns": len(header),
            }
        )
    return inventory


def _copy_report_figures(run_dir: Path, figures_dir: Path) -> dict[str, list[str]]:
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True)
    copied: dict[str, list[str]] = {stage: [] for stage in STAGES}
    copied["run"] = []
    candidates: list[tuple[str, Path]] = []
    for stage in STAGES:
        stage_dir = run_dir / stage
        if stage_dir.is_dir():
            candidates.extend(
                (stage, path)
                for path in stage_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".png", ".svg"}
            )
    for name in ("stage_scorecard.png", "stage_progress.png"):
        path = run_dir / name
        if path.is_file():
            candidates.append(("run", path))

    for scope, source in candidates:
        relative = source.relative_to(run_dir)
        safe_name = "__".join(relative.parts)
        destination = figures_dir / safe_name
        shutil.copy2(source, destination)
        copied[scope].append(f"figures/{destination.name}")
    return copied


def _manifest_items(manifest: Mapping[str, Any], aliases: Iterable[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for alias in aliases:
        if alias in manifest:
            selected[alias] = manifest[alias]
    return selected


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No recorded data._", ""]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend(
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows
    )
    lines.append("")
    return lines


def _metric_markdown(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for recording, values in metrics.items():
        scalars = _flatten_scalars(values)
        if not scalars:
            rows.append([recording, "_No metrics recorded_", ""])
            continue
        for key, value in list(sorted(scalars.items()))[:16]:
            rows.append([recording, key, _format_value(value)])
    return _markdown_table(["Recording", "Metric", "Measured value"], rows)


def _csv_markdown(inventory: list[dict[str, Any]]) -> list[str]:
    rows = [[str(item["path"]), str(item["rows"]), str(item["columns"])] for item in inventory]
    return _markdown_table(["CSV artifact", "Data rows", "Columns"], rows)


def _figure_markdown(paths: Iterable[str]) -> list[str]:
    paths = list(paths)
    if not paths:
        return ["_No figures were generated for this section._", ""]
    return [f"- [{Path(path).name}]({path})" for path in paths] + [""]


def _scorecard_rows(scorecard: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for stage in STAGES:
        entry = scorecard.get(stage, {})
        if not isinstance(entry, Mapping):
            entry = {}
        rows.append(
            [
                stage,
                _format_value(entry.get("score", 0)),
                _format_value(entry.get("raw_score", 0)),
                str(len(entry.get("passed", []))) if isinstance(entry.get("passed"), list) else "0",
                str(len(entry.get("failed", []))) if isinstance(entry.get("failed"), list) else "0",
                ", ".join(str(item) for item in entry.get("cap_reasons", []))
                if isinstance(entry.get("cap_reasons"), list)
                else "",
            ]
        )
    return rows


def _warning_failure_items(manifest: Mapping[str, Any], scorecard: Mapping[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("warnings", "failures"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            items.extend(f"{key[:-1].title()}: {_format_value(item)}" for item in value)
    for stage in STAGES:
        entry = scorecard.get(stage)
        if isinstance(entry, Mapping):
            failed = entry.get("failed", [])
            if isinstance(failed, list) and failed:
                items.append(f"{stage} failed checks: {', '.join(str(item) for item in failed)}")
    return items


def _limitation_items(manifest: Mapping[str, Any], scorecard: Mapping[str, Any]) -> list[str]:
    limitations = manifest.get("limitations", [])
    items = [str(item) for item in limitations] if isinstance(limitations, list) else []
    for stage in STAGES:
        entry = scorecard.get(stage)
        if not isinstance(entry, Mapping):
            continue
        reasons = entry.get("cap_reasons", [])
        if isinstance(reasons, list):
            items.extend(f"{stage}: {reason}" for reason in reasons)
    return list(dict.fromkeys(items))


def _build_markdown(
    manifest: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    csv_inventory: Mapping[str, list[dict[str, Any]]],
    figures: Mapping[str, list[str]],
) -> str:
    lines = ["# PG-AMCD Stage 1–4 Pipeline Report", ""]

    lines.extend(["## 1. Run overview", ""])
    overview = _manifest_items(
        manifest,
        (
            "run_id",
            "git_commit",
            "git_dirty",
            "git_worktree_sha256",
            "status",
            "start_timestamp",
            "start_iso",
            "end_timestamp",
            "end_iso",
            "pipeline_version",
            "feature_schema_version",
            "total_runtime",
            "success_count",
            "failure_count",
        ),
    )
    lines.extend(
        _markdown_table(
            ["Manifest field", "Recorded value"],
            [[key, _format_value(value)] for key, value in overview.items()],
        )
    )

    lines.extend(["## 2. Input validation", ""])
    validation = manifest.get("input_validation", manifest.get("validation", {}))
    validation_rows = [
        [key, _format_value(value)] for key, value in _flatten_scalars(validation).items()
    ]
    lines.extend(_markdown_table(["Validation metric", "Recorded value"], validation_rows))

    lines.extend(["## 3. Preprocessing summary", ""])
    lines.extend(_metric_markdown(metrics.get("Stage_1", {})))
    lines.extend(_csv_markdown(csv_inventory.get("Stage_1", [])))
    lines.extend(_figure_markdown(figures.get("Stage_1", [])))

    stage_sections = (
        ("Stage_1", "## 4. Stage 1 decomposition"),
        ("Stage_2", "## 5. Stage 2 IMF gating"),
        ("Stage_3", "## 6. Stage 3 wavelet denoising"),
        ("Stage_4", "## 7. Stage 4 feature extraction"),
    )
    for stage, heading in stage_sections:
        lines.extend([heading, ""])
        lines.extend(_metric_markdown(metrics.get(stage, {})))
        lines.extend(_csv_markdown(csv_inventory.get(stage, [])))
        lines.extend(_figure_markdown(figures.get(stage, [])))

    lines.extend(["## 8. Stage scorecard", ""])
    lines.extend(
        _markdown_table(
            ["Stage", "Score", "Raw score", "Passed", "Failed", "Cap reasons"],
            _scorecard_rows(scorecard),
        )
    )
    lines.extend(_figure_markdown(figures.get("run", [])))

    lines.extend(["## 9. Warnings and failures", ""])
    warning_items = _warning_failure_items(manifest, scorecard)
    lines.extend([f"- {item}" for item in warning_items] if warning_items else ["_None recorded._"])
    lines.append("")

    lines.extend(["## 10. Limitations", ""])
    limitation_items = _limitation_items(manifest, scorecard)
    lines.extend(
        [f"- {item}" for item in limitation_items] if limitation_items else ["_None recorded._"]
    )
    lines.append("")
    return "\n".join(lines)


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p><em>No recorded data.</em></p>"
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _metric_html(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    rows: list[list[str]] = []
    for recording, values in metrics.items():
        for key, value in list(sorted(_flatten_scalars(values).items()))[:16]:
            rows.append([recording, key, _format_value(value)])
    return _html_table(["Recording", "Metric", "Measured value"], rows)


def _figure_html(paths: Iterable[str]) -> str:
    paths = list(paths)
    if not paths:
        return "<p><em>No figures were generated for this section.</em></p>"
    return (
        "<div class='gallery'>"
        + "".join(
            f"<figure><a href='{html.escape(path)}'><img loading='lazy' src='{html.escape(path)}' "
            f"alt='{html.escape(Path(path).stem)}'></a><figcaption>{html.escape(Path(path).name)}</figcaption></figure>"
            for path in paths
        )
        + "</div>"
    )


def _build_html(
    manifest: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    csv_inventory: Mapping[str, list[dict[str, Any]]],
    figures: Mapping[str, list[str]],
) -> str:
    overview = _manifest_items(
        manifest,
        (
            "run_id",
            "git_commit",
            "git_dirty",
            "git_worktree_sha256",
            "status",
            "start_timestamp",
            "start_iso",
            "end_timestamp",
            "end_iso",
            "pipeline_version",
            "feature_schema_version",
            "total_runtime",
            "success_count",
            "failure_count",
        ),
    )
    validation = manifest.get("input_validation", manifest.get("validation", {}))
    warning_items = _warning_failure_items(manifest, scorecard)
    limitation_items = _limitation_items(manifest, scorecard)

    section_bodies: list[str] = []
    section_bodies.append(
        _html_table(
            ["Manifest field", "Recorded value"],
            [[key, _format_value(value)] for key, value in overview.items()],
        )
    )
    section_bodies.append(
        _html_table(
            ["Validation metric", "Recorded value"],
            [[key, _format_value(value)] for key, value in _flatten_scalars(validation).items()],
        )
    )
    section_bodies.append(
        _metric_html(metrics.get("Stage_1", {})) + _figure_html(figures.get("Stage_1", []))
    )
    for stage in STAGES:
        csv_rows = [
            [str(item["path"]), str(item["rows"]), str(item["columns"])]
            for item in csv_inventory.get(stage, [])
        ]
        section_bodies.append(
            _metric_html(metrics.get(stage, {}))
            + _html_table(["CSV artifact", "Data rows", "Columns"], csv_rows)
            + _figure_html(figures.get(stage, []))
        )
    section_bodies.append(
        _html_table(
            ["Stage", "Score", "Raw score", "Passed", "Failed", "Cap reasons"],
            _scorecard_rows(scorecard),
        )
        + _figure_html(figures.get("run", []))
    )
    section_bodies.append(
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warning_items) + "</ul>"
        if warning_items
        else "<p><em>None recorded.</em></p>"
    )
    section_bodies.append(
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in limitation_items) + "</ul>"
        if limitation_items
        else "<p><em>None recorded.</em></p>"
    )

    navigation = "".join(
        f"<a href='#section-{index}'>{index}. {html.escape(title)}</a>"
        for index, title in enumerate(SECTION_TITLES, start=1)
    )
    sections = "".join(
        f"<section id='section-{index}'><details open><summary><h2>{index}. {html.escape(title)}</h2></summary>"
        f"{body}</details></section>"
        for index, (title, body) in enumerate(zip(SECTION_TITLES, section_bodies), start=1)
    )
    generated = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PG-AMCD Stage 1–4 Pipeline Report</title>
<style>
:root{{--ink:#17212b;--blue:#1565c0;--paper:#f5f7fa;--card:#fff;--line:#d7dee8}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
header,main,footer{{max-width:1440px;margin:auto;padding:1.25rem 2rem}} header{{background:#0d2945;color:white;max-width:none}}
nav{{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem}} nav a{{color:white;background:#ffffff1f;padding:.35rem .6rem;border-radius:.35rem;text-decoration:none}}
section{{background:var(--card);margin:1rem 0;padding:.7rem 1.2rem;border:1px solid var(--line);border-radius:.6rem}}
summary{{cursor:pointer}} summary h2{{display:inline;color:#174f7a}} table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid var(--line);padding:.45rem;text-align:left;vertical-align:top}} th{{background:#eaf2fb}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}} figure{{margin:0}} img{{width:100%;height:auto;border:1px solid var(--line)}}
figcaption{{font-size:.8rem;overflow-wrap:anywhere}} footer{{color:#52606d}}
</style></head><body>
<header><h1>PG-AMCD Stage 1–4 Pipeline Report</h1><p>Run {html.escape(str(manifest.get("run_id", "unknown")))}</p><nav>{navigation}</nav></header>
<main>{sections}</main><footer>Generated from run artifacts at {html.escape(generated)}</footer>
</body></html>"""


def generate_pipeline_report(
    run_dir: str | Path,
    *,
    refresh_scorecard: bool = True,
) -> dict[str, str]:
    """Generate the required Markdown/HTML report from a run directory.

    The only input is the run directory. Measured values are read from its
    manifest, JSON and CSV artifacts; callers cannot inject report metrics.
    """

    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required run manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not manifest:
        raise ValueError(f"Run manifest is not a JSON object: {manifest_path}")

    score_paths: dict[str, str] = {}
    if refresh_scorecard or not (root / "stage_scorecard.json").is_file():
        score_paths = generate_stage_scorecard(root)
    scorecard = _read_json(root / "stage_scorecard.json")

    report_dir = root / "report"
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = _copy_report_figures(root, figures_dir)
    metrics = {stage: _stage_metrics(root, stage) for stage in STAGES}
    inventories = {stage: _csv_inventory(root, stage) for stage in STAGES}

    markdown_path = report_dir / "pipeline_report.md"
    html_path = report_dir / "pipeline_report.html"
    markdown_path.write_text(
        _build_markdown(manifest, scorecard, metrics, inventories, figures),
        encoding="utf-8",
    )
    html_path.write_text(
        _build_html(manifest, scorecard, metrics, inventories, figures),
        encoding="utf-8",
    )
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "figures_dir": str(figures_dir),
        **{f"scorecard_{key}": value for key, value in score_paths.items()},
    }


__all__ = ["SECTION_TITLES", "generate_pipeline_report"]
