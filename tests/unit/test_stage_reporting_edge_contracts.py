"""Fallback and malformed-input contracts for the combined Stage 1--4 report."""

from __future__ import annotations

import json
from pathlib import Path

from pg_amcd import stage_reporting


def _manifest(run_id: str = "report-edge") -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "completed",
        "input_validation": {"n_files": 1, "n_valid": 1, "n_invalid": 0},
        "warnings": [],
        "failures": [],
        "limitations": [],
    }


def _scorecard() -> dict[str, object]:
    return {
        stage: {
            "score": 80.0,
            "raw_score": 85.0,
            "passed": ["one"],
            "failed": ["two"],
            "cap_reasons": ["fixture cap"],
        }
        for stage in stage_reporting.STAGES
    }


def test_json_recording_scalar_and_format_fallbacks(tmp_path: Path) -> None:
    assert stage_reporting._read_json(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert stage_reporting._read_json(malformed) == {}
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert stage_reporting._read_json(sequence) == {}
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"value": 1}', encoding="utf-8")
    assert stage_reporting._read_json(mapping) == {"value": 1}
    assert stage_reporting._recording_dirs(tmp_path / "missing-stage") == []

    flattened = stage_reporting._flatten_scalars(
        {
            "short": [1, 2],
            "long": list(range(9)),
            "nested": {"value": 3},
            "undefined": None,
        }
    )
    assert flattened["short"] == [1, 2]
    assert flattened["long.count"] == 9
    assert flattened["nested.value"] == 3
    assert "undefined" not in flattened
    assert stage_reporting._format_value(1.23456789) == "1.23457"
    assert stage_reporting._format_value([True, 2.0]) == "yes, 2"
    assert stage_reporting._format_value(False) == "no"
    assert stage_reporting._format_value("value") == "value"


def test_csv_inventory_and_figure_copy_handle_absence_and_replace_stale_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert stage_reporting._csv_inventory(run_dir, "Stage_1") == []

    stage = run_dir / "Stage_1" / "sample"
    stage.mkdir(parents=True)
    (stage / "metrics.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    broken = stage / "broken.csv"
    broken.symlink_to(stage / "missing.csv")
    inventory = stage_reporting._csv_inventory(run_dir, "Stage_1")
    by_name = {Path(item["path"]).name: item for item in inventory}
    assert by_name["metrics.csv"]["rows"] == 2
    assert by_name["metrics.csv"]["columns"] == 2
    assert by_name["broken.csv"]["rows"] == 0

    (stage / "waveform.png").write_bytes(b"p" * 128)
    (stage / "vector.svg").write_text("<svg></svg>" * 20, encoding="utf-8")
    (run_dir / "stage_scorecard.png").write_bytes(b"s" * 128)
    figures_dir = run_dir / "report" / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "stale.png").write_bytes(b"old")

    copied = stage_reporting._copy_report_figures(run_dir, figures_dir)
    assert not (figures_dir / "stale.png").exists()
    assert "figures/Stage_1__sample__waveform.png" in copied["Stage_1"]
    assert "figures/Stage_1__sample__vector.svg" in copied["Stage_1"]
    assert "figures/stage_scorecard.png" in copied["run"]


def test_markdown_html_and_warning_helpers_cover_empty_and_malformed_entries() -> None:
    assert stage_reporting._markdown_table(["A"], []) == ["_No recorded data._", ""]
    table = stage_reporting._markdown_table(["A"], [["x|y"]])
    assert "x\\|y" in table[2]
    assert "No recorded data" in stage_reporting._html_table(["A"], [])
    assert "<td>x</td>" in stage_reporting._html_table(["A"], [["x"]])
    assert "No figures" in stage_reporting._figure_html([])
    assert "fig.png" in stage_reporting._figure_html(["figures/fig.png"])
    assert stage_reporting._figure_markdown([])[0].startswith("_No figures")

    metrics = {"sample": {}}
    assert "_No metrics recorded_" in "\n".join(stage_reporting._metric_markdown(metrics))
    scorecard = _scorecard()
    scorecard["Stage_2"] = "malformed"
    rows = stage_reporting._scorecard_rows(scorecard)
    assert rows[1][1] == "0"

    manifest = {
        "warnings": "not-a-list",
        "failures": [{"stage": "Stage_1", "error": "failed"}],
        "limitations": "not-a-list",
    }
    warnings = stage_reporting._warning_failure_items(manifest, scorecard)
    assert any("Failure:" in item for item in warnings)
    assert any("Stage_1 failed checks" in item for item in warnings)
    limitations = stage_reporting._limitation_items(manifest, scorecard)
    assert "Stage_1: fixture cap" in limitations
    assert all("Stage_2" not in item for item in limitations)


def test_generate_report_rejects_nonobject_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("[]", encoding="utf-8")

    try:
        stage_reporting.generate_pipeline_report(run_dir)
    except ValueError as exc:
        assert "not a JSON object" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("non-object manifest was accepted")


def test_generate_report_can_reuse_existing_scorecard_without_refresh(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    (run_dir / "stage_scorecard.json").write_text(json.dumps(_scorecard()), encoding="utf-8")
    for stage in stage_reporting.STAGES:
        (run_dir / stage).mkdir()

    paths = stage_reporting.generate_pipeline_report(run_dir, refresh_scorecard=False)

    assert set(paths) == {"markdown", "html", "figures_dir"}
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    report_html = Path(paths["html"]).read_text(encoding="utf-8")
    assert "fixture cap" in markdown
    assert "report-edge" in report_html
    assert "Stage_1 failed checks" in markdown
