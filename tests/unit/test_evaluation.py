"""Goal 6.1 canonical metadata schema enforcement (tested without real data)."""
import os
import tempfile

import openpyxl
import pytest

from pg_amcd.evaluation import (
    build_dataset_index,
    REQUIRED_METADATA_COLUMNS,
    _REQUIRED_ALIASES,
)


def _write_workbook(path, headers, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "combinations"
    ws.append(headers)
    for r in rows:
        ws.append(r)
    wb.save(path)


def _complete_headers():
    # All canonical columns present (with a couple of real aliases).
    return [
        "recording_id",
        "experiment_run_id",
        "stickout",
        "tooth_count",
        "tool_id",
        "sensor_id",
        "state",            # satisfies both 'state' and 'label'
        "chatter_onset_time",
        "spindle speed",    # alias for 'rpm'
        "doc (in)",         # alias for 'doc'
        "feed (in/rev)",
    ]


def test_complete_workbook_passes_schema():
    with tempfile.TemporaryDirectory() as d:
        wb_path = os.path.join(d, "meta.xlsx")
        _write_workbook(
            wb_path,
            _complete_headers(),
            [["r1", "e1", 2.0, 1, "t1", "s1", "chatter", 0.5, 570, 0.005, 0.002]],
        )
        # No MAT files -> empty index, but schema must pass (no raise).
        idx = build_dataset_index(d, wb_path)
        assert idx == []


def test_missing_required_column_raises():
    with tempfile.TemporaryDirectory() as d:
        wb_path = os.path.join(d, "meta.xlsx")
        headers = [h for h in _complete_headers() if h != "tool_id"]
        _write_workbook(
            wb_path,
            headers,
            [["r1", "e1", 2.0, 1, "t1", "s1", "chatter", 0.5, 570, 0.005, 0.002]],
        )
        with pytest.raises(ValueError) as exc:
            build_dataset_index(d, wb_path)
        assert "tool_id" in str(exc.value)
        assert "Goal 6.1" in str(exc.value)


def test_required_columns_constant_is_nonempty():
    assert "rpm" in REQUIRED_METADATA_COLUMNS
    assert "state" in REQUIRED_METADATA_COLUMNS
    assert "label" in REQUIRED_METADATA_COLUMNS
    # label is satisfiable via the state alias group.
    assert "label" in _REQUIRED_ALIASES
