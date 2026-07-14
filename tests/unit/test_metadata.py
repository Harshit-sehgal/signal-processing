"""Recording-safe machining metadata matching tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from pg_amcd.metadata import build_metadata_index, load_metadata_rows


def _recording(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MAT fixture")
    return path


def test_legacy_condition_workbook_maps_duplicate_basenames_without_collision(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    first = _recording(input_dir, "2p5inch_stickout/u_570_005.mat")
    second = _recording(input_dir, "3p5inch_stickout/u_570_005.mat")
    workbook = tmp_path / "conditions.xlsx"
    pd.DataFrame(
        [
            {"l/d": 2.5, "RPM": 570, "DOC (in)": 0.005, "State": "Chatter"},
            {"l/d": 3.5, "RPM": 570, "DOC (in)": 0.005, "State": "No Chatter"},
        ]
    ).to_excel(workbook, index=False)

    metadata, diagnostics = build_metadata_index(input_dir, [first, second], workbook)

    assert set(metadata) == {
        "2p5inch_stickout/u_570_005.mat",
        "3p5inch_stickout/u_570_005.mat",
    }
    assert metadata["2p5inch_stickout/u_570_005.mat"].label == "Chatter"
    assert metadata["3p5inch_stickout/u_570_005.mat"].label == "No Chatter"
    assert len({item.recording_id for item in metadata.values()}) == 2
    assert all(item.tooth_count is None for item in metadata.values())
    assert diagnostics["duplicate_basenames"] == ["u_570_005.mat"]
    assert diagnostics["missing_tooth_count"] == sorted(metadata)
    assert diagnostics["matched_recordings"] == 2


def test_explicit_relative_path_wins_over_custom_recording_id(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    first = _recording(input_dir, "2p5inch_stickout/u_570_005.mat")
    second = _recording(input_dir, "3p5inch_stickout/u_570_005.mat")
    metadata_csv = tmp_path / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording_id", "relative_path", "rpm", "tooth_count", "label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "recording_id": "custom-recording-7",
                "relative_path": r"3p5inch_stickout\u_570_005.mat",
                "rpm": "570",
                "tooth_count": "",
                "label": "stable",
            }
        )

    metadata, diagnostics = build_metadata_index(input_dir, [first, second], metadata_csv)

    assert list(metadata) == ["3p5inch_stickout/u_570_005.mat"]
    item = metadata["3p5inch_stickout/u_570_005.mat"]
    assert item.recording_id == "custom-recording-7"
    assert item.relative_path == "3p5inch_stickout/u_570_005.mat"
    assert item.rpm == 570.0
    assert item.tooth_count is None
    assert diagnostics["ambiguous_rows"] == 0
    assert diagnostics["missing_tooth_count"] == ["3p5inch_stickout/u_570_005.mat"]


def test_duplicate_rows_and_recording_ids_are_reported_without_overwrite(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    first = _recording(input_dir, "first.mat")
    second = _recording(input_dir, "second.mat")
    metadata_csv = tmp_path / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "recording_id", "rpm", "tooth_count"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": "first.mat",
                "recording_id": "duplicate-id",
                "rpm": 600,
                "tooth_count": 2,
            }
        )
        writer.writerow(
            {
                "relative_path": "first.mat",
                "recording_id": "should-not-overwrite",
                "rpm": 900,
                "tooth_count": 4,
            }
        )
        writer.writerow(
            {
                "relative_path": "second.mat",
                "recording_id": "duplicate-id",
                "rpm": 600,
                "tooth_count": 2,
            }
        )

    metadata, diagnostics = build_metadata_index(input_dir, [first, second], metadata_csv)

    assert metadata["first.mat"].recording_id == "duplicate-id"
    assert metadata["first.mat"].rpm == 600.0
    assert metadata["second.mat"].recording_id == "duplicate-id"
    assert diagnostics["ambiguous_rows"] == 1
    assert diagnostics["duplicate_recording_ids"] == ["duplicate-id"]


def test_metadata_loader_rejects_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metadata_rows(tmp_path / "missing.csv")

    unsupported = tmp_path / "metadata.json"
    unsupported.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported metadata format"):
        load_metadata_rows(unsupported)


def test_explicit_matching_handles_extension_basename_and_unmappable_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    first = _recording(input_dir, "nested/first.mat")
    second = _recording(input_dir, "other/second.mat")
    metadata_csv = tmp_path / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "RPM",
                "tooth count",
                "stickout",
                "depth of cut",
                "feed (in/rev)",
                "tool_id",
                "state",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": "nested/first",
                "RPM": "not-a-number",
                "tooth count": "2.5",
                "stickout": 2.5,
                "depth of cut": 0.005,
                "feed (in/rev)": 0.002,
                "tool_id": "T7",
                "state": "stable",
            }
        )
        writer.writerow({"relative_path": "second.mat", "RPM": 600, "tooth count": 2})
        writer.writerow({"relative_path": "absent.mat", "RPM": 600, "tooth count": 2})

    metadata, diagnostics = build_metadata_index(input_dir, [first, second], metadata_csv)

    first_metadata = metadata["nested/first.mat"]
    assert first_metadata.recording_id == "nested__first"
    assert first_metadata.rpm is None
    assert first_metadata.tooth_count is None
    assert first_metadata.feed_rate == 0.002
    assert first_metadata.tool_identifier == "T7"
    assert first_metadata.label == "stable"
    assert metadata["other/second.mat"].rpm == 600.0
    assert first_metadata.to_dict()["relative_path"] == "nested/first.mat"
    assert diagnostics["unmappable_rows"] == 1


def test_ambiguous_basename_and_incomplete_legacy_rows_are_diagnosed(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    first = _recording(input_dir, "a/same.mat")
    second = _recording(input_dir, "b/same.mat")
    metadata_csv = tmp_path / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "l/d", "RPM", "DOC (in)"],
        )
        writer.writeheader()
        writer.writerow({"relative_path": "same.mat"})
        writer.writerow({"l/d": "", "RPM": 600, "DOC (in)": 0.005})
        writer.writerow({"l/d": 9.0, "RPM": 600, "DOC (in)": 0.005})

    metadata, diagnostics = build_metadata_index(input_dir, [first, second], metadata_csv)

    assert metadata == {}
    assert diagnostics["ambiguous_rows"] == 1
    assert diagnostics["unmappable_rows"] == 2


def test_repeated_legacy_condition_rows_do_not_overwrite_first_mapping(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    recording = _recording(input_dir, "2p5inch_stickout/u_570_005.mat")
    workbook = tmp_path / "conditions.xlsx"
    pd.DataFrame(
        [
            {"l/d": 2.5, "RPM": 570, "DOC (in)": 0.005, "State": "first"},
            {"l/d": 2.5, "RPM": 570, "DOC (in)": 0.005, "State": "second"},
        ]
    ).to_excel(workbook, index=False)

    metadata, diagnostics = build_metadata_index(input_dir, [recording], workbook)

    assert metadata["2p5inch_stickout/u_570_005.mat"].label == "first"
    assert diagnostics["legacy_condition_rows"] == 2
    assert diagnostics["ambiguous_rows"] == 1
