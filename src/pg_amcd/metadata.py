"""Machining-metadata loading and recording-safe matching.

The historical workbook identifies machining *conditions* rather than file
paths.  This module supports both an explicit modern schema and that legacy
condition table without inventing missing values such as tool tooth count.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MachiningMetadata:
    """Validated metadata associated with one input recording."""

    recording_id: str
    relative_path: str
    rpm: float | None = None
    tooth_count: int | None = None
    stickout: float | None = None
    depth_of_cut: float | None = None
    feed_rate: float | None = None
    tool_identifier: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _lookup(row: dict[str, Any], *names: str) -> Any:
    normalised_row = {_normalise_key(key): value for key, value in row.items()}
    for name in names:
        key = _normalise_key(name)
        if key in normalised_row:
            return normalised_row[key]
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def load_metadata_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV or Excel metadata while preserving undefined cells as ``None``."""

    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(metadata_path)
        clean = frame.astype(object).where(pd.notna(frame), None)
        return [dict(row) for row in clean.to_dict(orient="records")]
    raise ValueError(f"Unsupported metadata format '{suffix}'; use CSV or XLSX")


def _legacy_condition(path: Path) -> tuple[float | None, float | None, float | None]:
    folder_match = re.search(r"([0-9]+(?:p[0-9]+)?)inch_stickout", path.parent.name.lower())
    stickout = float(folder_match.group(1).replace("p", ".")) if folder_match else None
    parts = path.stem.split("_")
    rpm = _optional_float(parts[1]) if len(parts) > 1 else None
    doc_match = re.match(r"([0-9]+)", parts[2]) if len(parts) > 2 else None
    doc = float(doc_match.group(1)) / 1000.0 if doc_match else None
    return stickout, rpm, doc


def _row_metadata(
    row: dict[str, Any], relative_path: str, *, legacy: bool = False
) -> MachiningMetadata:
    stickout = _optional_float(_lookup(row, "stickout", "l/d", "ld"))
    depth = _optional_float(_lookup(row, "depth_of_cut", "depth of cut", "doc (in)", "doc"))
    feed = _optional_float(_lookup(row, "feed_rate", "feed", "feed (in/rev)"))
    label_value = _lookup(row, "label", "state")
    label = str(label_value).strip() if label_value is not None else None
    tool_value = _lookup(row, "tool_identifier", "tool_id", "tool")
    recording_value = _lookup(row, "recording_id")
    recording_id = (
        str(recording_value).strip()
        if recording_value is not None and str(recording_value).strip()
        else Path(relative_path).with_suffix("").as_posix().replace("/", "__")
    )
    # The legacy workbook's DOC is explicitly inches. No unit conversion is
    # performed; provenance and schema retain the unit as supplied.
    return MachiningMetadata(
        recording_id=recording_id,
        relative_path=relative_path,
        rpm=_optional_float(_lookup(row, "rpm")),
        tooth_count=_optional_int(_lookup(row, "tooth_count", "tooth count", "teeth")),
        stickout=stickout,
        depth_of_cut=depth,
        feed_rate=feed,
        tool_identifier=str(tool_value).strip() if tool_value is not None else None,
        label=label,
    )


def build_metadata_index(
    input_dir: str | Path,
    mat_files: list[Path],
    metadata_path: str | Path,
) -> tuple[dict[str, MachiningMetadata], dict[str, Any]]:
    """Match metadata to inputs using relative paths or legacy conditions.

    The returned mapping is keyed by relative POSIX path, avoiding collisions
    when different stickout folders contain the same basename.
    """

    root = Path(input_dir).resolve()
    rows = load_metadata_rows(metadata_path)
    relative_files = {path.resolve().relative_to(root).as_posix(): path for path in mat_files}
    basename_counts: dict[str, int] = {}
    for relative in relative_files:
        basename_counts[Path(relative).name] = basename_counts.get(Path(relative).name, 0) + 1

    matched: dict[str, MachiningMetadata] = {}
    ambiguous_rows = 0
    unmappable_rows = 0
    condition_rows = 0

    for row in rows:
        explicit = _lookup(row, "file_path", "relative_path", "recording_id")
        if explicit is not None and str(explicit).strip():
            raw_key = str(explicit).strip().replace("\\", "/")
            candidates = [raw_key]
            if not raw_key.lower().endswith(".mat"):
                candidates.append(f"{raw_key}.mat")
            matched_relative = next((key for key in candidates if key in relative_files), None)
            if matched_relative is None:
                basename = Path(raw_key).name
                matches = [key for key in relative_files if Path(key).name == basename]
                if len(matches) == 1:
                    matched_relative = matches[0]
                elif len(matches) > 1:
                    ambiguous_rows += 1
                    continue
            if matched_relative is None:
                unmappable_rows += 1
                continue
            if matched_relative in matched:
                ambiguous_rows += 1
                continue
            matched[matched_relative] = _row_metadata(row, matched_relative)
            continue

        stickout = _optional_float(_lookup(row, "l/d", "stickout", "ld"))
        rpm = _optional_float(_lookup(row, "rpm"))
        doc = _optional_float(_lookup(row, "doc (in)", "depth_of_cut", "doc"))
        if stickout is None or rpm is None or doc is None:
            unmappable_rows += 1
            continue
        condition_rows += 1
        condition_matches = []
        for relative, path in relative_files.items():
            file_stickout, file_rpm, file_doc = _legacy_condition(path)
            if (
                file_stickout is not None
                and file_rpm is not None
                and file_doc is not None
                and math.isclose(file_stickout, stickout, abs_tol=1e-9)
                and math.isclose(file_rpm, rpm, abs_tol=1e-9)
                and math.isclose(file_doc, doc, abs_tol=5e-7)
            ):
                condition_matches.append(relative)
        if not condition_matches:
            unmappable_rows += 1
        for relative in condition_matches:
            if relative in matched:
                ambiguous_rows += 1
                continue
            matched[relative] = _row_metadata(row, relative, legacy=True)

    recording_id_counts: dict[str, int] = {}
    for metadata in matched.values():
        recording_id_counts[metadata.recording_id] = (
            recording_id_counts.get(metadata.recording_id, 0) + 1
        )

    diagnostics = {
        "metadata_rows": len(rows),
        "matched_recordings": len(matched),
        "missing_recordings": sorted(set(relative_files) - set(matched)),
        "ambiguous_rows": ambiguous_rows,
        "unmappable_rows": unmappable_rows,
        "legacy_condition_rows": condition_rows,
        "duplicate_basenames": sorted(
            name for name, count in basename_counts.items() if count > 1
        ),
        "duplicate_recording_ids": sorted(
            recording_id for recording_id, count in recording_id_counts.items() if count > 1
        ),
        "missing_tooth_count": sorted(
            relative for relative, metadata in matched.items() if metadata.tooth_count is None
        ),
    }
    return matched, diagnostics
