#!/usr/bin/env python3
"""Generate a metadata CSV template from a directory of MAT recordings.

The output contains one row per ``.mat`` file with the ``relative_path`` and
``recording_id`` columns populated and the physics columns left blank for the
user to fill in.  Tooth counts must be measured or confirmed from experiment
records; the template intentionally does not guess them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


METADATA_COLUMNS = [
    "relative_path",
    "recording_id",
    "rpm",
    "tooth_count",
    "stickout",
    "depth_of_cut",
    "feed_rate",
    "tool_identifier",
    "label",
    "repetition_id",
]


def generate_metadata_template(input_dir: str, output_path: str) -> None:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        raise ValueError(f"No MAT files found under {root}")

    rows: list[dict[str, str]] = []
    for path in mat_files:
        relative = path.relative_to(root).as_posix()
        recording_id = path.with_suffix("").name
        rows.append(
            {
                "relative_path": relative,
                "recording_id": recording_id,
                "rpm": "",
                "tooth_count": "",
                "stickout": "",
                "depth_of_cut": "",
                "feed_rate": "",
                "tool_identifier": "",
                "label": "",
                "repetition_id": "",
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote metadata template with {len(rows)} row(s) to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a metadata CSV template")
    parser.add_argument("input_dir", help="Directory containing MAT recordings")
    parser.add_argument("-o", "--output", default="metadata.csv", help="Output CSV path")
    args = parser.parse_args(argv)
    try:
        generate_metadata_template(args.input_dir, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
