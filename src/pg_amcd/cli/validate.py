"""Implementation of ``pg-amcd validate``."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pg_amcd.config import load_pipeline_config
from pg_amcd.io import validate_and_load_signal
from pg_amcd.metadata import MachiningMetadata, build_metadata_index
from pg_amcd.provenance import canonical_json_sha256, compute_file_sha256


def _discover_mat_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.mat") if path.is_file())


def run_validation_on_dataset(args: argparse.Namespace) -> int:
    """Validate signals and metadata, write a machine-readable report, and return status."""

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "files": [],
        "warnings": [],
        "failures": [],
    }

    try:
        config = load_pipeline_config(args.config)
        report["resolved_config_sha256"] = canonical_json_sha256(config)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        mat_files = _discover_mat_files(input_dir)
        if not mat_files:
            raise ValueError(f"No MAT files found in: {input_dir}")

        use_physics = bool(config.get("use_physics_gating", True))
        metadata_index: dict[str, MachiningMetadata] = {}
        if args.metadata:
            metadata_index, metadata_diagnostics = build_metadata_index(
                input_dir, mat_files, args.metadata
            )
            report["metadata"] = metadata_diagnostics
            report["metadata"]["path"] = str(Path(args.metadata).resolve())
            report["metadata"]["sha256"] = compute_file_sha256(args.metadata)
            if int(metadata_diagnostics.get("ambiguous_rows", 0)):
                report["failures"].append(
                    "Metadata contains ambiguous or duplicate rows; each recording must map once"
                )
            duplicate_ids = metadata_diagnostics.get("duplicate_recording_ids", [])
            if duplicate_ids:
                report["failures"].append(
                    "Metadata recording_id values must be unique: "
                    + ", ".join(str(value) for value in duplicate_ids)
                )
        elif use_physics:
            raise ValueError(
                "Physics-guided gating requires --metadata with positive RPM and tooth_count"
            )
        else:
            report["warnings"].append(
                "No metadata supplied; metadata-dependent Stage 4 features will be undefined"
            )

        fs = float(config["sampling_rate"])
        validation_cfg = config.get("validation", {})
        tolerance = float(
            validation_cfg.get("sampling_rate_tolerance", validation_cfg.get("tolerance", 0.05))
        )
        minimum_duration = float(
            validation_cfg.get(
                "minimum_duration_seconds",
                validation_cfg.get("min_duration_seconds", 1.0),
            )
        )
        signal_column = int(validation_cfg.get("signal_column", 1))
        jitter = float(validation_cfg.get("timestamp_jitter_tolerance", 0.05))

        for mat_path in mat_files:
            relative = mat_path.relative_to(input_dir).as_posix()
            entry: dict[str, Any] = {
                "path": relative,
                "sha256": compute_file_sha256(mat_path),
                "signal_valid": False,
                "metadata_valid": not use_physics,
                "valid": False,
                "errors": [],
            }
            try:
                _, signal, estimated_fs = validate_and_load_signal(
                    str(mat_path),
                    configured_fs=fs,
                    tolerance=tolerance,
                    min_duration_seconds=minimum_duration,
                    signal_column=signal_column,
                    max_timestamp_jitter=jitter,
                )
                entry.update(
                    signal_valid=True,
                    estimated_sampling_rate_hz=float(estimated_fs),
                    sample_count=int(signal.size),
                    duration_seconds=float(signal.size / estimated_fs),
                )
            except (FileNotFoundError, ValueError) as exc:
                entry["errors"].append(str(exc))

            metadata = metadata_index.get(relative)
            if metadata is not None:
                entry["metadata"] = metadata.to_dict()
                if use_physics:
                    if metadata.rpm is None or metadata.rpm <= 0:
                        entry["errors"].append("Physics metadata is missing a positive RPM")
                    if metadata.tooth_count is None or metadata.tooth_count < 1:
                        entry["errors"].append("Physics metadata is missing a positive tooth_count")
                    entry["metadata_valid"] = not any(
                        message.startswith("Physics metadata") for message in entry["errors"]
                    )
                else:
                    entry["metadata_valid"] = True
            elif use_physics:
                entry["errors"].append("No metadata row maps to this relative input path")

            entry["valid"] = bool(entry["signal_valid"] and entry["metadata_valid"])
            report["files"].append(entry)

        report["n_files"] = len(report["files"])
        report["n_signal_valid"] = sum(bool(item["signal_valid"]) for item in report["files"])
        report["n_valid"] = sum(bool(item["valid"]) for item in report["files"])
        report["n_invalid"] = report["n_files"] - report["n_valid"]
        report["status"] = (
            "valid" if report["n_invalid"] == 0 and not report["failures"] else "invalid"
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        report["status"] = "invalid"
        report["failures"].append(str(exc))
        report.setdefault("n_files", 0)
        report.setdefault("n_signal_valid", 0)
        report.setdefault("n_valid", 0)
        report.setdefault("n_invalid", 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"Validation {report['status']}: {report.get('n_valid', 0)}/"
        f"{report.get('n_files', 0)} recordings valid; report={output_path}"
    )
    for failure in report["failures"]:
        print(f"Error: {failure}")
    return 0 if report["status"] == "valid" else 1
