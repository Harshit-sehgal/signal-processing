"""Argument parsing and routing for the Stage 1--4 PG-AMCD CLI."""

from __future__ import annotations

import argparse
import multiprocessing
from collections.abc import Sequence


def _ensure_multiprocessing_spawn() -> None:
    """Use ``spawn`` so PyEMD's Pool avoids the fork-in-thread DeprecationWarning."""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


def _stage_number(value: str) -> int:
    try:
        stage = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--through-stage must be an integer from 1 to 4") from exc
    if stage > 4:
        raise argparse.ArgumentTypeError(
            "Stages above 4 are outside the current project scope; the workflow ends at feature extraction"
        )
    if stage < 1:
        raise argparse.ArgumentTypeError("--through-stage must be between 1 and 4")
    return stage


def build_parser() -> argparse.ArgumentParser:
    """Build the public Stage 1--4 command-line interface."""

    parser = argparse.ArgumentParser(
        prog="pg-amcd",
        description="PG-AMCD machining-signal processing through Stage 4 feature extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the canonical signal pipeline")
    run_parser.add_argument("--input-dir", required=True, help="Directory containing raw MAT files")
    run_parser.add_argument("--metadata", help="CSV/XLSX machining metadata file")
    run_parser.add_argument("--output-dir", default="outputs", help="Parent output directory")
    run_parser.add_argument("--config", help="JSON configuration (packaged default when omitted)")
    run_parser.add_argument("--through-stage", type=_stage_number, default=4)
    run_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Process remaining recordings, but still return non-zero if any recording fails",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate raw inputs and metadata without processing"
    )
    validate_parser.add_argument("--input-dir", required=True)
    validate_parser.add_argument("--metadata")
    validate_parser.add_argument("--config")
    validate_parser.add_argument("--output", default="validation_report.json")

    report_parser = subparsers.add_parser(
        "report", help="Regenerate the Stage 1--4 report for an existing run"
    )
    report_parser.add_argument("--run-dir", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process exit status."""

    _ensure_multiprocessing_spawn()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from pg_amcd.cli.run import run_pipeline_on_dataset

        return int(run_pipeline_on_dataset(args))
    if args.command == "validate":
        from pg_amcd.cli.validate import run_validation_on_dataset

        return int(run_validation_on_dataset(args))

    from pg_amcd.stage_reporting import generate_pipeline_report

    generate_pipeline_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
