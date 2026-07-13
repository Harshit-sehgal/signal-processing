"""Argument parsing and command routing for the PG-AMCD CLI."""

import argparse

from pg_amcd.cli.run import run_pipeline_on_dataset
from pg_amcd.cli.validate import run_validation_on_dataset


def main():
    """Entry point for the ``pg-amcd`` command-line interface."""
    parser = argparse.ArgumentParser(
        description="PG-AMCD Signal Processing CLI Command Line Interface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run signal processing pipeline on a dataset"
    )
    run_parser.add_argument(
        "--input-dir", required=True, help="Path to Vibration - ML raw data directory"
    )
    run_parser.add_argument("--metadata", required=False, help="Path to combination spreadsheet")
    run_parser.add_argument("--output-dir", required=True, help="Path to output processed results")
    run_parser.add_argument("--config", required=False, help="Path to config.json file")
    run_parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue processing on file failure"
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate raw signals against the input contract without processing",
    )
    validate_parser.add_argument(
        "--input-dir", required=True, help="Path to Vibration - ML raw data directory"
    )
    validate_parser.add_argument("--config", required=False, help="Path to config.json file")
    validate_parser.add_argument(
        "--output", required=False, help="Path to write the JSON validation report"
    )
    validate_parser.add_argument(
        "--metadata",
        required=False,
        help="Path to metadata CSV/XLSX for dataset validation reporting",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline_on_dataset(args)
    elif args.command == "validate":
        run_validation_on_dataset(args)


if __name__ == "__main__":
    main()
