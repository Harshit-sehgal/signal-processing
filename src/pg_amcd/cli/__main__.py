"""Allow ``python -m pg_amcd.cli`` to execute the CLI."""

from pg_amcd.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
