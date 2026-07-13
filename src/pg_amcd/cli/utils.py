"""Shared helpers for the PG-AMCD CLI."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import scipy
import pywt


def get_git_commit_sha() -> str:
    """Retrieves the current git commit SHA of the repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "Unknown"


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def _sha256_of_config(config: dict, config_path: str) -> str:
    """SHA-256 of the resolved configuration (Goal 4.3)."""
    from pg_amcd.provenance import compute_file_sha256

    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    if config_path and os.path.exists(config_path):
        h.update(compute_file_sha256(config_path).encode("utf-8"))
    return h.hexdigest()


def get_environment_info() -> dict:
    """Collects python, packages, and OS information."""
    import platform

    import numpy as np

    info = {
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywavelets": pywt.__version__,
        },
    }
    try:
        import PyEMD

        info["packages"]["PyEMD"] = PyEMD.__version__
    except Exception:
        pass
    return info


def _load_metadata_index(path: str) -> List[Dict[str, Any]]:
    """Load a dataset metadata spreadsheet (CSV or XLSX) into a list of row dicts."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading Excel metadata requires pandas+openpyxl; supply a CSV instead."
            ) from exc
        df = pd.read_excel(path)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    raise ValueError(f"Unsupported metadata format: {ext}")


def get_case_insensitive(
    d: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = None
):
    """Return the first value from ``d`` whose key matches one of ``keys`` case-insensitively."""
    if not d:
        return default
    for k in keys:
        for dk in d:
            if dk.lower() == k.lower():
                return d[dk]
    return default
