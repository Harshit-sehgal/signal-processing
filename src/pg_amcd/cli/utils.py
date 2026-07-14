"""Shared CLI helpers."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pg_amcd.metadata import load_metadata_rows
from pg_amcd.provenance import canonical_json_sha256


def get_git_commit_sha() -> str:
    """Return the checked-out Git commit or ``Unknown`` outside a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "Unknown"
    return result.stdout.strip()


def _git_is_dirty() -> bool:
    """Return whether tracked or untracked files differ from the commit."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.stdout.strip())


def get_git_worktree_sha256() -> str:
    """Hash tracked changes plus untracked, non-ignored file contents.

    The commit alone cannot identify code executed from a dirty checkout. The
    digest follows Git's ignore rules, so ignored datasets, environments, and
    generated outputs do not pollute the scientific run identity.
    """

    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
        )
        root = Path(root_result.stdout.decode("utf-8", errors="surrogateescape").strip())
        diff_result = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=root,
            capture_output=True,
            check=True,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff_result.stdout)
    untracked_paths = sorted(path for path in untracked_result.stdout.split(b"\0") if path)
    for encoded_path in untracked_paths:
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        digest.update(b"untracked-path\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        try:
            if candidate.is_symlink():
                digest.update(b"symlink\0")
                digest.update(candidate.readlink().as_posix().encode("utf-8"))
            elif candidate.is_file():
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                digest.update(b"missing-or-non-file")
        except OSError:
            return ""
    return digest.hexdigest()


def _sha256_of_config(config: Dict[str, Any], config_path: Optional[str] = None) -> str:
    """Return the canonical resolved-configuration digest.

    The optional path is accepted for compatibility; run identity is based on
    resolved values rather than formatting of the source JSON file.
    """

    del config_path
    return canonical_json_sha256(config)


def get_environment_info() -> Dict[str, Any]:
    """Collect versioned runtime dependencies used by the scientific workflow."""

    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "EMD-signal": "EMD-signal",
        "PyWavelets": "PyWavelets",
        "pandas": "pandas",
        "openpyxl": "openpyxl",
        "scikit-learn": "scikit-learn",
    }
    packages: Dict[str, str] = {}
    for label, distribution in distributions.items():
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[label] = "not-installed"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "packages": packages,
    }


def _load_metadata_index(path: str) -> List[Dict[str, Any]]:
    """Compatibility wrapper for historical callers."""

    return load_metadata_rows(Path(path))


def get_case_insensitive(
    mapping: Optional[Dict[str, Any]],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first case-insensitive matching mapping value."""

    if not mapping:
        return default
    wanted = {key.casefold() for key in keys}
    for key, value in mapping.items():
        if key.casefold() in wanted:
            return value
    return default
