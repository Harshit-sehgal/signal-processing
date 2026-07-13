"""Provenance and reproducibility helpers (Segment 4 / Goals 4.3-4.4)."""

import hashlib
import os


def compute_file_sha256(file_path: str) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def is_output_stale(input_path: str, outputs) -> bool:
    """Return True if any output is missing or older than the input file.

    Implements stale-output detection so re-runs only regenerate artifacts
    when the source signal has changed.
    """
    try:
        in_mtime = os.path.getmtime(input_path)
    except OSError:
        return True
    for op in outputs:
        if not os.path.exists(op):
            return True
        try:
            if os.path.getmtime(op) < in_mtime:
                return True
        except OSError:
            return True
    return False


def compute_run_id(
    config_sha256: str,
    git_commit: str,
    input_checksums: list,
    metadata_checksum: str = "",
) -> str:
    """Deterministic run identifier from config, git state and input checksums.

    Results are stored under ``outputs/<run_id>/`` and never reused unless the
    run ID matches (Goal 4.4).
    """
    h = hashlib.sha256()
    h.update(config_sha256.encode("utf-8"))
    h.update(git_commit.encode("utf-8"))
    for checksum in sorted(input_checksums):
        h.update(checksum.encode("utf-8"))
    h.update(metadata_checksum.encode("utf-8"))
    return h.hexdigest()
