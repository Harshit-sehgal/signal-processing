"""Deterministic run identity and provenance helpers.

The Stage 1--4 workflow never decides whether an output is reusable from file
modification times.  A run is identified by the complete scientific identity:
source revision (including dirty-worktree content), resolved configuration,
input and metadata contents, runtime dependencies, and the
pipeline/feature-schema versions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PIPELINE_VERSION = "4.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"


def compute_file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON-compatible data using a stable, whitespace-free encoding."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_run_id(
    config_sha256: str | Mapping[str, Any],
    git_commit: str,
    input_checksums: Sequence[str] | Mapping[str, str],
    metadata_checksum: str = "",
    *,
    dependency_versions: Mapping[str, str] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    git_worktree_sha256: str = "",
) -> str:
    """Return the full SHA-256 run identity.

    ``config_sha256`` still accepts the historical pre-computed digest so
    callers remain compatible, but passing the resolved configuration mapping
    is preferred. Input mappings retain paths in the identity; sequences are
    sorted for backwards compatibility with the original helper.
    """

    if isinstance(config_sha256, Mapping):
        config_identity = canonical_json_sha256(config_sha256)
    else:
        config_identity = str(config_sha256)

    if isinstance(input_checksums, Mapping):
        inputs: Any = dict(sorted((str(k), str(v)) for k, v in input_checksums.items()))
    else:
        inputs = sorted(str(value) for value in input_checksums)

    identity = {
        "feature_schema_version": feature_schema_version,
        "git_commit": git_commit,
        "git_worktree_sha256": git_worktree_sha256,
        "input_checksums": inputs,
        "metadata_checksum": metadata_checksum,
        "pipeline_version": pipeline_version,
        "resolved_config_sha256": config_identity,
        "dependency_versions": dict(sorted((dependency_versions or {}).items())),
    }
    return canonical_json_sha256(identity)


def manifest_matches_run(run_dir: str | Path, run_id: str) -> bool:
    """Return whether a completed run identity and every stored output match.

    A matching run ID is necessary but insufficient for reuse: all output
    files recorded in the completed manifest must still exist below the run
    directory and retain their recorded SHA-256 digest.
    """

    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    identity_matches = bool(
        manifest.get("run_id") == run_id
        and manifest.get("status") == "completed"
        and manifest.get("end_timestamp")
    )
    checksums = manifest.get("output_checksums")
    if not identity_matches or not isinstance(checksums, Mapping) or not checksums:
        return False
    for relative_path, expected_digest in checksums.items():
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            return False
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return False
        if compute_file_sha256(candidate) != expected_digest:
            return False
    return True


def is_output_stale(input_path: str | Path, outputs: Sequence[str | Path]) -> bool:
    """Legacy modification-time helper retained for external callers.

    The canonical CLI deliberately does not call this function; reuse is
    governed only by :func:`manifest_matches_run` and the full run identity.
    """

    try:
        input_mtime = os.path.getmtime(input_path)
    except OSError:
        return True
    for output in outputs:
        try:
            if os.path.getmtime(output) < input_mtime:
                return True
        except OSError:
            return True
    return False
