"""Focused tests for deterministic CLI environment/provenance helpers."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from unittest.mock import Mock

import pytest

from pg_amcd.cli import utils


def test_git_helpers_return_command_results(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        subprocess.CompletedProcess(["git"], 0, stdout="a" * 40 + "\n"),
        subprocess.CompletedProcess(["git"], 0, stdout=" M README.md\n"),
    ]
    runner = Mock(side_effect=responses)
    monkeypatch.setattr(utils.subprocess, "run", runner)

    assert utils.get_git_commit_sha() == "a" * 40
    assert utils._git_is_dirty() is True
    assert runner.call_count == 2


@pytest.mark.parametrize("exception", [OSError("missing"), subprocess.SubprocessError("bad")])
def test_git_helpers_have_explicit_non_repository_fallbacks(
    monkeypatch: pytest.MonkeyPatch, exception: Exception
) -> None:
    monkeypatch.setattr(utils.subprocess, "run", Mock(side_effect=exception))

    assert utils.get_git_commit_sha() == "Unknown"
    assert utils._git_is_dirty() is False


def test_dirty_worktree_digest_covers_tracked_diff_and_untracked_contents(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    untracked = tmp_path / "new.py"
    untracked.write_text("first", encoding="utf-8")

    def runner(*args, **kwargs):
        command = args[0]
        if command[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, stdout=str(tmp_path).encode())
        if command[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"tracked patch")
        return subprocess.CompletedProcess(command, 0, stdout=b"new.py\0")

    monkeypatch.setattr(utils.subprocess, "run", runner)
    first = utils.get_git_worktree_sha256()
    untracked.write_text("second", encoding="utf-8")
    second = utils.get_git_worktree_sha256()

    assert len(first) == len(hashlib.sha256().hexdigest()) == 64
    assert first != second


def test_dirty_worktree_digest_has_explicit_failure_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils.subprocess, "run", Mock(side_effect=OSError("missing")))

    assert utils.get_git_worktree_sha256() == ""


def test_environment_info_records_missing_dependency_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(distribution: str) -> str:
        if distribution == "openpyxl":
            raise utils.importlib_metadata.PackageNotFoundError(distribution)
        return f"test-{distribution}"

    monkeypatch.setattr(utils.importlib_metadata, "version", version)
    info = utils.get_environment_info()

    assert info["packages"]["openpyxl"] == "not-installed"
    assert info["packages"]["numpy"] == "test-numpy"
    assert info["python_version"]
    assert info["python_implementation"]
    assert info["python_executable"]
    assert info["os"]


def test_config_hash_uses_values_and_case_insensitive_lookup() -> None:
    first = {"sampling_rate": 1000, "nested": {"value": 1}}
    second = {"nested": {"value": 1}, "sampling_rate": 1000}

    assert utils._sha256_of_config(first, "ignored.json") == utils._sha256_of_config(second)
    mapping = {"RPM": 600, "Tooth_Count": 2}
    assert utils.get_case_insensitive(mapping, ["rpm"]) == 600
    assert utils.get_case_insensitive(mapping, ["missing"], "fallback") == "fallback"
    assert utils.get_case_insensitive(None, ["rpm"], "fallback") == "fallback"


def test_legacy_metadata_loader_is_a_typed_csv_wrapper(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "rpm"])
        writer.writeheader()
        writer.writerow({"relative_path": "sample.mat", "rpm": 600})

    assert utils._load_metadata_index(str(path)) == [{"relative_path": "sample.mat", "rpm": "600"}]
