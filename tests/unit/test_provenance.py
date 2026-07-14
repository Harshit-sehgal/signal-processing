import json
import os
import time

import pytest

from pg_amcd.provenance import (
    compute_file_sha256,
    compute_run_id,
    is_output_stale,
    manifest_matches_run,
)


def test_sha256_deterministic(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert compute_file_sha256(str(p)) == compute_file_sha256(str(p))


def test_sha256_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"1")
    b = tmp_path / "b.bin"
    b.write_bytes(b"2")
    assert compute_file_sha256(str(a)) != compute_file_sha256(str(b))


def test_is_output_stale_missing(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"d")
    assert is_output_stale(str(inp), [str(tmp_path / "missing")]) is True


def test_is_output_stale_fresh(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"d")
    out = tmp_path / "o.npz"
    out.write_bytes(b"x")
    older = time.time() - 10
    os.utime(str(inp), (older, older))
    os.utime(str(out), (time.time(), time.time()))
    assert is_output_stale(str(inp), [str(out)]) is False


def test_is_output_stale_for_missing_input_or_older_output(tmp_path):
    assert is_output_stale(tmp_path / "missing.mat", []) is True

    input_path = tmp_path / "input.mat"
    output_path = tmp_path / "output.npz"
    input_path.write_bytes(b"new")
    output_path.write_bytes(b"old")
    older = time.time() - 10
    os.utime(output_path, (older, older))
    assert is_output_stale(input_path, [output_path]) is True


def test_compute_run_id_deterministic():
    a = compute_run_id("c", "g", ["a", "b"])
    b = compute_run_id("c", "g", ["a", "b"])
    assert a == b and isinstance(a, str) and len(a) > 0


def _full_run_identity(**overrides):
    identity = {
        "config_sha256": {"sampling_rate": 1000.0, "through_stage": 4},
        "git_commit": "a" * 40,
        "input_checksums": {"a.mat": "1" * 64, "nested/b.mat": "2" * 64},
        "metadata_checksum": "3" * 64,
        "dependency_versions": {"numpy": "2.0.0", "scipy": "1.14.0"},
        "pipeline_version": "4.0.0",
        "feature_schema_version": "1.0.0",
        "git_worktree_sha256": "4" * 64,
    }
    identity.update(overrides)
    return compute_run_id(**identity)


@pytest.mark.parametrize(
    "changed",
    [
        {"config_sha256": {"sampling_rate": 2000.0, "through_stage": 4}},
        {"git_commit": "b" * 40},
        {"input_checksums": {"a.mat": "9" * 64, "nested/b.mat": "2" * 64}},
        {"metadata_checksum": "8" * 64},
        {"dependency_versions": {"numpy": "2.1.0", "scipy": "1.14.0"}},
        {"pipeline_version": "4.0.1"},
        {"feature_schema_version": "1.1.0"},
        {"git_worktree_sha256": "5" * 64},
    ],
    ids=[
        "resolved-config",
        "git-commit",
        "input-checksums",
        "metadata-checksum",
        "dependencies",
        "pipeline-version",
        "feature-schema-version",
        "git-worktree-sha256",
    ],
)
def test_run_id_changes_for_every_scientific_identity_dimension(changed):
    baseline = _full_run_identity()

    assert _full_run_identity(**changed) != baseline
    assert len(baseline) == 64
    int(baseline, 16)


def test_run_id_is_order_independent_for_mapping_inputs():
    first = _full_run_identity(
        config_sha256={"through_stage": 4, "sampling_rate": 1000.0},
        input_checksums={"nested/b.mat": "2" * 64, "a.mat": "1" * 64},
        dependency_versions={"scipy": "1.14.0", "numpy": "2.0.0"},
    )

    assert first == _full_run_identity()


def test_manifest_matches_only_completed_identity(tmp_path):
    manifest_path = tmp_path / "run_manifest.json"
    artifact = tmp_path / "Stage_4" / "features.csv"
    artifact.parent.mkdir()
    artifact.write_text("rms\n1.0\n", encoding="utf-8")
    complete = {
        "run_id": "expected",
        "status": "completed",
        "end_timestamp": "2026-07-14T00:00:01+00:00",
        "output_checksums": {
            "Stage_4/features.csv": compute_file_sha256(artifact),
        },
    }
    manifest_path.write_text(json.dumps(complete), encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is True

    for patch in (
        {"run_id": "different"},
        {"status": "running"},
        {"status": "partial_failure"},
        {"end_timestamp": None},
    ):
        manifest_path.write_text(json.dumps({**complete, **patch}), encoding="utf-8")
        assert manifest_matches_run(tmp_path, "expected") is False


def test_manifest_matching_rejects_missing_modified_or_unsafe_outputs(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    complete = {
        "run_id": "expected",
        "status": "completed",
        "end_timestamp": "2026-07-14T00:00:01+00:00",
        "output_checksums": {"artifact.json": compute_file_sha256(artifact)},
    }
    manifest_path.write_text(json.dumps(complete), encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is True

    artifact.write_text('{"changed": true}', encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is False

    complete["output_checksums"] = {"missing.json": "0" * 64}
    manifest_path.write_text(json.dumps(complete), encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is False

    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    complete["output_checksums"] = {"../outside.json": compute_file_sha256(outside)}
    manifest_path.write_text(json.dumps(complete), encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is False

    complete["output_checksums"] = {"artifact.json": 1}
    manifest_path.write_text(json.dumps(complete), encoding="utf-8")
    assert manifest_matches_run(tmp_path, "expected") is False


@pytest.mark.parametrize("payload", ["not-json", "[]", "null"])
def test_manifest_matching_rejects_malformed_or_non_object_json(tmp_path, payload):
    (tmp_path / "run_manifest.json").write_text(payload, encoding="utf-8")

    assert manifest_matches_run(tmp_path, "expected") is False
