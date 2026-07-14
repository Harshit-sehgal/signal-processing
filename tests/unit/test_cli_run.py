"""Fast in-process tests for CLI orchestration and failure accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pg_amcd.cli import run as run_module
from pg_amcd.metadata import MachiningMetadata


def _arguments(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "through_stage": 4,
        "input_dir": str(tmp_path / "inputs"),
        "output_dir": str(tmp_path / "outputs"),
        "config": None,
        "metadata": None,
        "continue_on_error": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _self_checks(*, passed: bool = True) -> dict[str, Any]:
    return {
        f"Stage_{stage}": {
            "unit": {"passed": passed},
            "synthetic": {"passed": passed},
        }
        for stage in range(1, 5)
    }


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        recording_id="recording",
        input_path="",
        metadata={},
        selected_parameters={"cutoff_frequency": 20.0},
        **{
            f"stage_{stage}": SimpleNamespace(runtime_seconds=0.01 * stage) for stage in range(1, 5)
        },
    )


def _patch_fast_run(monkeypatch: pytest.MonkeyPatch, *, self_checks_pass: bool = True) -> None:
    monkeypatch.setattr(
        run_module,
        "load_pipeline_config",
        lambda _path: {
            "sampling_rate": 1000.0,
            "use_physics_gating": False,
            "validation": {},
            "pipeline_version": "4.0.0",
            "feature_schema_version": "1.0.0",
        },
    )
    monkeypatch.setattr(
        run_module,
        "get_environment_info",
        lambda: {"python_version": "3.12", "os": "test", "packages": {"numpy": "2"}},
    )
    monkeypatch.setattr(run_module, "get_git_commit_sha", lambda: "a" * 40)
    monkeypatch.setattr(run_module, "_git_is_dirty", lambda: False)
    monkeypatch.setattr(run_module, "compute_run_id", lambda *_args, **_kwargs: "b" * 64)
    monkeypatch.setattr(
        run_module,
        "create_run_directories",
        lambda path: path.mkdir(parents=True),
    )
    monkeypatch.setattr(
        run_module,
        "run_scientific_self_checks",
        lambda: _self_checks(passed=self_checks_pass),
    )
    monkeypatch.setattr(run_module, "write_recording_artifacts", lambda *_args: {})
    monkeypatch.setattr(run_module, "write_aggregate_stage_4", lambda *_args: {})
    monkeypatch.setattr(run_module, "generate_pipeline_report", lambda *_args: {})


def _make_inputs(tmp_path: Path, names: tuple[str, ...] = ("sample.mat",)) -> Path:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for name in names:
        path = input_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return input_dir


def test_helper_functions_sanitise_ids_checksum_outputs_and_sum_runtime(tmp_path: Path) -> None:
    metadata = MachiningMetadata(recording_id="  odd/name... ", relative_path="sample.mat")
    assert run_module._recording_id("ignored.mat", metadata) == "odd__name"
    assert run_module._recording_id("folder/sample.mat", None) == "folder__sample"
    assert run_module._recording_id("___.mat", None) == "recording"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "artifact.txt").write_text("evidence", encoding="utf-8")
    checksums = run_module._output_checksums(run_dir)
    assert set(checksums) == {"artifact.txt"}
    assert len(checksums["artifact.txt"]) == 64

    result = _fake_result()
    assert run_module._stage_runtime([result], 3) == pytest.approx(0.03)
    result.stage_3 = None
    assert run_module._stage_runtime([result], 3) == 0.0
    assert run_module._selfcheck_passed({"unit": {"passed": True}}, "unit")
    assert not run_module._selfcheck_passed({"unit": True}, "unit")


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("stage", "fixed to --through-stage 4"),
        ("missing-input", "Input directory does not exist"),
        ("empty-input", "No MAT files found"),
    ],
)
def test_run_rejects_invalid_scope_or_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], setup: str, expected: str
) -> None:
    args = _arguments(tmp_path)
    if setup == "stage":
        args.through_stage = 3
    elif setup == "empty-input":
        Path(args.input_dir).mkdir()

    assert run_module.run_pipeline_on_dataset(args) == 2
    assert expected in capsys.readouterr().out


def test_run_records_config_and_metadata_loader_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_inputs(tmp_path)
    args = _arguments(tmp_path, config="missing.json")
    monkeypatch.setattr(
        run_module,
        "load_pipeline_config",
        lambda _path: (_ for _ in ()).throw(ValueError("bad configuration")),
    )
    assert run_module.run_pipeline_on_dataset(args) == 2
    assert "bad configuration" in capsys.readouterr().out

    monkeypatch.setattr(
        run_module,
        "load_pipeline_config",
        lambda _path: {"sampling_rate": 1000.0, "use_physics_gating": False},
    )
    args.metadata = str(tmp_path / "missing.csv")
    assert run_module.run_pipeline_on_dataset(args) == 2
    assert "missing.csv" in capsys.readouterr().out


def test_run_rejects_filesystem_normalised_recording_id_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_inputs(tmp_path, ("a/b.mat", "a__b.mat"))
    _patch_fast_run(monkeypatch)

    assert run_module.run_pipeline_on_dataset(_arguments(tmp_path)) == 2
    assert "not unique" in capsys.readouterr().out


def test_run_rejects_incomplete_identity_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_inputs(tmp_path)
    _patch_fast_run(monkeypatch)
    run_dir = tmp_path / "outputs" / ("b" * 64)
    run_dir.mkdir(parents=True)

    assert run_module.run_pipeline_on_dataset(_arguments(tmp_path)) == 2
    assert "Incomplete or mismatched" in capsys.readouterr().out


def test_failed_self_check_is_recorded_and_aborts_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_inputs(tmp_path)
    _patch_fast_run(monkeypatch, self_checks_pass=False)
    monkeypatch.setattr(
        run_module,
        "validate_and_load_signal",
        lambda *_args, **_kwargs: pytest.fail("recording processing must not start"),
    )

    assert run_module.run_pipeline_on_dataset(_arguments(tmp_path)) == 1
    manifest = json.loads(
        (tmp_path / "outputs" / ("b" * 64) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failures"][0]["stage"] == "self_check"
    assert manifest["stage_evidence"]["Stage_1"]["tests"]["unit"] is False


def test_continue_on_error_records_partial_failure_and_aggregate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_inputs(tmp_path, ("good.mat", "bad.mat"))
    _patch_fast_run(monkeypatch)
    monkeypatch.setattr(
        run_module,
        "validate_and_load_signal",
        lambda path, **_kwargs: (
            np.arange(1000) / 1000.0,
            np.sin(2 * np.pi * 20 * np.arange(1000) / 1000.0),
            1000.0,
        ),
    )

    def process(_time, _signal, _config, *, metadata, mode):
        del metadata, mode
        if not hasattr(process, "called"):
            process.called = True
            raise ValueError("controlled recording failure")
        return _fake_result()

    monkeypatch.setattr(run_module, "process_recording", process)
    monkeypatch.setattr(
        run_module,
        "write_aggregate_stage_4",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("aggregate failed")),
    )
    args = _arguments(tmp_path, continue_on_error=True)

    assert run_module.run_pipeline_on_dataset(args) == 1
    manifest = json.loads(
        (tmp_path / "outputs" / ("b" * 64) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "partial_failure"
    assert manifest["success_count"] == 1
    assert manifest["input_validation"] == {"n_files": 2, "n_valid": 1, "n_invalid": 1}
    assert any(item.get("stage") == "Stage_4" for item in manifest["failures"])


def test_report_failure_changes_an_otherwise_complete_run_to_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_inputs(tmp_path)
    _patch_fast_run(monkeypatch)
    monkeypatch.setattr(
        run_module,
        "validate_and_load_signal",
        lambda *_args, **_kwargs: (
            np.arange(1000) / 1000.0,
            np.sin(2 * np.pi * 20 * np.arange(1000) / 1000.0),
            1000.0,
        ),
    )
    monkeypatch.setattr(run_module, "process_recording", lambda *_args, **_kwargs: _fake_result())
    monkeypatch.setattr(
        run_module,
        "generate_pipeline_report",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("report failed")),
    )

    assert run_module.run_pipeline_on_dataset(_arguments(tmp_path)) == 1
    manifest = json.loads(
        (tmp_path / "outputs" / ("b" * 64) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "partial_failure"
    assert manifest["failure_count"] == 1
    assert manifest["failures"][0]["stage"] == "report"
