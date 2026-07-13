"""End-to-end CLI smoke test for the PG-AMCD pipeline.

Generates a synthetic ``tsDS`` MAT file, a lightweight configuration, runs
the real ``pg-amcd`` CLI, and asserts that the run completes successfully
with finite IMF / MAIW / Clean outputs and a valid provenance record.
"""
import os
import sys
import json
import subprocess

import numpy as np
import pytest
import scipy.io

# Repository root = tests/integration/../../  (two levels up)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")

# Lightweight configuration with reduced CEEMDAN settings for speed.
# ``wavelet`` / ``maiw`` are intentionally omitted; the config loader
# deep-merges them from the packaged default.
TEST_CONFIG = {
    "sampling_rate": 1000,
    "segment_points": 1000,
    "ceemdan": {
        "trials": 2,
        "search_trials": 1,
        "epsilon": 0.02,
        "noise_seed": 42,
        "sifting_iterations": 2,
        "search_cutoffs": [20],
    },
}


def _make_synthetic_mat(path: str, fs: float = 1000.0, n_samples: int = 2000) -> None:
    """Write a synthetic two-column ``tsDS`` MAT file.

    Columns are [time, signal] with a chatter-like sinusoid plus broadband
    noise so that CEEMDAN produces a meaningful set of IMFs.
    """
    rng = np.random.default_rng(1234)
    t = np.arange(n_samples) / fs
    signal = (
        0.6 * np.sin(2 * np.pi * 40.0 * t)          # tooth-passing harmonic
        + 0.4 * np.sin(2 * np.pi * 320.0 * t)       # chatter resonance
        + rng.normal(0.0, 0.15, n_samples)           # broadband noise
    )
    tsDS = np.column_stack((t, signal))
    scipy.io.savemat(path, {"tsDS": tsDS})


@pytest.fixture()
def cli_fixture_dir(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # 1. Synthetic tsDS MAT file
    mat_path = input_dir / "sample.mat"
    _make_synthetic_mat(str(mat_path), fs=1000.0, n_samples=2000)

    # 2. Lightweight configuration
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CONFIG, f)

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
    }


def _run_cli(input_dir, output_dir, config_path):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pg_amcd.cli",
            "run",
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
            "--config",
            config_path,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc


def test_cli_run_end_to_end(cli_fixture_dir):
    input_dir = cli_fixture_dir["input_dir"]
    output_dir = cli_fixture_dir["output_dir"]
    config_path = cli_fixture_dir["config_path"]

    proc = _run_cli(input_dir, output_dir, config_path)

    # 5. Process exits successfully
    assert proc.returncode == 0, (
        f"CLI exited with code {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # 4. Output files and provenance exist
    provenance_path = os.path.join(output_dir, "provenance.json")
    assert os.path.exists(provenance_path), "provenance.json missing"

    imf_files = []
    maiw_files = []
    clean_files = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            full = os.path.join(root, name)
            if name.endswith("_IMFs.npz"):
                imf_files.append(full)
            elif name.endswith("_Clean.mat"):
                clean_files.append(full)
            elif name.endswith(".mat") and not name.endswith("_Clean.mat"):
                # MAIW output keeps the original base name (e.g. sample.mat)
                maiw_files.append(full)

    assert imf_files, "IMF output missing"
    assert maiw_files, "MAIW output missing"
    assert clean_files, "Clean output missing"

    # 3. Provenance reports zero failures
    with open(provenance_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)
    assert provenance.get("failure_count", 1) == 0, (
        f"Failures in provenance: {provenance.get('failure_count')} "
        f"-> {provenance.get('failures')}"
    )
    assert provenance.get("success_count", 0) >= 1

    # All arrays contain finite values
    imf_path = imf_files[0]
    data = np.load(imf_path)
    for key in ("time", "original_signal", "imfs"):
        assert np.all(np.isfinite(data[key])), f"Non-finite values in IMF {key}"

    clean_path = clean_files[0]
    clean_mat = scipy.io.loadmat(clean_path)
    assert np.all(np.isfinite(clean_mat["tsDS"])), "Non-finite values in Clean output"

    maiw_path = maiw_files[0]
    maiw_mat = scipy.io.loadmat(maiw_path)
    assert np.all(np.isfinite(maiw_mat["tsDS"])), "Non-finite values in MAIW output"


def test_cli_help():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-m", "pg_amcd.cli", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0


def _run_validate(input_dir, config_path, output=None):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + existing if existing else "")
    cmd = [
        sys.executable,
        "-m",
        "pg_amcd.cli",
        "validate",
        "--input-dir",
        input_dir,
        "--config",
        config_path,
    ]
    if output is not None:
        cmd += ["--output", output]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    return proc


def test_cli_validate_end_to_end(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(str(input_dir / "sample.mat"), fs=1000.0, n_samples=2000)
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CONFIG, f)
    report_path = tmp_path / "report.json"
    proc = _run_validate(str(input_dir), str(config_path), str(report_path))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text())
    assert report["n_files"] == 1
    assert report["n_valid"] == 1
    assert report["n_invalid"] == 0
    assert report["files"][0]["valid"] is True
    assert report["files"][0]["fs_estimated"] == pytest.approx(1000.0, rel=0.05)


def test_cli_validate_rejects_invalid(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Time spacing implies 2000 Hz, deviating >5% from the configured 1000 Hz.
    _make_synthetic_mat(str(input_dir / "bad.mat"), fs=2000.0, n_samples=2000)
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CONFIG, f)
    proc = _run_validate(str(input_dir), str(config_path))
    assert proc.returncode == 1, proc.stdout
    assert "0/1" in proc.stdout


def test_cli_validate_help():
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-m", "pg_amcd.cli", "validate", "--help"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0


def test_cli_validate_with_metadata(tmp_path):
    import csv
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_synthetic_mat(str(input_dir / "sample.mat"), fs=1000.0, n_samples=2000)
    _make_synthetic_mat(str(input_dir / "orphan.mat"), fs=1000.0, n_samples=2000)
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CONFIG, f)
    meta_path = tmp_path / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "label"])
        w.writeheader()
        w.writerow({"file_path": "sample.mat", "label": "chatter"})
        w.writerow({"file_path": "sample.mat", "label": "stable"})  # duplicate entry
        w.writerow({"file_path": "other.mat", "label": ""})        # missing label
    report_path = tmp_path / "report.json"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-m", "pg_amcd.cli", "validate",
         "--input-dir", str(input_dir), "--config", str(config_path),
         "--metadata", str(meta_path), "--output", str(report_path)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text())
    assert report["n_files"] == 2
    meta = report["metadata"]
    assert meta["missing_metadata"] == 1
    assert meta["duplicate_metadata_entries"] == 1
    assert meta["missing_chatter_label"] == 1
    assert "Metadata validation" in proc.stdout
