import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ablation_runner_runs_baselines_only(tmp_path):
    """The harness runs synthetic baselines and writes JSON/Markdown/PNG even without real data."""
    script = os.path.join(ROOT, "scripts", "run_ablation_and_baselines.py")
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--out-dir",
            str(out_dir),
            "--n-signals",
            "2",
            "--duration",
            "0.3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "ablation_and_baseline_results.json").exists()
    assert (out_dir / "ablation_and_baseline_report.md").exists()
    assert (out_dir / "ablation_summary.png").exists()

    data = json.loads((out_dir / "ablation_and_baseline_results.json").read_text())
    methods = data["synthetic_baselines"]["methods"]
    assert "full_proposed" in methods
    assert data["synthetic_baselines"]["best_rmse_method"] in methods
    assert all(method in data["synthetic_baselines"]["aggregated"] for method in methods)
    assert data["real_data_ablations"] == {"full": {}, "no_physics_gating": {}}
