"""Real-data evaluation on labelled Vibration_Clean recordings (Segment 6/7).

Uses a tiny subset copied from the real dataset (preserving stickout
subdirectories so grouped cross-validation is well-defined) so the test is fast
(~3 s) and does not depend on the full 61-file corpus. Labels come from the
real ``<label>_<rpm>_<feed>.mat`` filename convention; no labels are
fabricated.
"""
import os
import shutil
import tempfile

import numpy as np

from pg_amcd.config import load_pipeline_config
from pg_amcd.evaluation import (
    _align_finite_feature_rows,
    _exploratory_segment_config,
    evaluate_real_dataset,
    evaluate_real_dataset_temporal,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dynamic_feature_alignment_uses_common_finite_values_only():
    rows = [
        {
            "features": {
                "time_rms": 1.0,
                "freq_entropy": 0.5,
                "imf_10_bandwidth_hz": 4.0,
                "nonfinite": np.nan,
            }
        },
        {
            "features": {
                "time_rms": 2.0,
                "freq_entropy": 0.4,
                "imf_7_bandwidth_hz": 3.0,
                "nonfinite": 1.0,
            }
        },
    ]

    keys, matrix, evidence = _align_finite_feature_rows(rows)

    assert keys == ["freq_entropy", "time_rms"]
    np.testing.assert_allclose(matrix, [[0.5, 1.0], [0.4, 2.0]])
    assert evidence["strategy"] == "intersection_of_present_finite_features"
    assert evidence["missing_feature_policy"] == (
        "drop_from_experimental_matrix; never zero-fill"
    )
    assert set(evidence["dropped_features"]) == {
        "imf_10_bandwidth_hz",
        "imf_7_bandwidth_hz",
        "nonfinite",
    }


def test_exploratory_segment_config_is_local_and_duration_aware():
    config = {
        "sampling_rate": 1_000.0,
        "validation": {"minimum_duration_seconds": 1.0},
        "nested": {"unchanged": True},
    }

    local, evidence = _exploratory_segment_config(config, 200)
    local["nested"]["unchanged"] = False

    assert config["validation"]["minimum_duration_seconds"] == 1.0
    assert config["nested"]["unchanged"] is True
    assert local["validation"]["minimum_duration_seconds"] == 0.2
    assert evidence == {
        "source_minimum_duration_seconds": 1.0,
        "effective_minimum_duration_seconds": 0.2,
        "segment_duration_seconds": 0.2,
        "adjusted": True,
        "scope": "experimental_evaluation_segment_only",
    }


def _build_subset(tmp, include_suffix=False):
    """Copy 2 chatter + 2 stable recordings spanning 2 stickouts / 2 rpm bands.

    Files are placed under ``<tmp>/subset/<stickout>/`` so the stickout grouping
    used by leave-one-stickout-out is meaningful, and the rpm selection is
    explicit (570 vs 770) so every leave-one-rpm-out train fold keeps >=2 groups.
    """
    sources = [
        ("chatter", "2p5inch_stickout", "c_570_014.mat"),
        ("stable", "2p5inch_stickout", "s_570_015.mat"),
        ("chatter", "3p5inch_stickout", "c_770_015.mat"),
        ("stable", "3p5inch_stickout", "s_770_010.mat"),
    ]
    sub = os.path.join(tmp, "subset")
    for _label, st, fname in sources:
        src = os.path.join(ROOT, "Vibration_Clean", st, fname)
        assert os.path.exists(src), f"missing fixture {src}"
        dest_dir = os.path.join(sub, st)
        os.makedirs(dest_dir, exist_ok=True)
        if include_suffix and _label == "chatter" and fname.startswith("c_570_014"):
            # Write a copy whose feed carries a trailing letter; this must NOT
            # be dropped by the numeric-suffix parse path.
            shutil.copy(src, os.path.join(dest_dir, "c_570_015s.mat"))
        else:
            shutil.copy(src, os.path.join(dest_dir, fname))
    return sub


def test_evaluate_real_dataset_subset():
    tmp = tempfile.mkdtemp()
    try:
        sub = _build_subset(tmp)
        cfg = load_pipeline_config(os.path.join(ROOT, "configs", "research_fast.json"))
        res = evaluate_real_dataset(sub, cfg)

        # 4 recordings, 2 chatter + 2 stable (filename-derived labels).
        assert res["n_recordings"] == 4
        assert res["label_counts"]["chatter"] == 2
        assert res["label_counts"]["stable"] == 2

        # All baseline models that fit on the small corpus are trained and
        # evaluated; unfittable models (e.g. SVM on a 2-sample fold) are
        # excluded by design. At least random_forest always fits.
        models = {"logistic_regression", "random_forest", "svm", "gradient_boosting"}
        for level in ("leave_one_recording_out", "leave_one_stickout_out",
                      "leave_one_rpm_out"):
            assert set(res[level].keys()) <= models
            assert "random_forest" in res[level]
            assert len(res[level]) >= 2
            for name, metrics in res[level].items():
                assert metrics is not None
                for mk in ("balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
                    assert np.isfinite(metrics[mk]), f"{level}/{name}/{mk} not finite"

        # recording_id uniqueness prevents cross-recording leakage.
        assert res["n_recordings"] == res["n_windows"]
        alignment = res["feature_alignment"]
        assert alignment["aligned_feature_count"] == len(res["feature_keys"])
        assert alignment["dropped_feature_count"] > 0
        assert any(name.startswith("imf_") for name in alignment["dropped_features"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_real_dataset_recovers_numeric_suffix():
    """A filename with a non-numeric feed suffix must not be dropped."""
    tmp = tempfile.mkdtemp()
    try:
        sub = _build_subset(tmp, include_suffix=True)
        cfg = load_pipeline_config(os.path.join(ROOT, "configs", "research_fast.json"))
        res = evaluate_real_dataset(sub, cfg)
        # The suffixed chatter file is still counted (not silently skipped).
        assert res["label_counts"]["chatter"] == 2
        assert res["label_counts"]["stable"] == 2
        assert res["n_skipped"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_real_dataset_temporal_smoothing():
    """Multi-window eval applies temporal smoothing (Goal 6.5) on real data."""
    tmp = tempfile.mkdtemp()
    try:
        sub = _build_subset(tmp)
        cfg = load_pipeline_config(os.path.join(ROOT, "configs", "research_fast.json"))
        configured_minimum = cfg["validation"]["minimum_duration_seconds"]
        res = evaluate_real_dataset_temporal(sub, cfg, n_windows=3, window_points=2048)

        # 4 recordings -> 4 * 3 windows.
        assert res["n_windows"] == 12
        # Label counts are per-window: 2 chatter + 2 stable recordings x 3 windows.
        assert res["label_counts"]["chatter"] == 6
        assert res["label_counts"]["stable"] == 6

        # Per-window metrics present and finite for random_forest.
        rf = res["per_window_metrics"].get("random_forest", {})
        assert rf, "random_forest per-window metrics missing"
        for mk in ("balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
            assert np.isfinite(rf[mk]), f"per_window {mk} not finite"

        # Temporal smoothing produced a smoothed metric for the best model.
        best = res["best_model"]
        assert best is not None
        assert best in res["smoothed_metrics"]
        sm = res["smoothed_metrics"][best]
        for mk in ("balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
            assert np.isfinite(sm[mk]), f"smoothed {mk} not finite"

        # One smoothed probability per window, none NaN.
        sp = np.asarray(res["smoothed_proba"], dtype=float)
        assert sp.shape[0] == 12
        assert np.all(np.isfinite(sp))

        # Feature importances captured for all 27 features.
        assert len(res["feature_importances"]) == len(res["feature_keys"])
        assert cfg["validation"]["minimum_duration_seconds"] == configured_minimum
        adjustments = res["exploratory_config_adjustments"]
        assert len(adjustments) == 12
        assert all(item["adjusted"] for item in adjustments)
        assert all(
            item["scope"] == "experimental_evaluation_segment_only"
            for item in adjustments
        )
        assert all(
            item["effective_minimum_duration_seconds"]
            == item["segment_duration_seconds"]
            for item in adjustments
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
