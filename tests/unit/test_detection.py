"""Chatter-detection tests (Goals 6.3-6.5).

Temporal smoothing is fully data-free and tested directly. Classifier
training/evaluation is tested on synthetic labelled data (no real chatter
labels are fabricated); it requires a real dataset to be meaningful.
"""

import numpy as np
import pytest

from pg_amcd.detection import (
    median_smooth,
    temporal_smooth_probabilities,
    train_baseline_classifiers,
    predict_window_probabilities,
    evaluate_detector,
    fit_probability_calibrator,
)


def test_median_smooth_basic():
    probs = [0.0, 0.0, 1.0, 0.0, 0.0]
    out = median_smooth(probs, window=3)
    # central window becomes 0.0 (median of [0,1,0]); edges stay 0
    assert np.allclose(out, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_temporal_smooth_sustained_burst():
    probs = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1])
    labels, _ = temporal_smooth_probabilities(
        probs, enter_threshold=0.75, exit_threshold=0.40, min_positive_windows=3
    )
    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_temporal_smooth_short_spike_suppressed():
    probs = np.array([0.1, 0.9, 0.9, 0.1])  # run length 2 < min 3
    labels, _ = temporal_smooth_probabilities(
        probs, enter_threshold=0.75, exit_threshold=0.40, min_positive_windows=3
    )
    assert labels.tolist() == [0, 0, 0, 0]


def test_temporal_smooth_hysteresis_exit():
    probs = np.array([0.1, 0.9, 0.5, 0.1])  # median_window=1 keeps raw
    labels, _ = temporal_smooth_probabilities(
        probs, enter_threshold=0.75, exit_threshold=0.40,
        min_positive_windows=1, median_window=1,
    )
    assert labels.tolist() == [0, 1, 1, 0]


def test_train_baseline_classifiers_grouped():
    rng = np.random.default_rng(0)
    X0 = rng.normal(-1.0, 0.3, (50, 4))
    X1 = rng.normal(1.0, 0.3, (50, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    groups = np.array([i // 10 for i in range(100)])  # 10 recordings
    results = train_baseline_classifiers(X, y, groups=groups)
    assert set(results) == {
        "logistic_regression", "random_forest", "svm", "gradient_boosting"
    }
    for name, r in results.items():
        assert "balanced_accuracy" in r["mean_metrics"]
        assert r["mean_metrics"]["balanced_accuracy"] > 0.8
        assert len(r["cv_metrics"]) >= 2


def test_predict_and_evaluate_detector():
    rng = np.random.default_rng(1)
    X0 = rng.normal(-1.0, 0.3, (40, 4))
    X1 = rng.normal(1.0, 0.3, (40, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * 40 + [1] * 40)
    results = train_baseline_classifiers(X, y, groups=np.zeros(len(y), dtype=int))
    model = results["random_forest"]["model"]
    proba = predict_window_probabilities(X, model)
    pred = (proba >= 0.5).astype(int)
    metrics = evaluate_detector(y, pred, proba)
    assert metrics["balanced_accuracy"] > 0.9
    assert "roc_auc" in metrics


def test_fit_probability_calibrator():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200)
    # Raw scores correlated with the label but miscalibrated
    y_proba = y_true * 0.6 + rng.normal(0.0, 0.25, 200)
    cal_f = fit_probability_calibrator(y_true, y_proba, method="isotonic")
    cal = cal_f(y_proba)
    assert np.all(np.isfinite(cal))
    assert np.all(cal >= 0.0) and np.all(cal <= 1.0)
    # Calibrated probabilities should track the raw ordering
    assert np.corrcoef(y_proba, cal)[0, 1] > 0.5


def test_fit_probability_calibrator_methods():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 120)
    y_proba = y_true * 0.5 + rng.normal(0.0, 0.2, 120)
    for method in ("isotonic", "sigmoid"):
        cal_f = fit_probability_calibrator(y_true, y_proba, method=method)
        out = cal_f(y_proba)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
    with pytest.raises(ValueError):
        fit_probability_calibrator(y_true, y_proba, method="nope")
    with pytest.raises(ValueError):
        fit_probability_calibrator(y_true[:-1], y_proba, method="isotonic")
