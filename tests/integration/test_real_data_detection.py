"""Real-data detection integration smoke test (Segment 6 scaffolding).

``testing/t1/`` ships three processed experiment outputs whose filenames carry
the real machining state label: ``c_*`` = chatter, ``s_*`` = stable. Each
``*_IMFs.npz`` contains the raw ``original_signal``, the CEEMDAN ``imfs``, and
the segment ``start_index``; the matching ``*_Clean.mat`` carries the pipeline's
denoised physical signal.

This is a *tiny* (3-sample) corpus, far below the Segment 6 accuracy targets, so
it is an integration smoke test: it proves the detection scaffolding
(features -> grouped training -> evaluation -> calibration) runs end-to-end on
real signal shapes with real labels, without fabricating any data. It does NOT
claim detection accuracy.

Because the corpus has only one chatter sample, a strict ``GroupKFold`` split
places that single positive in a test fold, leaving the training fold
single-class. Solvers that require >=2 classes per fold (logistic regression,
gradient boosting, SVM) are therefore exercised on synthetic data in
``test_detection.py``; here the grouped path is exercised with the
single-class-tolerant random forest, and a logistic regressor is fit on all
real samples (both classes present) to cover predict/evaluate/calibrate.
"""
import os
import glob

import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pg_amcd.features import extract_window_features
from pg_amcd.detection import (
    train_baseline_classifiers,
    predict_window_probabilities,
    evaluate_detector,
    fit_probability_calibrator,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "testing", "t1")
FS = 10_000.0  # matches the pipeline default that produced these artifacts


def _load_recording(npz_path):
    base = os.path.basename(npz_path).replace("_IMFs.npz", "")
    clean_mat = os.path.join(DATA_DIR, base + "_Clean.mat")
    if not os.path.exists(clean_mat):
        return None
    d = np.load(npz_path)
    original = np.asarray(d["original_signal"], dtype=float)
    imfs = np.asarray(d["imfs"], dtype=float)
    clean = np.asarray(loadmat(clean_mat)["tsDS"], dtype=float)
    denoised = clean[:, 1] if clean.ndim == 2 else clean
    label = 1 if base.startswith("c") else 0
    return {
        "name": base,
        "raw": original,
        "imfs": imfs,
        "denoised": denoised,
        "label": label,
    }


def _collect():
    recs = []
    for npz in sorted(glob.glob(os.path.join(DATA_DIR, "*_IMFs.npz"))):
        rec = _load_recording(npz)
        if rec is not None:
            recs.append(rec)
    return recs


def test_real_data_detection_runs_end_to_end():
    recs = _collect()
    if not recs:
        pytest.skip("testing/t1 real-data artifacts not present")

    feats = []
    keys = None
    y = []
    groups = []
    for rec in recs:
        f = extract_window_features(
            rec["raw"], rec["raw"], rec["denoised"], rec["imfs"], FS, 600.0, 1
        )
        if keys is None:
            keys = list(f.keys())
        feats.append([float(f[k]) for k in keys])
        y.append(rec["label"])
        groups.append(rec["name"])

    X = np.asarray(feats, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    # No NaNs/Infs in the feature matrix built from real signals.
    assert np.all(np.isfinite(X)), "non-finite features from real data"
    assert X.shape[0] == len(recs)

    # Grouped CV on real data (random forest tolerates single-class folds).
    rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
    results = train_baseline_classifiers(X, y, groups=groups, models={"random_forest": rf})
    assert "random_forest" in results
    # Grouped CV ran on real data; degenerate single-class folds are skipped.
    assert len(results["random_forest"]["cv_metrics"]) >= 1

    rf_model = results["random_forest"]["model"]
    proba_rf = predict_window_probabilities(X, rf_model)
    assert proba_rf.shape == (len(recs),)
    assert np.all(np.isfinite(proba_rf))
    assert np.all((proba_rf >= 0.0) & (proba_rf <= 1.0))

    # Full predict/evaluate/calibrate integration on real samples (both
    # classes present when fit on the whole corpus).
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X, y)
    proba = lr.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = evaluate_detector(y, pred, proba)
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics
        assert np.isfinite(metrics[key])

    cal = fit_probability_calibrator(y, proba, method="isotonic")
    cal_proba = cal(proba)
    assert np.all(np.isfinite(cal_proba))
    assert np.all((cal_proba >= 0.0) & (cal_proba <= 1.0))


def test_real_data_labels_are_meaningful():
    recs = _collect()
    if not recs:
        pytest.skip("testing/t1 real-data artifacts not present")
    labels = [r["label"] for r in recs]
    # Both classes present: at least one chatter and one stable recording.
    assert set(labels) == {0, 1}
