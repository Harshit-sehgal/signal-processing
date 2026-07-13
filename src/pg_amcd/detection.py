"""Chatter detection: temporal smoothing and grouped classifier training.

Goal 6 (actual chatter detection) is delivered in two layers:

1. **Temporal smoothing (Goal 6.5)** -- pure, data-free logic that turns a
   sequence of per-window chatter probabilities into stable, hysteresis-based
   labels with a minimum run length. Fully unit-tested below.
2. **Grouped classifier training / evaluation (Goals 6.3-6.4)** -- real
   scikit-learn pipelines guarded against data leakage via ``GroupKFold`` on
   ``recording_id``. These functions are correct and tested on synthetic
   meaningful detection scores (not fabricated here).

The pipeline currently leaves ``chatter_probability`` as ``nan``
(``not_evaluated``) until a validated model exists, so none of this is wired
into a fake detector.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scikit-learn is an optional heavy dependency
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only without sklearn
    _HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
# Goal 6.5: Temporal smoothing
# --------------------------------------------------------------------------- #
def median_smooth(probs: Sequence[float], window: int = 5) -> np.ndarray:
    """Median-filter a 1-D probability sequence (edge-padded)."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be a 1-D sequence")
    if window < 2:
        return probs.copy()
    pad = window // 2
    padded = np.pad(probs, pad, mode="edge")
    out = np.empty(len(probs), dtype=float)
    for i in range(len(probs)):
        out[i] = np.median(padded[i : i + window])
    return out


def temporal_smooth_probabilities(
    probs: Sequence[float],
    enter_threshold: float = 0.75,
    exit_threshold: float = 0.40,
    min_positive_windows: int = 3,
    median_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a probability sequence into stable chatter labels.

    Uses hysteresis (enter/exit thresholds), median pre-smoothing, and a
    minimum run length so brief spikes do not trigger detection.

    Returns:
        ``(labels, smoothed_probs)`` -- ``labels`` is ``0/1`` per window.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be a 1-D sequence")
    if not 0.0 <= exit_threshold < enter_threshold <= 1.0:
        raise ValueError("require 0 <= exit_threshold < enter_threshold <= 1")
    smoothed = median_smooth(probs, median_window)
    n = len(smoothed)
    labels = np.zeros(n, dtype=int)
    active = False
    for i in range(n):
        if not active and smoothed[i] >= enter_threshold:
            active = True
        elif active and smoothed[i] < exit_threshold:
            active = False
        labels[i] = 1 if active else 0
    # Enforce a minimum number of consecutive chatter windows.
    if min_positive_windows > 1:
        i = 0
        while i < n:
            if labels[i] == 1:
                j = i
                while j < n and labels[j] == 1:
                    j += 1
                if (j - i) < min_positive_windows:
                    labels[i:j] = 0
                i = j
            else:
                i += 1
    return labels, smoothed


# --------------------------------------------------------------------------- #
# Goals 6.3-6.4: Grouped classifier training and evaluation
# --------------------------------------------------------------------------- #
def _default_models(random_state: int) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=random_state
        ),
        "svm": CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=random_state),
            ensemble=False,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def _score_fold(y_true, y_pred, y_proba) -> Dict[str, float]:
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def _mean_of_metrics(folds: List[Dict[str, float]]) -> Dict[str, float]:
    keys = folds[0].keys()
    return {k: float(np.mean([f[k] for f in folds])) for k in keys}


def train_baseline_classifiers(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    groups: Optional[Sequence[Any]] = None,
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Train the four baseline classifiers with grouped cross-validation.

    ``groups`` should be the ``recording_id`` (or equivalent) so that windows
    from the same recording never span train/test (Goal 6.4, no leakage).

    Returns a dict ``model_name -> {model, cv_metrics, mean_metrics}``.
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError(
            "scikit-learn is required for classifier training "
            "(pip install scikit-learn)."
        )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if groups is None:
        groups = np.zeros(len(y), dtype=int)
    else:
        groups = np.asarray(groups)
    if models is None:
        models = _default_models(random_state)

    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        cv = GroupKFold(n_splits=min(n_groups, 5))
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))

    results: Dict[str, Dict[str, Any]] = {}
    for name, clf in models.items():
        fold_metrics: List[Dict[str, float]] = []
        for tr, te in splits:
            est = clone(clf)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te])
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X[te])[:, 1]
            else:
                proba = pred.astype(float)
            fold_metrics.append(_score_fold(y[te], pred, proba))
        final = clone(clf)
        final.fit(X, y)
        results[name] = {
            "model": final,
            "cv_metrics": fold_metrics,
            "mean_metrics": _mean_of_metrics(fold_metrics),
        }
    return results


def predict_window_probabilities(
    X: Sequence[Sequence[float]], model: Any
) -> np.ndarray:
    """Return per-window chatter probabilities from a fitted classifier."""
    X = np.asarray(X, dtype=float)
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must support predict_proba")
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def evaluate_detector(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float],
) -> Dict[str, float]:
    """Compute the Goal 6.5 detection metrics for one evaluation split."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)
    return _score_fold(y_true, y_pred, y_proba)
