"""Unit tests for the progress tracking module."""

import os

import pytest

from pg_amcd.tracking import (
    ProjectScorecard,
    calculate_scorecard,
    save_scorecard,
    append_score_history,
    load_score_history,
    score_correctness,
)


def test_project_scorecard_overall():
    card = ProjectScorecard(
        architecture=80.0,
        correctness=70.0,
        input_validation=90.0,
        reproducibility=60.0,
        mathematical_validation=50.0,
        chatter_detection=40.0,
        research_readiness=30.0,
        visualisation=20.0,
    )
    assert card.overall == pytest.approx(55.0, rel=1e-3)


def test_calculate_scorecard_with_empty_artifacts():
    card, details = calculate_scorecard()
    assert 0.0 <= card.architecture <= 100.0
    assert 0.0 <= card.correctness <= 100.0
    assert 0.0 <= card.input_validation <= 100.0
    assert 0.0 <= card.reproducibility <= 100.0
    assert 0.0 <= card.mathematical_validation <= 100.0
    assert 0.0 <= card.chatter_detection <= 100.0
    assert 0.0 <= card.research_readiness <= 100.0
    assert 0.0 <= card.visualisation <= 100.0
    assert "architecture" in details


def test_score_correctness_with_check_results():
    score, details = score_correctness(
        check_results={
            "tests_returncode": 0,
            "lint_returncode": 0,
            "type_returncode": 0,
            "tests_passed": 100,
        }
    )
    assert score == 80.0
    assert details["tests_returncode"] == 0
    assert details["lint_returncode"] == 0
    assert details["type_returncode"] == 0


def test_score_correctness_with_failed_checks():
    score, details = score_correctness(
        check_results={
            "tests_returncode": 1,
            "lint_returncode": 1,
            "type_returncode": 1,
            "tests_passed": 0,
        }
    )
    assert score < 80.0
    assert details["tests_returncode"] == 1


def test_save_scorecard(tmp_path):
    card = ProjectScorecard(
        architecture=80.0,
        correctness=70.0,
        input_validation=90.0,
        reproducibility=60.0,
        mathematical_validation=50.0,
        chatter_detection=40.0,
        research_readiness=30.0,
        visualisation=20.0,
    )
    path = save_scorecard(card, str(tmp_path), {"detail": "value"})
    assert os.path.exists(path)
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["scorecard"]["architecture"] == 80.0
    assert data["overall"] == card.overall
    assert data["details"]["detail"] == "value"


def test_score_history_roundtrip(tmp_path):
    history_path = os.path.join(str(tmp_path), "history.json")
    card = ProjectScorecard(
        architecture=80.0,
        correctness=70.0,
        input_validation=90.0,
        reproducibility=60.0,
        mathematical_validation=50.0,
        chatter_detection=40.0,
        research_readiness=30.0,
        visualisation=20.0,
    )
    append_score_history(card, {"detail": "value"}, history_path)
    history = load_score_history(history_path)
    assert len(history) == 1
    assert history[0].scorecard.architecture == 80.0
