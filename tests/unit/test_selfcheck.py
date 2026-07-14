"""Fast scientific self-check evidence tests."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pg_amcd.selfcheck import run_scientific_self_checks


def test_all_four_stage_self_checks_pass_and_are_manifest_serialisable() -> None:
    evidence = run_scientific_self_checks()

    assert tuple(evidence) == ("Stage_1", "Stage_2", "Stage_3", "Stage_4")
    for stage in evidence.values():
        assert set(stage) == {"unit", "synthetic"}
        for check in stage.values():
            assert check["passed"] is True
            assert isinstance(check.get("details"), Mapping)
            assert check["details"]

    encoded = json.dumps(evidence, allow_nan=False, sort_keys=True)
    assert "Stage_4" in encoded
