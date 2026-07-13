"""Configuration loader tests (Segment 4.1 discovery + 2.4 no silent failures)."""

import json
import pytest

from pg_amcd.config import load_pipeline_config


def test_default_config_has_sampling_rate():
    cfg = load_pipeline_config(None)
    assert "sampling_rate" in cfg
    assert cfg["sampling_rate"] > 0


def test_explicit_config_overrides_default(tmp_path):
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"sampling_rate": 1234.0, "ceemdan": {"trials": 7}}, f)
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg["sampling_rate"] == 1234.0
    assert cfg["ceemdan"]["trials"] == 7
    # default maiw/wavelet still merged in
    assert "maiw" in cfg and "wavelet" in cfg

def test_invalid_json_raises_value_error(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text("{ not valid json ")
    with pytest.raises(ValueError):
        load_pipeline_config(str(cfg_path))


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline_config(str(tmp_path / "does_not_exist.json"))
