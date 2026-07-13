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


def test_default_config_has_centralized_physics_and_wavelet_keys():
    cfg = load_pipeline_config(None)
    assert "physics_gating" in cfg
    assert "wavelet" in cfg
    assert "pipeline" in cfg
    pg = cfg["physics_gating"]
    assert pg["chatter_energy_weight"] == 4.0
    assert pg["correlation_weight"] == 2.0
    assert pg["kurtosis_weight"] == 1.0
    assert pg["harmonic_penalty"] == 5.0
    assert pg["offset"] == 1.5
    assert pg["harmonic_tolerance_hz"] == 15.0
    assert pg["harmonic_count"] == 5
    assert pg["kurtosis_scale"] == 10.0
    wavelet = cfg["wavelet"]
    assert wavelet["band_aware"] is True
    assert wavelet["chatter_threshold_scale"] == 0.5
    assert wavelet["noise_threshold_scale"] == 1.4
    pipeline = cfg["pipeline"]
    assert pipeline["fallback_rpm"] == 570.0
    assert pipeline["fallback_tooth_count"] == 1


def test_explicit_config_can_override_centralized_keys(tmp_path):
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "physics_gating": {"kurtosis_scale": 20.0, "offset": 2.0},
                "wavelet": {"chatter_threshold_scale": 0.3, "noise_threshold_scale": 1.8},
                "pipeline": {"fallback_rpm": 1000.0, "fallback_tooth_count": 4},
            },
            f,
        )
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg["physics_gating"]["kurtosis_scale"] == 20.0
    assert cfg["physics_gating"]["offset"] == 2.0
    assert cfg["wavelet"]["chatter_threshold_scale"] == 0.3
    assert cfg["wavelet"]["noise_threshold_scale"] == 1.8
    assert cfg["pipeline"]["fallback_rpm"] == 1000.0
    assert cfg["pipeline"]["fallback_tooth_count"] == 4
    # Other defaults are preserved by deep merge.
    assert cfg["physics_gating"]["harmonic_count"] == 5
    assert cfg["wavelet"]["wavelet_name"] == "db8"
