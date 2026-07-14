"""Configuration loader tests (Segment 4.1 discovery + 2.4 no silent failures)."""

import copy
from importlib.resources import files as resource_files
import json
from pathlib import Path

import pytest

from pg_amcd.config import DEFAULT_CONFIG, load_pipeline_config, validate_pipeline_config


ROOT = Path(__file__).resolve().parents[2]


def test_default_config_has_sampling_rate():
    cfg = load_pipeline_config(None)
    assert "sampling_rate" in cfg
    assert cfg["sampling_rate"] > 0


def test_feature_schema_identity_cannot_diverge_from_implementation():
    config = copy.deepcopy(load_pipeline_config(None))
    config["feature_schema_version"] = "9.9.9"

    with pytest.raises(ValueError, match=r"implemented Stage 4 schema \(1.0.0\)"):
        validate_pipeline_config(config)


def test_packaged_default_config_resource_is_available():
    resource = resource_files("pg_amcd").joinpath("configs", "default.json")

    assert resource.is_file()
    packaged = json.loads(resource.read_text(encoding="utf-8"))
    assert packaged["through_stage"] == 4
    assert packaged["pipeline_version"] == "4.0.0"
    assert "pipeline" not in packaged
    repository_default = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    assert packaged == repository_default
    assert packaged == DEFAULT_CONFIG


def test_explicit_config_overrides_default(tmp_path):
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampling_rate": 1234.0,
                "ceemdan": {"trials": 7},
                "maiw": {"chatter_band_center": 400.0, "chatter_band_spread": 100.0},
                "features": {"band_energy_ranges_hz": [[0.0, 200.0], [200.0, 600.0]]},
            },
            f,
        )
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


def test_low_pass_auto_resolution_is_valid_at_one_kilohertz():
    config = copy.deepcopy(load_pipeline_config(None))
    config["sampling_rate"] = 1000.0
    config["preprocessing"]["low_pass_cutoff_hz"] = None
    config["ceemdan"]["search_cutoffs"] = [50.0, 200.0, 480.0]
    config["maiw"]["chatter_band_center"] = 300.0
    config["maiw"]["chatter_band_spread"] = 100.0
    config["features"]["band_energy_ranges_hz"] = [
        [0.0, 100.0],
        [100.0, 300.0],
        [300.0, 500.0],
    ]

    validate_pipeline_config(config)

    at_automatic_limit = copy.deepcopy(config)
    at_automatic_limit["ceemdan"]["search_cutoffs"] = [490.0]
    with pytest.raises(ValueError, match="between 0 and the low-pass cutoff"):
        validate_pipeline_config(at_automatic_limit)

    at_nyquist = copy.deepcopy(config)
    at_nyquist["preprocessing"]["low_pass_cutoff_hz"] = 500.0
    with pytest.raises(ValueError, match="below Nyquist"):
        validate_pipeline_config(at_nyquist)


def test_repository_profiles_are_valid_and_do_not_invent_physics_metadata():
    test_config = load_pipeline_config(str(ROOT / "configs" / "test.json"))
    research_config = load_pipeline_config(str(ROOT / "configs" / "research_fast.json"))

    assert test_config["through_stage"] == 4
    assert test_config["sampling_rate"] == 1000
    assert test_config["preprocessing"]["low_pass_cutoff_hz"] is None
    assert test_config["use_physics_gating"] is False
    assert research_config["use_physics_gating"] is False


def test_default_config_has_centralized_physics_and_wavelet_keys():
    cfg = load_pipeline_config(None)
    assert "physics_gating" in cfg
    assert "wavelet" in cfg
    assert "pipeline" not in cfg
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


def test_explicit_config_can_override_centralized_keys(tmp_path):
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "physics_gating": {"kurtosis_scale": 20.0, "offset": 2.0},
                "wavelet": {"chatter_threshold_scale": 0.3, "noise_threshold_scale": 1.8},
            },
            f,
        )
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg["physics_gating"]["kurtosis_scale"] == 20.0
    assert cfg["physics_gating"]["offset"] == 2.0
    assert cfg["wavelet"]["chatter_threshold_scale"] == 0.3
    assert cfg["wavelet"]["noise_threshold_scale"] == 1.8
    # Other defaults are preserved by deep merge.
    assert cfg["physics_gating"]["harmonic_count"] == 5
    assert cfg["wavelet"]["wavelet_name"] == "db8"


def _set_nested(config, dotted_path, value):
    target = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("use_physics_gating", "yes", "use_physics_gating"),
        ("sampling_rate", 0.0, "sampling_rate"),
        ("through_stage", 5, "through_stage=4"),
        ("segment_points", 31, "segment_points"),
        ("validation.sampling_rate_tolerance", 1.0, "sampling_rate_tolerance"),
        ("validation.timestamp_jitter_tolerance", -0.1, "timestamp_jitter_tolerance"),
        ("validation.minimum_duration_seconds", -1.0, "minimum_duration_seconds"),
        ("validation.signal_column", 1.5, "signal_column"),
        ("preprocessing.low_pass_cutoff_hz", 5000.0, "below Nyquist"),
        ("preprocessing.filter_order", 0, "filter_order"),
        ("preprocessing.scale_percentile", 0.0, "scale_percentile"),
        ("ceemdan.search_cutoffs", "50", "must be a sequence"),
        ("ceemdan.search_cutoffs", [], "at least one"),
        ("ceemdan.search_cutoffs", [5000.0], "between 0 and the low-pass"),
        ("ceemdan.trials", 0, "trials and search_trials"),
        ("ceemdan.epsilon", 0.0, "epsilon"),
        ("ceemdan.noise_seed", -1, "noise_seed"),
        ("ceemdan.sifting_iterations", 0, "sifting_iterations"),
        ("ceemdan.search_seeds", True, "search_seeds"),
        ("maiw.chatter_band_center", 0.0, "centre and spread"),
        ("maiw.chatter_band_center", 6000.0, "does not overlap"),
        ("maiw.alpha", -1.0, "maiw.alpha"),
        ("physics_gating.selection_threshold", 2.0, "selection_threshold"),
        ("physics_gating.include_residual", True, "include_residual"),
        ("physics_gating.offset", float("nan"), "offset"),
        ("physics_gating.harmonic_count", 0, "harmonic_count"),
        ("wavelet.wavelet_name", "not-a-wavelet", "wavelet_name"),
        ("wavelet.level", 0, "wavelet.level"),
        ("wavelet.threshold_mode", "median", "threshold_mode"),
        ("wavelet.minimum_noise_sigma", 0.0, "minimum_noise_sigma"),
        ("wavelet.band_aware", 1, "band_aware"),
        ("features.window_seconds", 0.0, "window_seconds"),
        ("features.overlap_ratio", 1.0, "overlap_ratio"),
        ("features.harmonic_count", 0, "features.harmonic_count"),
        ("features.band_energy_ranges_hz", [], "frequency pairs"),
        ("features.band_energy_ranges_hz", [[0.0]], "Each feature energy band"),
        ("features.band_energy_ranges_hz", [[10.0, 5.0]], "low < high"),
        ("output.png_dpi", 20, "png_dpi"),
        ("output.write_svg", "yes", "write_svg"),
    ],
)
def test_invalid_resolved_config_is_rejected(path, value, message):
    config = copy.deepcopy(load_pipeline_config(None))
    _set_nested(config, path, value)

    with pytest.raises(ValueError, match=message):
        validate_pipeline_config(config)


def test_missing_or_non_object_config_sections_are_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_pipeline_config([])

    missing_key = copy.deepcopy(load_pipeline_config(None))
    del missing_key["physics_gating"]["offset"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_pipeline_config(missing_key)

    wrong_section = copy.deepcopy(load_pipeline_config(None))
    wrong_section["wavelet"] = []
    with pytest.raises(ValueError, match="wavelet must be a configuration object"):
        validate_pipeline_config(wrong_section)


@pytest.mark.parametrize("payload", ["[]", '"not-an-object"', "null"])
def test_explicit_config_root_must_be_an_object(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_pipeline_config(str(path))
