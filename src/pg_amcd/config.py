import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from importlib.resources import files as _importlib_files
import pywt

from .features import FEATURE_SCHEMA_VERSION

# Default configuration values used as the packaged fallback and as the
# base layer when an explicit configuration file is provided.
DEFAULT_CONFIG = {
    "pipeline_version": "4.0.0",
    "feature_schema_version": "1.0.0",
    "through_stage": 4,
    "sampling_rate": 10000.0,
    "segment_points": 10000,
    "use_physics_gating": True,
    "validation": {
        "sampling_rate_tolerance": 0.05,
        "timestamp_jitter_tolerance": 0.05,
        "minimum_duration_seconds": 1.0,
        "signal_column": 1,
    },
    "preprocessing": {
        "filter_order": 3,
        "low_pass_cutoff_hz": None,
        "scale_percentile": 99.5,
    },
    "ceemdan": {
        "trials": 300,
        "epsilon": 0.02,
        "noise_seed": 42,
        "sifting_iterations": 16,
        "search_cutoffs": [50, 100, 150, 200],
        "search_trials": 15,
        "search_seeds": 3,
        "stability_seeds": [42, 43, 44],
    },
    "maiw": {
        "alpha": 0.25,
        "beta": 0.25,
        "gamma": 0.25,
        "delta": 0.25,
        "chatter_band_center": 1250.0,
        "chatter_band_spread": 500.0,
    },
    "physics_gating": {
        "chatter_energy_weight": 4.0,
        "correlation_weight": 2.0,
        "kurtosis_weight": 1.0,
        "frequency_proximity_weight": 1.0,
        "harmonic_penalty": 5.0,
        "offset": 1.5,
        "harmonic_tolerance_hz": 15.0,
        "harmonic_count": 5,
        "kurtosis_scale": 10.0,
        "selection_threshold": 0.5,
        "include_residual": False,
    },
    "wavelet": {
        "wavelet_name": "db8",
        "level": 4,
        "threshold_mode": "soft",
        "band_aware": True,
        "chatter_threshold_scale": 0.5,
        "noise_threshold_scale": 1.4,
        "minimum_noise_sigma": 1e-12,
    },
    "features": {
        "window_seconds": 1.0,
        "overlap_ratio": 0.75,
        "harmonic_count": 5,
        "harmonic_tolerance_hz": 15.0,
        "sideband_tolerance_hz": 10.0,
        "band_energy_ranges_hz": [[0.0, 250.0], [250.0, 750.0], [750.0, 1750.0], [1750.0, 4000.0]],
    },
    "output": {"png_dpi": 140, "write_svg": True},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    Nested dictionaries are merged key-by-key; all other values (including
    lists) are replaced by the override value.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_packaged_default() -> Dict[str, Any]:
    """Load the packaged default configuration shipped with the package.

    The packaged ``configs/default.json`` is the single source of truth for
    defaults. It is loaded from ``importlib.resources`` when the package is
    installed, or from the source tree when running via ``PYTHONPATH``. The
    in-code ``DEFAULT_CONFIG`` is only used as a last-resort fallback if the
    JSON resource is missing or unreadable.
    """
    candidates = []
    try:
        candidates.append(_importlib_files("pg_amcd") / "configs" / "default.json")
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    candidates.append(Path(__file__).resolve().parent / "configs" / "default.json")

    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
    return copy.deepcopy(DEFAULT_CONFIG)


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load pipeline configuration.

    Resolution rule:

    * Explicit ``--config`` path wins. The file is deep-merged on top of
      the packaged default so that partial configurations still receive
      defaults for keys they omit (e.g. ``wavelet`` / ``maiw``).
    * Without an explicit path, the packaged default is used.

    Arbitrary working-directory searching is intentionally NOT performed.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                file_cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc
        if not isinstance(file_cfg, dict):
            raise ValueError("The configuration root must be a JSON object")
        resolved = _deep_merge(_load_packaged_default(), file_cfg)
        validate_pipeline_config(resolved)
        return resolved

    resolved = _load_packaged_default()
    validate_pipeline_config(resolved)
    return resolved


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """Validate the resolved Stage 1--4 configuration.

    The validation is intentionally strict at the package boundary so invalid
    scientific settings fail before a long CEEMDAN run begins.
    """

    if not isinstance(config, Mapping):
        raise ValueError("The resolved configuration must be a mapping")
    sections: dict[str, Mapping[str, Any]] = {}
    for section_name in (
        "validation",
        "preprocessing",
        "ceemdan",
        "maiw",
        "physics_gating",
        "wavelet",
        "features",
        "output",
    ):
        value = config.get(section_name)
        if not isinstance(value, Mapping):
            raise ValueError(f"{section_name} must be a configuration object")
        sections[section_name] = value
    if not isinstance(config.get("use_physics_gating"), bool):
        raise ValueError("use_physics_gating must be true or false")
    configured_schema = config.get("feature_schema_version")
    if configured_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "feature_schema_version must match the implemented Stage 4 schema "
            f"({FEATURE_SCHEMA_VERSION})"
        )

    fs = float(config.get("sampling_rate", 0.0))
    if not 0.0 < fs < 1.0e7:
        raise ValueError("sampling_rate must be between 0 and 10,000,000 Hz")
    if int(config.get("through_stage", 4)) != 4:
        raise ValueError("The resolved configuration must end at through_stage=4")
    if int(config.get("segment_points", 0)) < 32:
        raise ValueError("segment_points must be at least 32")

    validation = sections["validation"]
    for key in ("sampling_rate_tolerance", "timestamp_jitter_tolerance"):
        value = float(validation.get(key, -1.0))
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"validation.{key} must be in [0, 1)")
    minimum_duration = float(validation.get("minimum_duration_seconds", -1.0))
    if not math.isfinite(minimum_duration) or minimum_duration < 0.0:
        raise ValueError("validation.minimum_duration_seconds must be non-negative")
    signal_column = validation.get("signal_column", -1)
    if (
        isinstance(signal_column, bool)
        or not float(signal_column).is_integer()
        or int(signal_column) < 0
    ):
        raise ValueError("validation.signal_column must be a non-negative integer")

    preprocessing = sections["preprocessing"]
    configured_low_pass = preprocessing.get("low_pass_cutoff_hz")
    low_pass = (
        min(4000.0, fs / 2.0 - 10.0) if configured_low_pass is None else float(configured_low_pass)
    )
    if not 0.0 < low_pass < fs / 2.0:
        raise ValueError("preprocessing.low_pass_cutoff_hz must be below Nyquist")
    if int(preprocessing.get("filter_order", 3)) < 1:
        raise ValueError("preprocessing.filter_order must be at least 1")
    percentile = float(preprocessing.get("scale_percentile", 99.5))
    if not 0.0 < percentile <= 100.0:
        raise ValueError("preprocessing.scale_percentile must be in (0, 100]")

    ceemdan = sections["ceemdan"]
    raw_candidates = ceemdan.get("search_cutoffs", [])
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(raw_candidates, Sequence):
        raise ValueError("ceemdan.search_cutoffs must be a sequence")
    candidates = [float(value) for value in raw_candidates]
    if not candidates:
        raise ValueError("ceemdan.search_cutoffs must contain at least one high-pass cutoff")
    if any(value <= 0.0 or value >= low_pass for value in candidates):
        raise ValueError("Every CEEMDAN search cutoff must lie between 0 and the low-pass cutoff")
    if int(ceemdan.get("trials", 0)) < 1 or int(ceemdan.get("search_trials", 0)) < 1:
        raise ValueError("CEEMDAN trials and search_trials must be positive")
    if float(ceemdan.get("epsilon", 0.0)) <= 0.0:
        raise ValueError("ceemdan.epsilon must be positive")
    for key in ("noise_seed", "sifting_iterations", "search_seeds"):
        value = ceemdan.get(key, -1)
        minimum = 0 if key == "noise_seed" else 1
        if isinstance(value, bool) or not float(value).is_integer() or int(value) < minimum:
            raise ValueError(f"ceemdan.{key} must be an integer of at least {minimum}")

    maiw = sections["maiw"]
    chatter_center = float(maiw.get("chatter_band_center", 0.0))
    chatter_spread = float(maiw.get("chatter_band_spread", 0.0))
    if chatter_center <= 0.0 or chatter_spread <= 0.0:
        raise ValueError("maiw chatter-band centre and spread must be positive")
    if chatter_center - chatter_spread >= fs / 2.0:
        raise ValueError("The configured chatter band does not overlap [0, Nyquist]")
    for key in ("alpha", "beta", "gamma", "delta"):
        value = float(maiw.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"maiw.{key} must be finite and non-negative")

    physics = sections["physics_gating"]
    required_physics = {
        "chatter_energy_weight",
        "correlation_weight",
        "kurtosis_weight",
        "frequency_proximity_weight",
        "harmonic_penalty",
        "offset",
        "harmonic_tolerance_hz",
        "harmonic_count",
        "kurtosis_scale",
        "selection_threshold",
        "include_residual",
    }
    missing_physics = sorted(required_physics - set(physics))
    if missing_physics:
        raise ValueError("physics_gating is missing required keys: " + ", ".join(missing_physics))
    if not 0.0 <= float(physics["selection_threshold"]) <= 1.0:
        raise ValueError("physics_gating.selection_threshold must be in [0, 1]")
    if bool(physics["include_residual"]):
        raise ValueError("physics_gating.include_residual must remain false through Stage 4")
    for key in required_physics - {"include_residual", "harmonic_count"}:
        if not math.isfinite(float(physics[key])):
            raise ValueError(f"physics_gating.{key} must be finite")
    harmonic_count = physics["harmonic_count"]
    if (
        isinstance(harmonic_count, bool)
        or not float(harmonic_count).is_integer()
        or int(harmonic_count) < 1
    ):
        raise ValueError("physics_gating.harmonic_count must be a positive integer")

    wavelet = sections["wavelet"]
    try:
        pywt.Wavelet(str(wavelet.get("wavelet_name", "")))
    except ValueError as exc:
        raise ValueError("wavelet.wavelet_name must identify a discrete wavelet") from exc
    if int(wavelet.get("level", 0)) < 1:
        raise ValueError("wavelet.level must be at least 1")
    if wavelet.get("threshold_mode", "soft") not in {"soft", "hard"}:
        raise ValueError("wavelet.threshold_mode must be 'soft' or 'hard'")
    for key in ("chatter_threshold_scale", "noise_threshold_scale", "minimum_noise_sigma"):
        if float(wavelet.get(key, 0.0)) <= 0.0:
            raise ValueError(f"wavelet.{key} must be positive")

    if not isinstance(wavelet.get("band_aware"), bool):
        raise ValueError("wavelet.band_aware must be true or false")

    features = sections["features"]
    window_seconds = float(features.get("window_seconds", 0.0))
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("features.window_seconds must be finite and positive")
    overlap = float(features.get("overlap_ratio", 0.75))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("features.overlap_ratio must be in [0, 1)")
    feature_harmonic_count = features.get("harmonic_count", 0)
    if (
        isinstance(feature_harmonic_count, bool)
        or not float(feature_harmonic_count).is_integer()
        or int(feature_harmonic_count) < 1
    ):
        raise ValueError("features.harmonic_count must be a positive integer")
    bands = features.get("band_energy_ranges_hz")
    if isinstance(bands, (str, bytes)) or not isinstance(bands, Sequence) or not bands:
        raise ValueError("features.band_energy_ranges_hz must contain frequency pairs")
    for band in bands:
        if isinstance(band, (str, bytes)) or not isinstance(band, Sequence) or len(band) != 2:
            raise ValueError("Each feature energy band must contain [low_hz, high_hz]")
        low, high = float(band[0]), float(band[1])
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high <= fs / 2.0):
            raise ValueError("Feature energy bands must satisfy 0 <= low < high <= Nyquist")

    output = sections["output"]
    dpi = output.get("png_dpi", 0)
    if isinstance(dpi, bool) or not float(dpi).is_integer() or int(dpi) < 36:
        raise ValueError("output.png_dpi must be an integer of at least 36")
    if not isinstance(output.get("write_svg"), bool):
        raise ValueError("output.write_svg must be true or false")
