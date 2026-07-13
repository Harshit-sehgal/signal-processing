import os
import json
import copy
from pathlib import Path
from typing import Any, Dict, Optional

from importlib.resources import files as _importlib_files

# Default configuration values used as the packaged fallback and as the
# base layer when an explicit configuration file is provided.
DEFAULT_CONFIG = {
    "sampling_rate": 10000.0,
    "segment_points": 10000,
    "ceemdan": {
        "trials": 300,
        "epsilon": 0.02,
        "noise_seed": 42,
        "sifting_iterations": 16,
        "search_cutoffs": [50, 100, 150, 200],
        "search_trials": 15
    },
    "maiw": {
        "alpha": 0.25,
        "beta": 0.25,
        "gamma": 0.25,
        "delta": 0.25,
        "chatter_band_center": 1250.0,
        "chatter_band_spread": 500.0
    },
    "physics_gating": {
        "chatter_energy_weight": 4.0,
        "correlation_weight": 2.0,
        "kurtosis_weight": 1.0,
        "harmonic_penalty": 5.0,
        "offset": 1.5,
        "harmonic_tolerance_hz": 15.0,
        "harmonic_count": 5,
        "kurtosis_scale": 10.0
    },
    "wavelet": {
        "wavelet_name": "db8",
        "level": 4,
        "band_aware": True,
        "chatter_threshold_scale": 0.5,
        "noise_threshold_scale": 1.4
    },
    "pipeline": {
        "fallback_rpm": 570.0,
        "fallback_tooth_count": 1
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    Nested dictionaries are merged key-by-key; all other values (including
    lists) are replaced by the override value.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
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
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc
        return _deep_merge(_load_packaged_default(), file_cfg)

    return _load_packaged_default()
