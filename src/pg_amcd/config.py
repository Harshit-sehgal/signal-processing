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
    "wavelet": {
        "wavelet_name": "db8",
        "level": 4
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

    Tries ``importlib.resources`` first (works for installed/editable
    packages), then falls back to the file path relative to this module
    (works when running from a source tree via ``PYTHONPATH``). If the
    resource is missing or unreadable, ``DEFAULT_CONFIG`` is used.
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
