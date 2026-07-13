import os
import json
from typing import Any, Dict

# Default configuration values in case config.json is missing
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

def load_pipeline_config(config_path: str = None) -> Dict[str, Any]:
    """Loads configuration parameters from JSON.
    
    If config_path is not specified, it searches for config.json in the
    current directory and Python source root. If none is found, fallback to defaults.
    """
    if config_path is not None:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
    # Search paths
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Python")
    ]
    
    for d in search_dirs:
        p = os.path.join(d, "config.json")
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
                
    return DEFAULT_CONFIG.copy()
