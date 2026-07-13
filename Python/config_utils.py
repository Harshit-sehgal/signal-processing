import os
import json

def load_pipeline_config():
    default_config = {
        "sampling_rate": 10000.0,
        "segment_points": 10000,
        "ceemdan": {
            "trials": 300,
            "epsilon": 0.05,
            "noise_seed": 42,
            "search_trials": 50,
            "search_cutoffs": [50, 100, 150, 200]
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
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            # Merge with default to ensure no missing keys
            for key, val in default_config.items():
                if key not in user_config:
                    user_config[key] = val
                elif isinstance(val, dict):
                    for subkey, subval in val.items():
                        if subkey not in user_config[key]:
                            user_config[key][subkey] = subval
            return user_config
        except Exception as e:
            print(f"Error loading config.json: {e}. Using default values.")
            return default_config
    return default_config
