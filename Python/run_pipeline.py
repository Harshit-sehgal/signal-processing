"""Legacy master runner.

Retired the multi-stage subprocess orchestration (``iceemdan.py`` ->
``maiw_weighting.py`` -> ``wavelet_denoise.py``) in favour of the single
canonical CLI entry point (``pg-amcd run``). Preserved as a backwards-compatible
entry-point wrapper.
"""

from pg_amcd.cli import main

if __name__ == "__main__":
    main()
