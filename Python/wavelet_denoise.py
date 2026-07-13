"""Legacy wavelet-denoising stage runner.

Retired as an alternate pipeline implementation (architectural objective:
"one source of truth"). The canonical PG-AMCD pipeline now runs end-to-end via
``pg-amcd run``; this module preserves the historical entry-point name while
routing execution through the single CLI entry point.
"""

from pg_amcd.cli import main

if __name__ == "__main__":
    main()
