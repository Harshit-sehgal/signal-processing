# Legacy Python Scripts

This directory contains early-stage scripts that predate the modular
`src/pg_amcd/` package. They are kept for historical reference and for
reproducing the original exploratory analysis, but they are **not part of the
supported PG-AMCD pipeline**.

## Status

- **Not imported by the package**: `src/pg_amcd/` does not depend on any file
  in this directory.
- **Not covered by CI**: These scripts are not run by the test suite.
- **May drift**: They are not guaranteed to stay in sync with the main package
  API.

## Contents

| Script | Purpose |
|--------|---------|
| `01_preprocess.py` | Early preprocessing exploration |
| `run_pipeline.py` | Pre-refactor monolithic pipeline runner |
| `maiw_weighting.py` | Standalone MAIW weighting experiments |
| `wavelet_denoise.py` | Standalone wavelet denoising experiments |
| `iceemdan.py` | ICEEMDAN exploration |
| `diagnose_decomposition.py` | Decomposition diagnostic plots |
| `visualize_pipeline.py` / `visualize_preprocess.py` | Visualization helpers |
| `config_utils.py` / `config.json` | Early config experiments |
| `test_*.py` / `*.png` / `optimized_params.txt` | Ad-hoc test outputs |

## When to use

If you are reproducing the original research step-by-step or need to understand
the evolution of the implementation, these scripts may be useful. For any
production or reproducible work, use the `pg-amcd` CLI or the `src/pg_amcd/`
Python API instead.
