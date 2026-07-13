# Mathematical Validation

This document specifies the mathematically rigorous validation checks applied
to every EMD decomposition and reconstruction produced by PG-AMCD
(Research Goal 2). The implementations live in `src/pg_amcd/validation.py` and
are pure functions operating on an IMF matrix `imfs` of shape
`(n_imfs, n_samples)`.

## Metrics

### 1. Reconstruction error (NRMSE)
Normalised root-mean-square error between the original signal and the sum of
its IMFs:

```
NRMSE = ||x - sum_i imf_i||_2 / ||x||_2
```

* **Ideal:** 0.0. A small NRMSE confirms the decomposition is lossless.

### 2. Orthogonality Index (OI)
Measures cross-energy between distinct IMFs:

```
OI = 2 * sum_{i<j} <imf_i, imf_j> / sum_k ||imf_k||^2
```

* **Ideal:** 0.0 (perfectly orthogonal IMFs). Larger values indicate energy
  leakage between modes.

### 3. Mode-Mixing Index (MMI)
Mean absolute Pearson correlation between adjacent IMFs:

```
MMI = mean_i |corr(imf_i, imf_{i+1})|
```

* **Ideal:** 0.0. High MMI signals mode mixing (a single oscillatory mode
  split across adjacent IMFs).

### 4. Energy distribution
Per-IMF energy expressed as a percentage of total IMF energy:

```
E_i = ||imf_i||^2 / sum_k ||imf_k||^2 * 100%
```

Used to detect degenerate decompositions where a single IMF dominates or
energy is not spread across physically meaningful modes.

### 5. Frequency-ordering index
Pearson correlation between IMF index and each IMF's spectral centroid
frequency, mapped from `[-1, 1]` to `[0, 1]`:

* **Ideal:** 1.0 — IMF mean frequencies decrease monotonically with index,
  the canonical EMD ordering. Values near 0.0 indicate an inverted or
  disordered spectrum.

## Acceptance thresholds (research targets)

| Metric | Target |
|---|---|
| NRMSE | < 1e-2 |
| Orthogonality Index | < 0.1 |
| Mode-Mixing Index | < 0.3 |
| Frequency-ordering index | > 0.8 |

These thresholds are tracked per dataset in the evaluation reports
(Sprint 6) and gate the chatter-detection experiments.


## Cutoff optimisation (Goals 5.1–5.3)

The preprocessing low-pass cutoff is **not** hard-coded. Every candidate cutoff
is evaluated on the *same* raw max-energy segment (Goal 5.1: never pick a
different window per cutoff), and the cutoff with the lowest 5-component
objective is selected (implementation in `src/pg_amcd/optimization.py`).

```
badness = 0.25 * min(1, spectral_overlap)
        + 0.20 * min(1, max_adj_imf_corr)
        + 0.20 * min(1, |orthogonality_index|)
        + 0.20 * min(1, seed_instability)
        + 0.15 * min(1, chatter_band_distortion)
```

* **Lower is better.** Each term is normalised to `[0, 1]`.
* `seed_instability` is the standard deviation of reconstruction NRMSE across
  `search_seeds` independent CEEMDAN noise realisations (Goal 5.3: multi-seed
  stability). A robust cutoff has low instability.
* The selected cutoff minimises `mean(badness) + 0.20 * seed_instability`
  (the `final_score`). The full per-cutoff table is written to
  `PipelineResult.selected_parameters["cutoff_search"]`.

## Synthetic ground-truth data (Goal 5.4)

`src/pg_amcd/synthetic.py` generates signals with known components
(`forced`, `chatter`, `drift`, `noise`, `clean`) so denoising / detection
quality can be measured against truth. `evaluate_denoising_performance` returns:

| Metric | Meaning |
|---|---|
| `rmse` | Reconstruction error vs. clean reference |
| `snr_db` | SNR of denoised vs. clean (higher better) |
| `spectral_distortion` | L1 distance between normalised spectra |
| `chatter_band_retention` | Chatter-band energy preserved (1.0 = perfect) |
| `noise_band_attenuation` | `1 - noise_band_energy_kept` (higher = more removed) |
| `onset_detection_error` | `|estimated_onset - true_onset|` in seconds |

## Reproducibility & provenance (Goals 4.3 / 4.4)

* Every run records a `config_sha256`, `git_commit`, `git_dirty`,
  per-file `sha256`, and per-file wall-clock runtime in `provenance.json`.
* A deterministic `run_id` is the SHA-256 of `config_sha256 || git_commit ||
  sorted(input_checksums)`, so a re-run only reuses `outputs/<run_id>/` when
  inputs and config are byte-identical.
* Stale-output detection skips a file when all outputs exist and are newer
  than the input (mtime based), making re-runs idempotent.
* `pg-amcd validate --input-dir DIR [--config C] [--output R]` runs the strict
  input contract (fs tolerance, finite/monotonic time, non-zero variance,
  minimum duration) non-destructively and exits non-zero if any file fails.
* `--metadata PATH` adds dataset-validation reporting: it loads a CSV/XLSX
  metadata index and prints counts of missing metadata rows, duplicate
  metadata entries (by `file_path`/`recording_id`), missing chatter labels,
  and sampling-rate mismatches. These counts are written into the JSON report
  under `report["metadata"]`.

## Chatter detection scaffolding (Goal 6)

The data-independent parts of Segment 6 are implemented and unit-tested:

* `src/pg_amcd/features.py` — `extract_window_features` computes the full
  Goal 6.2 feature set (time, frequency, IMF, time-frequency domains) and is
  wired into `process_recording`.
* `src/pg_amcd/detection.py` — `temporal_smooth_probabilities` (hysteresis +
  median pre-smoothing + minimum run length, Goal 6.5) and
  `train_baseline_classifiers` / `evaluate_detector` (LogisticRegression,
  RandomForest, SVM, GradientBoosting under `GroupKFold` on `recording_id`,
  Goals 6.3-6.4, no leakage).
* `src/pg_amcd/evaluation.py` — `build_dataset_index` (Excel metadata index,
  Goal 6.1).

What remains **data-blocked** (no fabricated detector): training/evaluating on
a *real* labelled chatter dataset and hitting the detection-accuracy targets.
Until then `process_recording` keeps `chatter_probability = nan`,
`predicted_label = "not_evaluated"`, `confidence = nan`.

## Test suite (Goals 1–5)

The repository ships a layered test suite exercising every module:

* `tests/unit/` — `test_config`, `test_io`, `test_preprocessing`,
  `test_segmentation`, `test_decomposition`, `test_weighting`, `test_denoising`,
  `test_pipeline`, `test_provenance`, `test_validation`, `test_scientific`
  (one file per source module, plus scientific/validation suites).
* `tests/integration/test_cli.py` — end-to-end `run` and `validate` CLI tests,
  including metadata validation and the input-contract rejection path.

Run with `PYTHONPATH=src Python/venv/bin/python -m pytest -q` → **83 passed**.
The ruff / mypy / coverage gates (Goals 2.1, 2.2) are enforced in
`.github/workflows/ci.yml`; they require the dev extras (`pip install -e ".[dev]"`),
which the offline environment cannot install locally.
