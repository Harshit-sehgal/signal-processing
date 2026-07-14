# PG-AMCD Stage 1–4 validation contract

This document describes the validation performed by the live canonical package. It is a reproducibility contract, not a claim that a particular real-data run achieved a fixed score. Stage scores and measured signal metrics must be read from a completed `outputs/<run_id>/` tree.

## 1. Validation layers

PG-AMCD uses four complementary layers:

1. `pg-amcd validate` checks the raw MAT and metadata contracts without running CEEMDAN.
2. The canonical pipeline checks scientific invariants and records per-stage quantitative metrics.
3. Fast deterministic self-checks supply unit/synthetic evidence in `run_manifest.json`.
4. The artifact scorer verifies the finished run tree, required metrics, tests, figures, provenance, and summaries.

A successful test command is not a substitute for a successful data run, and a generated file is not sufficient evidence unless the scorer can trace its content and manifest evidence.

## 2. Raw-input validation

The canonical loader is `validate_and_load_signal` in `src/pg_amcd/io.py`. For every `.mat` file it requires:

- a numeric, nonempty, two-dimensional `tsDS` matrix with at least two columns;
- time in column 0 and a configured signal column within bounds;
- real, finite time and signal values;
- strictly increasing timestamps with no duplicates;
- timestamp jitter no greater than the configured relative tolerance;
- an estimated sampling rate `1 / median(diff(time))` within the loader's 1 Hz to 10 MHz safety bounds and the configured rate tolerance;
- at least the loader's three-sample minimum and the configured minimum duration;
- a signal that is neither constant nor numerically flat.

The `validate` command writes a JSON report containing per-file SHA-256, validity, estimated sampling rate, sample count, duration, metadata, and errors. It returns 0 only when the report status is `valid`; otherwise it returns 1.

Example:

```bash
pg-amcd validate \
  --input-dir "Vibration - ML" \
  --metadata "/path/to/enriched_metadata.csv" \
  --config "configs/default.json" \
  --output "validation_report.json"
```

## 3. Metadata validation

`src/pg_amcd/metadata.py` matches modern rows by explicit relative path and supports the legacy workbook's stickout/RPM/depth condition format. The index is keyed by relative POSIX path so duplicate basenames in different folders cannot silently collide.

The diagnostics record:

- metadata-row and matched-recording counts;
- unmatched recordings and unmappable rows;
- ambiguous/duplicate rows;
- duplicate basenames and duplicate recording IDs;
- recordings whose tooth count is missing.

With `use_physics_gating=true`, both `validate` and `run` require a finite positive RPM and a positive integer tooth count for every recording. There is no fallback RPM or tooth count. Duplicate/ambiguous paths and duplicate normalized recording IDs are rejected. A chatter/stable label is optional through Stage 4.

### Supplied-workbook audit

The supplied `Vibration - ML/rpm_doc_combinations.xlsx` is useful as a legacy condition index but is not sufficient for a default physics-guided run:

- 114 of 115 MAT recordings condition-map;
- `3p5inch_stickout/c_1030_002.mat` is unmatched;
- the workbook has no tooth-count field, so mapped tooth counts remain `None`;
- repeated conditions are diagnosed rather than overwritten.

Do not invent a tooth count. Use an enriched explicit metadata file, or set `use_physics_gating=false` and accept that Stage 2 is then the legacy MAIW baseline.

## 4. Configuration validation

`src/pg_amcd/config.py` deep-merges an explicit JSON file onto the packaged defaults and validates the resolved result before processing. Among other checks, it enforces:

- `through_stage == 4` for the resolved project scope;
- positive, bounded sampling rate and adequate segment length;
- valid filter order, scaling percentile, and low-pass cutoff below Nyquist;
- at least one positive CEEMDAN high-pass candidate below the resolved low-pass cutoff;
- positive CEEMDAN trial counts and epsilon;
- a chatter band that overlaps `[0, Nyquist]`;
- complete physics-gating coefficients, a selection threshold in `[0, 1]`, and residual exclusion;
- a valid wavelet level/mode and positive threshold scales;
- valid Stage 4 window overlap and feature-energy bands within Nyquist.

When `preprocessing.low_pass_cutoff_hz` is `null`, the implementation resolves it as:

```text
min(4000 Hz, sampling_rate / 2 - 10 Hz)
```

At 1 kHz this is 490 Hz. `configs/test.json` supplies a self-consistent 1 kHz Stage-4 profile whose feature bands stop at 490 Hz. `configs/research_fast.json` explicitly disables physics gating.

## 5. Stage 1 validation

The canonical Stage 1 metric implementation is in `src/pg_amcd/decomposition.py`. `src/pg_amcd/validation.py` remains a standalone compatibility utility; in particular, its legacy frequency-ordering implementation is not the metric used by the production pipeline.

### CEEMDAN residual invariant

PyEMD returns a component matrix. The package treats the final row as the residual only after verifying:

```text
returned final row ~= source - sum(physical IMFs)
```

Failure of this tolerance check raises an error. The stored reconstruction is:

```text
x_hat(t) = sum_k IMF_k(t) + residual(t)
```

### Reconstruction NRMSE

```text
NRMSE = RMS(x - x_hat) / RMS(x)
```

The residual is included explicitly. A finite nonnegative value is required; lower is better.

### Orthogonality

For the intended component set `c_i`:

```text
OI_signed = 2 * sum_(i<j) <c_i, c_j> / sum_i ||c_i||^2
OI_absolute = abs(OI_signed)
OI_pairwise_absolute = 2 * sum_(i<j) abs(<c_i, c_j>) / sum_i ||c_i||^2
```

Stage 1's aggregate orthogonality calculation includes the verified residual. Adjacent-correlation, spectral-overlap, frequency-ordering, and per-IMF descriptors use physical IMFs only.

### Adjacent mode separation

For adjacent physical IMFs, the package records absolute Pearson correlation and the intersection of normalized Welch spectra:

```text
overlap(i, i+1) = sum_f min(P_i(f), P_(i+1)(f))
```

It records mean and maximum values. Lower correlation and overlap indicate better separation, but the package does not fabricate a universal real-dataset threshold.

### Frequency ordering

The production score is the fraction of adjacent IMF spectral centers that are non-increasing:

```text
ordering = mean(center_k >= center_(k+1))
```

This lies in `[0, 1]`; 1 means every adjacent pair follows descending EMD order.

### Controlled cutoff objective

Every candidate high-pass cutoff is applied to the exact same raw segment. Each candidate is decomposed under the configured CEEMDAN seeds. The default lower-is-better objective is:

```text
J = 0.20 * spectral_overlap
  + 0.15 * maximum_adjacent_correlation
  + 0.15 * absolute_orthogonality
  + 0.15 * (1 - frequency_ordering)
  + 0.20 * structural_seed_instability
  + 0.15 * chatter_band_distortion
```

The weights are normalized if overridden. Structural seed instability combines center-frequency variation, energy-distribution variation, optimally matched IMF correlation, spectral-overlap variation, and IMF-count variation. Reconstruction NRMSE variance is intentionally not used as the stability signal because valid CEEMDAN decompositions reconstruct their own input nearly exactly.

## 6. Stage 2 validation

The default method validates RPM/tooth metadata before expensive CEEMDAN work. It records, per physical IMF:

- absolute correlation with the Stage 1 source;
- relative energy and Pearson kurtosis;
- chatter, spindle-harmonic, tooth-harmonic, and combined forced-harmonic energy ratios;
- frequency proximity, spectral center, bandwidth, and entropy;
- an independent gate bounded in `[0, 1]`.

The gate is a sigmoid relevance score. Gates are not normalized to sum to one, and reconstruction uses every continuous gate value. The verified CEEMDAN residual is excluded. Quantitative diagnostics compare RMS, energy, source correlation, chatter retention, spindle/tooth/out-of-band attenuation, spectral distortion, runtime, and multi-seed gate stability.

When physics gating is explicitly disabled, the package records `legacy_maiw_baseline`; its sum-normalized correlation/energy/kurtosis/proximity weights must not be interpreted as the canonical physics gates.

## 7. Stage 3 validation

Stage 3 denoises the single Stage 2 weighted reconstruction. It validates finite one-dimensional input, wavelet support, requested/applied level, coefficient shapes, finite thresholds, and a length-preserving inverse transform.

Recorded evidence includes:

- the unmodified approximation and raw detail coefficients, with thresholded output energy recorded per level;
- one row per wavelet level with ideal dyadic frequency range, chatter-band overlap, input/output energy, threshold, and scale;
- RMS and energy before/after;
- input/output correlation, chatter retention, out-of-band attenuation, spectral distortion, and transient preservation;
- noise sigma and runtime;
- reference RMSE/SNR only for the internal synthetic clean-reference check.

Real-signal SNR improvement is not reported without a known clean reference.

## 8. Stage 4 validation

Stage 4 validates aligned raw/preprocessed/denoised windows, finite defined values, explicit residual handling, optional metadata, and a versioned feature schema. Physical-IMF descriptors and gates exclude the final residual row, while the Stage 1 orthogonality feature deliberately includes the complete component matrix. The implemented schema version is `1.0.0`.

Each definition stores its name, family, description, unit, required source stage, metadata requirement, dimensionless flag, and undefined-value policy. Dynamic band, IMF, and wavelet fields are added to the emitted schema. Undefined canonical values remain JSON `null`/CSV missing, with the reason retained in the JSON feature record; they are not silently replaced with a measured zero.

HEGR is validated against growing and steady synthetic signals. Its exact implementation is documented in [project_details.md](project_details.md).

Run-level aggregate validation writes feature summary, missingness, correlations, schema, distributions, variance, grouped views, and repeatability. Stage 4 performs the feature extraction twice from the identical canonical Stage 1–3 arrays, verifies window/schema/undefined-reason alignment, records each feature's maximum absolute difference in `stage_4_metrics.json` and `feature_repeatability.csv`, and plots those deltas in `aggregate_feature_stability.png`. Stage 4 records explicitly state that feature selection, model training, probability generation, and decisions were not performed.

## 9. Self-check and scorecard evidence

Before processing recordings, `run_scientific_self_checks()` exercises all four stages with deterministic analytic/synthetic inputs. The manifest records unit and synthetic pass/fail evidence for every stage. A failed self-check prevents a completed run.

The artifact-derived scorecard assigns 100 points per stage:

| Category | Points |
|---|---:|
| Algorithmic correctness | 20 |
| Input/output validation | 15 |
| Quantitative metrics | 15 |
| Automated tests | 15 |
| Required artifacts | 15 |
| Visualizations | 10 |
| Reproducibility/provenance | 5 |
| Documentation accuracy | 5 |

Missing/failing automated-test evidence, required output files, or required visualizations caps a stage at 89. Stage 2 is also capped for unvalidated metadata assumptions. Known P0 issues, fabricated metrics, multiple active implementations, or a failing integration test are explicit cap reasons.

The scorecard is run-specific. Do not copy an integration-fixture score into a claim about the real machining dataset.

## 10. Exact verification commands

Portable commands after `python -m pip install -e ".[dev]"`:

```bash
ruff check .
mypy src/pg_amcd
pytest --cov=pg_amcd --cov-report=term-missing --cov-fail-under=90
```

Equivalent commands for the checked-in local environment used during this audit:

```bash
PYTHONPATH=src Python/.venv/bin/ruff check .
PYTHONPATH=src Python/.venv/bin/mypy src/pg_amcd
PYTHONPATH=src Python/.venv/bin/pytest \
  --cov=pg_amcd \
  --cov-report=term-missing \
  --cov-fail-under=90
```

Focused Stage 1–4 regression commands:

```bash
PYTHONPATH=src Python/.venv/bin/pytest -q \
  tests/unit/test_stage1_scientific_core.py \
  tests/unit/test_stage234_diagnostics.py \
  tests/unit/test_metadata.py \
  tests/unit/test_provenance.py \
  tests/unit/test_config.py \
  tests/unit/test_selfcheck.py \
  tests/unit/test_stage_scoring.py \
  tests/unit/test_stage_reporting.py

PYTHONPATH=src Python/.venv/bin/pytest -q \
  tests/integration/test_cli.py \
  tests/integration/test_reporting_cli.py
```

CI runs Ruff, MyPy, and the 90% branch-aware coverage command from `.github/workflows/ci.yml`. This document deliberately does not hard-code a transient pass count or final stage score; the command exit statuses and a completed run's `stage_scorecard.json` are the evidence sources.

## 11. Provenance checks

The SHA-256 run identity covers:

- Git commit and, for a dirty checkout, a digest of tracked changes plus untracked non-ignored file contents;
- resolved configuration;
- input paths and content checksums;
- metadata checksum;
- dependency versions;
- pipeline version;
- feature-schema version.

Reuse requires a matching run ID, a completed manifest with an end timestamp, a nonempty output-checksum map, and successful re-verification that every recorded output exists below the run directory with the same SHA-256. Partial, failed, malformed, checksum-mismatched, or identity-mismatched runs are not reusable. Modification time is a legacy compatibility helper only and is not used by the canonical CLI.

The manifest also records Git dirty state and the worktree digest, timestamps, CLI command, Python/OS/dependencies, selected parameters, warnings/failures, runtimes, input validation, self-checks, stage evidence, and output checksums.

### Schema-version caution

The implementation emits Stage 4 schema version `1.0.0`. Configuration validation requires `feature_schema_version` to match that code-level constant, so run identity and generated feature artifacts cannot advertise a schema the extractor does not implement.
