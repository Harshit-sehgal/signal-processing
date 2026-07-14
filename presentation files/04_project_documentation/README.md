# PG-AMCD: machining-signal processing through Stage 4

PG-AMCD is a reproducible signal-processing workflow for machining-vibration recordings. The canonical production path validates MAT inputs, preprocesses one controlled signal segment, performs CEEMDAN decomposition, applies physics-guided IMF gates, denoises the weighted reconstruction, and extracts a versioned feature table.

The active workflow ends at Stage 4:

```text
MAT input + machining metadata
          |
          v
validation and preprocessing
          |
          v
Stage 1: PyEMD CEEMDAN decomposition
          |
          v
Stage 2: independent physics-guided IMF gates
          |
          v
Stage 3: reconstruction-level BayesShrink wavelet denoising
          |
          v
Stage 4: sliding-window feature extraction
```

Stage 5 feature selection, Stage 6 classifiers, and Stage 7 probability or decision output are not active in the primary CLI, run artifacts, scorecard, or report. Experimental compatibility modules may remain in the repository, but they are not part of the Stage 1–4 production contract.

## Important distinction from the supplied framework image

The supplied PG-AMCD image is a conceptual research roadmap, not an exact diagram of the live implementation:

- The package uses PyEMD `CEEMDAN`, not ICEEMDAN.
- Preprocessing preserves a physical-amplitude signal and stores one separate robust scale factor; it does not repeatedly normalize each stage as the image's generic normalization block might suggest.
- The default Stage 2 method uses independent sigmoid gates that are not normalized to sum to one. Sum-normalized MAIW is an explicit non-physics baseline only.
- Stage 3 denoises the complete Stage 2 reconstruction once. It is not IMF-specific and does not use a time-varying threshold.
- Implemented HEGR is a scalar positive Hilbert-energy growth statistic defined in [project_details.md](project_details.md), not simply an unqualified `dE/dt` trace.
- The pictured stability-margin feature is not emitted because the available inputs do not support a justified stability-margin formula.
- The illustrated Stage 5–7 blocks are future concepts and are not activated.

## What each active stage does

| Step | Canonical behavior | Primary source |
|---|---|---|
| Input validation | Loads `tsDS`, validates time and signal columns, sampling rate, jitter, duration, finite values, and nonconstant signal | `src/pg_amcd/io.py` |
| Preprocessing | SOS Butterworth band-pass filtering, detrending, physical-amplitude preservation, and one stored robust scale factor | `src/pg_amcd/preprocessing.py` |
| Stage 1 | Selects one controlled index range from the first-candidate preprocessed signal, evaluates every high-pass cutoff on those same raw samples, runs seeded PyEMD CEEMDAN, and verifies/splits the final residual row | `src/pg_amcd/optimization.py`, `src/pg_amcd/decomposition.py` |
| Stage 2 | Requires positive RPM and tooth count in physics mode; computes per-IMF indicators and independent sigmoid gates; excludes the residual | `src/pg_amcd/weighting.py` |
| Stage 3 | Applies per-detail BayesShrink thresholds to the Stage 2 weighted reconstruction and preserves complete diagnostics | `src/pg_amcd/denoising.py` |
| Stage 4 | Extracts overlapping time, frequency, IMF, wavelet, early-chatter, and physics-guided features with explicit null reasons | `src/pg_amcd/features.py` |
| Artifacts | Writes the exact Stage 1–4 tree, machine outputs, summaries, metrics, PNG figures, and optional SVG copies | `src/pg_amcd/stage_artifacts.py` |
| Evidence | Generates an artifact-derived scorecard and ten-section Markdown/HTML report | `src/pg_amcd/stage_scoring.py`, `src/pg_amcd/stage_reporting.py` |

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

The `EMD-signal` distribution supplies the `PyEMD.CEEMDAN` implementation. The packaged default configuration is `pg_amcd/configs/default.json`; the source checkout also provides the equivalent user-facing file at `configs/default.json`.

## Metadata contract

The default configuration sets `use_physics_gating=true`. In this mode every MAT recording must map to metadata containing:

- a unique relative input path or otherwise unambiguous condition mapping;
- a unique `recording_id` after filesystem-safe normalization;
- a finite positive `rpm`;
- a positive integer `tooth_count`.

Useful optional fields are `stickout`, `depth_of_cut`, `feed_rate`, `tool_identifier`, and `label`. Labels are not required through Stage 4 because the production workflow does not train or evaluate a classifier.

An explicit CSV can use this shape:

```csv
relative_path,recording_id,rpm,tooth_count,stickout,depth_of_cut,feed_rate,label
2p5inch_stickout/u_570_005.mat,rod25_u_570_005,570,4,2.5,0.005,0.002,stable
```

### Limitation of the supplied workbook

`Vibration - ML/rpm_doc_combinations.xlsx` is a legacy condition table rather than a complete physics-metadata index. A live audit maps 114 of the 115 MAT files by stickout/RPM/depth condition; `3p5inch_stickout/c_1030_002.mat` is not mapped. The workbook contains no tooth-count column, so all mapped tooth counts remain `None`. It also contains repeated conditions that are reported as ambiguous rather than silently overwritten.

Consequently, the supplied workbook alone cannot satisfy the default physics-guided run. Provide an enriched metadata file with explicit relative paths and tooth counts, or deliberately use a configuration with `use_physics_gating=false`. The latter selects the legacy MAIW baseline and is not equivalent to the physics-guided method.

## Command-line interface

The installed entry point is `pg-amcd`. `python -m pg_amcd.cli` is equivalent.

### Validate before processing

Physics-guided validation:

```bash
pg-amcd validate \
  --input-dir "Vibration - ML" \
  --metadata "/path/to/enriched_metadata.csv" \
  --config "configs/default.json" \
  --output "validation_report.json"
```

Signal-only/non-physics validation:

```bash
pg-amcd validate \
  --input-dir "Vibration - ML" \
  --config "configs/research_fast.json" \
  --output "validation_report.json"
```

Validation is read-only with respect to scientific outputs and exits nonzero when the input or required metadata contract fails.

### Run the complete active workflow

```bash
pg-amcd run \
  --input-dir "Vibration - ML" \
  --metadata "/path/to/enriched_metadata.csv" \
  --output-dir "outputs" \
  --config "configs/default.json" \
  --through-stage 4
```

The production artifact workflow requires `--through-stage 4`; 4 is the default. Values above 4 are rejected as out of scope, and a run request ending before Stage 4 is rejected because it cannot satisfy the complete artifact and scorecard contract.

`--continue-on-error` processes later recordings after a recording failure, but the run cannot be completed successfully: it is `partial_failure` when at least one recording succeeded and `failed` otherwise, and the command returns nonzero.

### Regenerate an existing report

```bash
pg-amcd report --run-dir "outputs/<run_id>"
```

The report command reads only the selected run's manifest and existing artifacts, including its JSON, CSV, NPZ, Markdown, and figure evidence. It does not accept caller-supplied metrics or compare a second run.

## Run identity and output contract

`run_id` is the full SHA-256 of the Git commit, a content digest of tracked changes and untracked non-ignored files when the checkout is dirty, resolved configuration, path-aware input checksums, metadata checksum, dependency versions, pipeline version, and feature-schema version. A pre-existing directory is reused only when its manifest has the same run ID, `status="completed"`, an end timestamp, a nonempty output-checksum map, and every recorded output still exists below that run directory with the same SHA-256. File modification time is not a reuse criterion.

Each run has this structure:

```text
outputs/<run_id>/
├── run_manifest.json
├── stage_scorecard.json
├── stage_scorecard.png
├── stage_progress.png
├── Stage_1/
│   └── <recording_id>/
├── Stage_2/
│   └── <recording_id>/
├── Stage_3/
│   └── <recording_id>/
├── Stage_4/
│   ├── <recording_id>/
│   └── aggregate/
└── report/
    ├── pipeline_report.md
    ├── pipeline_report.html
    └── figures/
```

Per-recording stage folders contain the required NPZ/CSV/JSON/Markdown files and semantic PNG figures, with SVG copies when enabled. Stage 4 also writes aggregate features, summary, missingness, correlations, schema, and grouped visualizations.

`run_manifest.json` contains the Git state, timestamps, CLI command, Python/OS/dependencies, resolved configuration, input and metadata checksums, selected parameters, warnings/failures, per-stage and per-recording runtimes, scientific self-checks, stage evidence, and output checksums. There is no separate canonical `provenance.json`.

## Scorecard and report

The scorecard is derived from the run tree rather than from hard-coded or caller-supplied values. Each stage is evaluated out of 100 using:

- algorithmic correctness: 20;
- input/output validation: 15;
- quantitative metrics: 15;
- automated tests: 15;
- required artifacts: 15;
- visualizations: 10;
- reproducibility/provenance: 5;
- documentation accuracy: 5.

A stage is capped at 89 when required test evidence, artifacts, or visualizations are missing/failing; Stage 2 is also capped for unvalidated metadata assumptions. The generated report contains run overview, input validation, preprocessing, Stages 1–4, scorecard, warnings/failures, and limitations. Scores are properties of a particular completed run; this repository does not claim a fabricated final real-dataset score.

## Verification

After installing developer dependencies, run exactly:

```bash
ruff check .
mypy src/pg_amcd
pytest --cov=pg_amcd --cov-report=term-missing --cov-fail-under=90
```

The same commands are used by `.github/workflows/ci.yml` and `make quality`. [VALIDATION.md](VALIDATION.md) defines the evidence contract and the repository-local command variants without freezing transient pass counts or scores.

## Canonical package map

```text
src/pg_amcd/
├── cli/                    # run, validate, report routing
├── config.py               # packaged defaults, deep merge, strict validation
├── io.py                   # MAT/tsDS input contract
├── metadata.py             # explicit and legacy metadata matching
├── preprocessing.py        # physical/scaled SOS preprocessing
├── segmentation.py         # controlled max-energy segment and windows
├── optimization.py         # cutoff objective and seed stability
├── decomposition.py        # PyEMD CEEMDAN and Stage 1 metrics
├── weighting.py            # Stage 2 gates and explicit legacy baseline
├── denoising.py            # Stage 3 BayesShrink diagnostics
├── features.py             # Stage 4 features and schema 1.0.0
├── pipeline.py             # canonical in-memory Stage 1–4 composition
├── stage_artifacts.py      # required files and figures
├── provenance.py           # SHA-256 identity and completed-run matching
├── selfcheck.py            # deterministic Stage 1–4 scientific checks
├── stage_scoring.py        # traceable per-stage rubric
└── stage_reporting.py      # manifest/artifact-derived Markdown and HTML
```

See [project_details.md](project_details.md) for the implemented mathematics and [VALIDATION.md](VALIDATION.md) for validation evidence and limitations.
