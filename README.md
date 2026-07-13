# Physics-Guided Adaptive Multi-Stage Chatter Detection (PG-AMCD)

This repository implements the **Physics-Guided Adaptive Multi-Stage Chatter Detection (PG-AMCD)** framework. The pipeline is designed to preprocess raw vibration signals, decompose them into Intrinsic Mode Functions (IMFs) with minimal mode mixing using CEEMDAN, apply physics-aware sigmoidal gating to isolate chatter resonances from tooth-passing harmonics, and perform Bayesian band-aware wavelet denoising to extract clean vibration signals for classifier training.

---

## 📁 Package Layout

All core pipeline steps and mathematical processing blocks are unified under the `pg_amcd` package:

```plaintext
src/pg_amcd/
├── cli.py            # Command Line Interface (validate, run commands)
├── config.py         # Configuration loader and schemas
├── models.py         # Dataclasses and pipeline types
├── io.py             # File loader with strict checks
├── preprocessing.py  # Second-Order Sections (SOS) Butterworth filter and detrending
├── segmentation.py   # High-energy window selection
├── decomposition.py  # Parallel CEEMDAN execution & parameter search
├── weighting.py      # Multi-criteria and physics-gated IMF weighting
├── denoising.py      # Band-aware Bayesian wavelet denoising
├── features.py       # Time-frequency feature extraction
├── detection.py      # Hysteresis temporal decision logic & grouped training
├── evaluation.py     # Cross-validation & ROC AUC metrics
├── validation.py     # Mathematical decomposition validation metrics
└── pipeline.py       # Canonical end-to-end recording processor
```

---

## ⚙️ Setup & Installation

To install the package in development mode along with its dependencies:

```bash
# 1. Install package and developer dependencies
python -m pip install -e ".[dev]"
```

Dependencies are managed in [pyproject.toml](file:///home/harshit/Documents/Research/pyproject.toml) and pinned in `requirements.lock`.

---

## 🚀 Execution & Command-Line Interface

The package exposes a unified CLI entry point `pg-amcd`:

### 1. Dataset Contract Validation
Verify that your dataset files and metadata conform to input contracts without initiating heavy decomposition steps:

```bash
pg-amcd validate \
  --input-dir "Vibration - ML" \
  --metadata "Vibration - ML/rpm_doc_combinations.xlsx" \
  --config "src/pg_amcd/configs/default.json" \
  --output "outputs/validation_report.json"
```

### 2. End-to-End Pipeline Execution
Run the complete decomposition, gating, denoising, and feature extraction pipeline on the entire dataset:

```bash
pg-amcd run \
  --input-dir "Vibration - ML" \
  --metadata "Vibration - ML/rpm_doc_combinations.xlsx" \
  --output-dir "outputs" \
  --config "src/pg_amcd/configs/default.json"
```

---

## 🧪 Testing and Reproduction

### Run Automated Tests
```bash
# Run unit, regression, and integration tests
make test
```

### Full Reproduction Suite
```bash
# Clean up cache, run tests, and regenerate all comparison figures/tables
make reproduce
```
