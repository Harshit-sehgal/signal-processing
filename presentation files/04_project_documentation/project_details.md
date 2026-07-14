# PG-AMCD Stage 1–4 technical design

This document specifies the live canonical implementation in `src/pg_amcd`. It covers the validated production path through feature extraction only. It does not describe a planned classifier as though one were implemented.

```text
validated MAT signal + resolved machining metadata
                      |
                      v
Stage 1  controlled preprocessing and PyEMD CEEMDAN
                      |
                      v
Stage 2  independent physics-guided IMF gates
                      |
                      v
Stage 3  reconstruction-level BayesShrink denoising
                      |
                      v
Stage 4  versioned sliding-window feature extraction
```

The supplied framework image is a useful conceptual roadmap, but it differs from this implementation in important ways: the code uses CEEMDAN rather than ICEEMDAN; physical amplitude is preserved alongside one traceable scale factor rather than repeatedly normalized; normalized MAIW is an optional legacy baseline rather than the default method; wavelet denoising is applied once to the complete weighted reconstruction rather than separately to each IMF; HEGR has the exact scalar definition below; no unsupported "stability margin" is emitted; and the illustrated Stage 5–7 blocks are not active.

## 1. Input, segment control, and preprocessing

The canonical loader reads `tsDS` from each MAT file. Column 0 is time and the configured signal column is the vibration channel. It rejects invalid dimensionality, complex or non-finite samples, non-increasing or excessively jittered time, inconsistent sampling rate, inadequate duration, and constant or numerically flat signals.

Stage 1 first applies the first candidate filter to the complete recording and uses that physical-amplitude result only to locate the maximum-energy segment. The configured `segment_points` controls its length; the default is 10,000 samples, which is one second at the default 10 kHz sampling rate. Every cutoff candidate is then evaluated on the same raw samples. After selection, the full recording is processed with the selected filter and the same segment indices are sliced. This prevents different candidates from winning because they were evaluated on different time intervals.

Preprocessing uses a zero-phase Butterworth band-pass implemented as second-order sections with `scipy.signal.sosfiltfilt`. The default order is 3. Candidate high-pass edges are configured in `ceemdan.search_cutoffs`. When no low-pass edge is supplied, it resolves to

```text
min(4000 Hz, sampling_rate / 2 - 10 Hz).
```

Filtering and linear detrending produce the authoritative physical-amplitude signal. A separate numerical signal is

```text
x_scaled(t) = x_physical(t) / percentile(|x_physical|, 99.5).
```

The divisor is stored as that recording's Stage 1 scale factor. Stage 2 and Stage 3 restore physical units with this same factor; the physical signal is never mislabeled as normalized data.

## 2. Stage 1: controlled CEEMDAN decomposition

### 2.1 Canonical algorithm and residual contract

The only production decomposition is `PyEMD.CEEMDAN` from the `EMD-signal` distribution. It is CEEMDAN, not ICEEMDAN. The default full-run settings are 300 ensemble trials, epsilon `0.02`, seed `42`, and 16 fixed sifting iterations; cutoff search may deliberately use cheaper configured trial and seed counts.

For a scaled source `x`, PyEMD returns a component matrix. The implementation does not assume silently that the last row is an IMF. It separates

```text
physical IMFs = components[:-1]
residual       = components[-1]
```

only after verifying at runtime that

```text
residual ~= x - sum(physical IMFs).
```

Failure of that check aborts the run. The complete decomposition reconstruction is

```text
x_hat(t) = sum_k IMF_k(t) + residual(t),
NRMSE    = RMS(x - x_hat) / RMS(x).
```

The residual participates in reconstruction and global orthogonality evidence. It is excluded from physical-IMF correlation, spectral-overlap, frequency-ordering, gating, and reconstruction in Stage 2.

### 2.2 Structural diagnostics

For component rows `c_i`, global orthogonality is recorded as

```text
OI_signed = 2 * sum_(i<j) <c_i,c_j> / sum_i ||c_i||^2
OI_abs    = abs(OI_signed)
OI_pair   = 2 * sum_(i<j) abs(<c_i,c_j>) / sum_i ||c_i||^2.
```

For adjacent physical IMFs, the implementation records absolute centered Pearson correlation. It also normalizes each Welch spectrum to unit mass and computes

```text
overlap(k,k+1) = sum_f min(P_k(f), P_(k+1)(f)).
```

The spectral center of an IMF is the PSD-weighted mean frequency. Frequency ordering is the fraction of adjacent centers that are non-increasing:

```text
ordering = mean(center_k >= center_(k+1)).
```

Seed stability compares decompositions structurally. It combines center-frequency instability, energy-distribution change, optimally matched IMF correlation loss, spectral-overlap variation, and IMF-count variation. It does not substitute reconstruction-error variance, because valid complete decompositions almost exactly reconstruct their own inputs.

### 2.3 Cutoff selection objective

Each configured high-pass candidate is scored on the identical controlled raw segment. The default lower-is-better objective is

```text
J = 0.20 * mean_adjacent_spectral_overlap
  + 0.15 * maximum_adjacent_correlation
  + 0.15 * absolute_orthogonality
  + 0.15 * (1 - frequency_ordering)
  + 0.20 * structural_seed_instability
  + 0.15 * chatter_band_distortion.
```

Override weights are normalized to sum to one. Chatter-band distortion compares raw and physically filtered band energy symmetrically:

```text
D = 1 - exp(-abs(log(E_filtered / E_raw))),
```

with explicit handling for an effectively zero raw-band energy. The selected cutoff minimizes `J`; the implementation does not select on a one-metric "MMI" shortcut.

## 3. Stage 2: physics-guided independent IMF gates

The default configuration sets `use_physics_gating=true`. Before CEEMDAN is started, it requires a finite positive RPM and a positive integer tooth count. No fallback machining values are invented. The two forced-vibration fundamentals are

```text
f_spindle = RPM / 60
f_tooth   = RPM * tooth_count / 60.
```

For each physical IMF `i`, Welch PSD diagnostics include relative energy, spectral center, bandwidth, normalized spectral entropy, chatter-band energy ratio `R_chatter`, spindle and tooth harmonic ratios, their union `R_forced`, and Gaussian center-frequency proximity. The harmonic masks cover configured multiples of `f_spindle` and `f_tooth` within the configured tolerance. Source correlation is absolute Pearson correlation. With configured chatter center `f_c` and spread `b`, proximity is

```text
P_i = exp(-0.5 * ((IMF_center_i - f_c) / b)^2).
```

Pearson kurtosis is converted to a bounded excess-kurtosis indicator:

```text
K_i = clip((PearsonKurtosis_i - 3) / kurtosis_scale, 0, 1).
```

The default physics score and gate are

```text
s_i = 4.0 * R_chatter
    + 2.0 * |corr(IMF_i, source)|
    + 1.0 * K_i
    + 1.0 * P_i
    - 5.0 * R_forced
    - 1.5

g_i = sigmoid(s_i).
```

All coefficients come from the resolved configuration. Relative energy, bandwidth, and entropy are retained as diagnostics but do not enter this gate equation. Every `g_i` is independently bounded in `[0,1]`; the vector is deliberately not normalized to sum to one. The selection threshold is reporting and stability evidence only. Reconstruction always uses every continuous gate:

```text
x_weighted(t) = sum_i g_i * IMF_i(t).
```

The verified CEEMDAN residual is never gated as a physical IMF and is not added to this reconstruction.

### Optional legacy baseline

Only `use_physics_gating=false` activates `legacy_maiw_baseline`. It separately sum-normalizes absolute source correlation, relative energy, raw Pearson kurtosis, and Gaussian dominant-frequency proximity; combines them with configured `alpha`, `beta`, `gamma`, and `delta`; and normalizes the result again to sum to one. This compatibility baseline does not use RPM/tooth physics and must not be reported as the canonical physics-guided result.

## 4. Stage 3: reconstruction-level Bayesian wavelet denoising

Stage 3 receives the single scaled `x_weighted` reconstruction. It does not denoise individual IMFs and does not recompute a time-varying threshold. The default transform is `db8`, requested level 4, with the applied level capped to the maximum supported by the signal length.

For the finest detail coefficients `cD1`, noise is estimated with centered median absolute deviation:

```text
sigma_n = max(minimum_noise_sigma,
              median(|cD1 - median(cD1)|) / 0.6745).
```

The production configuration uses `minimum_noise_sigma = 1e-12`. For every detail band `d_j`:

```text
sigma_y^2 = mean(d_j^2)
sigma_x^2 = max(0, sigma_y^2 - sigma_n^2)

T_base,j = sigma_n^2 / sqrt(sigma_x^2),  if sigma_x^2 is resolvable
           max(|d_j|),                    otherwise.
```

The ideal dyadic support for `cD_j` is `[fs*2^(-(j+1)), fs*2^(-j)]`. Let `q_j` be the fraction of that interval overlapped by the configured chatter band. With band-aware processing,

```text
scale_j = chatter_threshold_scale * q_j
        + noise_threshold_scale * (1 - q_j)
T_j     = T_base,j * scale_j.
```

Default scales are `0.5` in the chatter band and `1.4` outside it. Soft thresholding is the default:

```text
d'_j[n] = sign(d_j[n]) * max(|d_j[n]| - T_j, 0).
```

The approximation coefficients remain unchanged. The inverse transform is trimmed or edge-padded only as needed to preserve the exact input length. The denoised scaled signal is restored to physical units with the Stage 1 scale factor. Each coefficient band, threshold, overlap fraction, and input/output energy is retained as evidence.

## 5. Stage 4: versioned feature extraction

Stage 4 uses aligned raw, preprocessed physical, denoised physical, IMF, and gate evidence plus the resolved wavelet name and level. It computes a fresh DWT for each denoised window rather than consuming Stage 3 coefficient arrays. The default window is 1 second with 75% overlap; if it exceeds the controlled segment, it is capped to the segment duration and the run records a warning.

Schema version `1.0.0` covers these feature families:

- time-domain statistics;
- Welch frequency and configured band-energy features;
- physical-IMF composition, separation, and gate features;
- wavelet and STFT time-frequency features;
- early-chatter/Hilbert-energy features;
- RPM/tooth physics and harmonic/sideband features.

Dynamic fields follow documented patterns for configured bands, IMF indices, and wavelet coefficients. Each schema definition records its family, formula description, unit, required source stage, metadata requirement, dimensionality, and undefined-value policy. If a denominator or prerequisite is invalid, the canonical value remains JSON `null`/CSV missing and an explicit reason is stored. A compatibility finite-scalar dictionary helper can fill undefined values for legacy callers, but artifact writers preserve nulls and reasons.

### Exact HEGR definition

For the denoised physical window `x[n]` sampled at `f_s`:

```text
z[n]   = hilbert(x[n])
E[n]   = |z[n]|^2
L      = min(N, max(3, round(0.010 * f_s)))
E_s[n] = convolve(E, ones(L)/L, mode="same")
dE[n]  = gradient(E_s, 1/f_s).
```

Let `edge = floor(L/2)`. If removing both edges leaves at least one sample, HEGR uses `dE[edge:-edge]`; otherwise it uses the full derivative. The emitted scalar is

```text
HEGR = mean(max(dE_valid, 0)).
```

Its unit is signal-unit squared per second. The full derivative is available in the in-memory `FeatureExtractionResult.traces`; canonical `window_features.json` currently serializes with `include_traces=false`, so the scalar and its definition—not the derivative array—are persisted. This is a positive-growth statistic, not a classifier score and not an unqualified plot of `dE/dt`.

Stage 4 also writes run-level summary, missingness, correlation, distribution, variance, grouped-view, schema, and stability artifacts. It repeats feature extraction from the identical canonical Stage 1–3 arrays, verifies window/schema/undefined-reason alignment, stores per-feature maximum absolute deltas in `stage_4_metrics.json` and `feature_repeatability.csv`, and plots those deltas in `aggregate_feature_stability.png`. This establishes deterministic Stage 4 repeatability without rerunning the stochastic upstream CEEMDAN stage. Stage 4 explicitly records that feature selection, model training, probabilities, and decisions were not performed.

## 6. Run identity, artifacts, and evidence

The canonical CLI is `pg-amcd run|validate|report`. A production `run` must end at Stage 4. Values above Stage 4 are out of scope, and the complete artifact workflow does not accept a partial Stage 1–3 closeout.

The run ID is a SHA-256 over the Git commit, a content digest of tracked changes and untracked non-ignored files for dirty checkouts, resolved configuration, path-aware input checksums, metadata checksum, dependency versions, pipeline version, and feature-schema version. A prior directory is reusable only when its manifest matches the same identity, records a completed run with an end timestamp, has a nonempty output-checksum map, and every recorded output exists below that directory with the same SHA-256; file modification time is not used as provenance.

Every completed run writes:

```text
outputs/<run_id>/
├── run_manifest.json
├── stage_scorecard.json
├── stage_scorecard.png
├── stage_progress.png
├── Stage_1/<recording_id>/
├── Stage_2/<recording_id>/
├── Stage_3/<recording_id>/
├── Stage_4/<recording_id>/
├── Stage_4/aggregate/
└── report/
    ├── pipeline_report.md
    ├── pipeline_report.html
    └── figures/
```

The scorecard is derived from artifacts rather than caller-supplied scores. Its categories are algorithmic correctness, validation, quantitative metrics, automated tests, required artifacts, visualizations, provenance, and documentation. The report is likewise reconstructed from the selected run's manifest and machine-readable artifacts. See [VALIDATION.md](VALIDATION.md) for formulas, evidence requirements, score caps, and exact verification commands.

## 7. Scope boundaries and current data limitation

- The supplied `Vibration - ML/rpm_doc_combinations.xlsx` condition-maps 114 of 115 MAT files; `3p5inch_stickout/c_1030_002.mat` remains unmatched.
- The workbook has no tooth-count column. Therefore all condition-mapped tooth counts remain missing and the workbook alone cannot satisfy default physics-guided Stage 2.
- Repeated workbook conditions are diagnosed as ambiguous rather than silently overwritten. A production physics run needs enriched, preferably path-specific metadata with explicit tooth counts.
- Real-signal SNR improvement is not claimed without a known clean reference. Reference SNR/RMSE is restricted to deterministic synthetic validation.
- The extractor's implemented schema constant is `1.0.0`; configuration validation rejects a different `feature_schema_version`, keeping run identity and generated feature artifacts honest.
- Stage 5 feature selection, Stage 6 classifier training, and Stage 7 probability/decision logic are not activated in the CLI, run tree, scorecard, or report. Compatibility or experimental modules elsewhere in the repository do not change that production boundary.

No fixed real-data stage score is asserted in this design document. Scores and measured metrics are properties of a specific completed `outputs/<run_id>/` run and must be read from that run's scorecard and report.
