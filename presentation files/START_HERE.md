# PG-AMCD presentation bundle — start here

This folder is a self-contained evidence bundle for presenting the project through Stage 4.

## Fastest way to present

Open [`START_HERE.html`](START_HERE.html) in a web browser. It links to the complete reports and displays the main figures in a recommended presentation order.

## Recommended presentation flow

1. **Problem and framework** — show [`04_project_documentation/conceptual_framework.jpeg`](04_project_documentation/conceptual_framework.jpeg) and explain the Stage 1–4 signal-processing path.
2. **Acceptance status** — show [`05_quick_showcase/01_acceptance_stage_scorecard.png`](05_quick_showcase/01_acceptance_stage_scorecard.png). The deterministic acceptance run scored 100/100 in every requested stage.
3. **Stage 1: CEEMDAN decomposition** — show [`05_quick_showcase/03_stage1_ceemdan_decomposition.png`](05_quick_showcase/03_stage1_ceemdan_decomposition.png).
4. **Stage 2: IMF gates and reconstruction** — show [`05_quick_showcase/04_stage2_imf_gate_values.png`](05_quick_showcase/04_stage2_imf_gate_values.png).
5. **Stage 3: synthetic recovery validation** — show [`05_quick_showcase/05_stage3_synthetic_recovery.png`](05_quick_showcase/05_stage3_synthetic_recovery.png).
6. **Stage 4: HEGR and feature repeatability** — show [`05_quick_showcase/06_stage4_hegr_timeline.png`](05_quick_showcase/06_stage4_hegr_timeline.png) and [`05_quick_showcase/07_stage4_feature_repeatability.png`](05_quick_showcase/07_stage4_feature_repeatability.png).
7. **Real-recording demonstration** — show [`05_quick_showcase/08_real_recording_scorecard.png`](05_quick_showcase/08_real_recording_scorecard.png), then open the complete real-record report.
8. **Limitations and next work** — explain the real-dataset metadata gap recorded in [`03_real_dataset_validation/VALIDATION_SUMMARY.md`](03_real_dataset_validation/VALIDATION_SUMMARY.md).

## Complete evidence

- [`01_100_score_acceptance/report/pipeline_report.html`](01_100_score_acceptance/report/pipeline_report.html) — full acceptance report with every linked figure.
- [`01_100_score_acceptance/stage_scorecard.json`](01_100_score_acceptance/stage_scorecard.json) — machine-readable acceptance scores.
- [`01_100_score_acceptance/run_manifest.json`](01_100_score_acceptance/run_manifest.json) — checksummed acceptance-run manifest.
- [`02_real_recording_demo/report/pipeline_report.html`](02_real_recording_demo/report/pipeline_report.html) — full one-record real-data report.
- [`02_real_recording_demo/stage_scorecard.json`](02_real_recording_demo/stage_scorecard.json) — machine-readable real-record scores.
- [`03_real_dataset_validation/real_dataset_validation.json`](03_real_dataset_validation/real_dataset_validation.json) — validation results for the supplied real dataset.
- [`04_project_documentation/`](04_project_documentation/) — project README, validation guide, detailed design, original requirements, and conceptual framework.
- [`05_quick_showcase/`](05_quick_showcase/) — curated figures for a short presentation.

## How to describe the results accurately

- The **100/100 scores** are from the deterministic synthetic/physics acceptance fixture and prove that the implemented Stage 1–4 contracts pass reproducibly.
- The **real-recording demonstration** scores **100, 99.38, 100, and 100** for Stages 1–4. The 0.62-point Stage 2 deduction records inconsistent seed IMF counts, not a hidden failure.
- The broader real dataset contains 115 files: 114 have valid signal data and were matched; 8 mappings are metadata-ambiguous; 1 file is unmatched; and tooth-count metadata is absent for all 114 mapped records. Strict physics processing therefore stops honestly instead of inventing missing inputs.
- No real-data SNR improvement is claimed without a known clean reference.
- Stages 5–7 are outside this completed Stage 4 scope.

The run manifests retain the original temporary execution paths for provenance. The reports themselves use copied relative assets and remain browsable from this folder.
