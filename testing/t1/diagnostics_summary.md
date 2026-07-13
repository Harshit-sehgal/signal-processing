# High-Fidelity Validation Run: Testing Summary (T1)

This report presents EMD diagnostics for three randomly selected data files processed with the fine-tuned parameters (`epsilon=0.02`, `sifting_iterations=16`).

---

## 📊 Summary Comparison Metrics

| File Path | Cutoff Frequency | Number of IMFs | Reconstruction Error (NRMSE) | Overall Orthogonality (OI) | Mean Adjacent IMF Corr (MMI) |
|---|---|---|---|---|---|
| **2inch_stickout/s_320_040.mat** | 150 Hz | 13 | 2.41e-16 | 0.0420 | 0.0855 |
| **2inch_stickout/s_320_030.mat** | 150 Hz | 12 | 2.24e-16 | 0.0060 | 0.0436 |
| **2p5inch_stickout/c_770_005.mat** | 100 Hz | 12 | 2.29e-16 | -0.0203 | 0.0749 |

---

## ⚡ Spectral Analysis (Mean Frequencies of IMF Layers)
Below are the spectral centroid frequencies (in Hz) for the physical IMFs of each test file:

| Layer | s_320_040.mat | s_320_030.mat | c_770_005.mat |
|---| --- | --- | --- |
| **IMF 1** | 2288.1 Hz | 1133.2 Hz | 1288.8 Hz |
| **IMF 2** | 1146.3 Hz | 928.2 Hz | 1142.3 Hz |
| **IMF 3** | 657.3 Hz | 542.4 Hz | 534.8 Hz |
| **IMF 4** | 330.4 Hz | 256.9 Hz | 151.1 Hz |
| **IMF 5** | 127.5 Hz | 122.8 Hz | 114.5 Hz |
| **IMF 6** | 97.6 Hz | 74.1 Hz | 56.1 Hz |
| **IMF 7** | 47.4 Hz | 36.7 Hz | 28.8 Hz |
| **IMF 8** | 25.3 Hz | 19.3 Hz | 15.3 Hz |
| **IMF 9** | 11.4 Hz | 10.7 Hz | 7.6 Hz |
| **IMF 10** | 7.9 Hz | 6.9 Hz | 7.7 Hz |
| **IMF 11** | 4.5 Hz | 1.2 Hz | 2.0 Hz |
| **IMF 12** | 2.7 Hz | N/A | N/A |
| **Residual** | Residual | Residual | Residual |
