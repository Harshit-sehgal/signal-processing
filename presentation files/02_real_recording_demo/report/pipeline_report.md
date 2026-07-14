# PG-AMCD Stage 1–4 Pipeline Report

## 1. Run overview

| Manifest field | Recorded value |
| --- | --- |
| run_id | ea3ee0528b6980082ec6b13e0a5a62873c8ee2454e276a35831e1aa7de1342f4 |
| git_commit | f920df6c8701442a8a3eda5d81478fc4ea636b6b |
| git_dirty | yes |
| git_worktree_sha256 | c6f19698f47f903e82850c93cedb30fa0c4ab1c35a17d28ada24d66266e4dd72 |
| status | completed |
| start_timestamp | 2026-07-14T01:21:16.655084+00:00 |
| end_timestamp | 2026-07-14T01:21:33.614329+00:00 |
| pipeline_version | 4.0.0 |
| feature_schema_version | 1.0.0 |
| success_count | 1 |
| failure_count | 0 |

## 2. Input validation

| Validation metric | Recorded value |
| --- | --- |
| n_files | 1 |
| n_invalid | 0 |
| n_valid | 1 |

## 3. Preprocessing summary

| Recording | Metric | Measured value |
| --- | --- | --- |
| u_570_005 | absolute_orthogonality_index | 0.0271804 |
| u_570_005 | adjacent_imf_correlations | 0.0096659, 0.0124262, 0.0856854, 0.0489386, 0.0756414, 0.0413604, 0.169345, 0.139321 |
| u_570_005 | adjacent_spectral_overlaps | 0.0450154, 0.121494, 0.372402, 0.105747, 0.196209, 0.354925, 0.862475, 0.882505 |
| u_570_005 | ceemdan_runtime_seconds | 0.205277 |
| u_570_005 | controlled_segment_end_index | 24953 |
| u_570_005 | controlled_segment_samples | 10000 |
| u_570_005 | controlled_segment_start_index | 14953 |
| u_570_005 | frequency_ordering_score | 0.875 |
| u_570_005 | imf_metrics.count | 9 |
| u_570_005 | maximum_adjacent_imf_correlation | 0.169345 |
| u_570_005 | maximum_adjacent_spectral_overlap | 0.882505 |
| u_570_005 | mean_adjacent_imf_correlation | 0.072798 |
| u_570_005 | number_of_imfs | 9 |
| u_570_005 | pairwise_absolute_orthogonality_index | 0.067947 |
| u_570_005 | preprocessing_parameters.detrend_before_filter | no |
| u_570_005 | preprocessing_parameters.detrend_type | linear |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_1/u_570_005/cutoff_search.csv | 1 | 23 |
| Stage_1/u_570_005/imf_metrics.csv | 9 | 6 |

- [Stage_1__u_570_005__04_psd_comparison.png](figures/Stage_1__u_570_005__04_psd_comparison.png)
- [Stage_1__u_570_005__06_ceemdan_decomposition.svg](figures/Stage_1__u_570_005__06_ceemdan_decomposition.svg)
- [Stage_1__u_570_005__09_imf_energy_distribution.png](figures/Stage_1__u_570_005__09_imf_energy_distribution.png)
- [Stage_1__u_570_005__02_selected_segment.png](figures/Stage_1__u_570_005__02_selected_segment.png)
- [Stage_1__u_570_005__07_individual_imfs.svg](figures/Stage_1__u_570_005__07_individual_imfs.svg)
- [Stage_1__u_570_005__05_cutoff_search.png](figures/Stage_1__u_570_005__05_cutoff_search.png)
- [Stage_1__u_570_005__05_cutoff_search.svg](figures/Stage_1__u_570_005__05_cutoff_search.svg)
- [Stage_1__u_570_005__15_time_frequency.png](figures/Stage_1__u_570_005__15_time_frequency.png)
- [Stage_1__u_570_005__09_imf_energy_distribution.svg](figures/Stage_1__u_570_005__09_imf_energy_distribution.svg)
- [Stage_1__u_570_005__07_individual_imfs.png](figures/Stage_1__u_570_005__07_individual_imfs.png)
- [Stage_1__u_570_005__03_preprocessing_comparison.png](figures/Stage_1__u_570_005__03_preprocessing_comparison.png)
- [Stage_1__u_570_005__12_adjacent_imf_correlation.svg](figures/Stage_1__u_570_005__12_adjacent_imf_correlation.svg)
- [Stage_1__u_570_005__13_seed_stability.svg](figures/Stage_1__u_570_005__13_seed_stability.svg)
- [Stage_1__u_570_005__10_imf_frequency_ordering.svg](figures/Stage_1__u_570_005__10_imf_frequency_ordering.svg)
- [Stage_1__u_570_005__06_ceemdan_decomposition.png](figures/Stage_1__u_570_005__06_ceemdan_decomposition.png)
- [Stage_1__u_570_005__01_raw_signal.png](figures/Stage_1__u_570_005__01_raw_signal.png)
- [Stage_1__u_570_005__14_reconstruction_error.svg](figures/Stage_1__u_570_005__14_reconstruction_error.svg)
- [Stage_1__u_570_005__03_preprocessing_comparison.svg](figures/Stage_1__u_570_005__03_preprocessing_comparison.svg)
- [Stage_1__u_570_005__13_seed_stability.png](figures/Stage_1__u_570_005__13_seed_stability.png)
- [Stage_1__u_570_005__14_reconstruction_error.png](figures/Stage_1__u_570_005__14_reconstruction_error.png)
- [Stage_1__u_570_005__02_selected_segment.svg](figures/Stage_1__u_570_005__02_selected_segment.svg)
- [Stage_1__u_570_005__08_residual.png](figures/Stage_1__u_570_005__08_residual.png)
- [Stage_1__u_570_005__11_imf_bandwidth.png](figures/Stage_1__u_570_005__11_imf_bandwidth.png)
- [Stage_1__u_570_005__12_adjacent_imf_correlation.png](figures/Stage_1__u_570_005__12_adjacent_imf_correlation.png)
- [Stage_1__u_570_005__04_psd_comparison.svg](figures/Stage_1__u_570_005__04_psd_comparison.svg)
- [Stage_1__u_570_005__15_time_frequency.svg](figures/Stage_1__u_570_005__15_time_frequency.svg)
- [Stage_1__u_570_005__01_raw_signal.svg](figures/Stage_1__u_570_005__01_raw_signal.svg)
- [Stage_1__u_570_005__10_imf_frequency_ordering.png](figures/Stage_1__u_570_005__10_imf_frequency_ordering.png)
- [Stage_1__u_570_005__11_imf_bandwidth.svg](figures/Stage_1__u_570_005__11_imf_bandwidth.svg)
- [Stage_1__u_570_005__08_residual.svg](figures/Stage_1__u_570_005__08_residual.svg)

## 4. Stage 1 decomposition

| Recording | Metric | Measured value |
| --- | --- | --- |
| u_570_005 | absolute_orthogonality_index | 0.0271804 |
| u_570_005 | adjacent_imf_correlations | 0.0096659, 0.0124262, 0.0856854, 0.0489386, 0.0756414, 0.0413604, 0.169345, 0.139321 |
| u_570_005 | adjacent_spectral_overlaps | 0.0450154, 0.121494, 0.372402, 0.105747, 0.196209, 0.354925, 0.862475, 0.882505 |
| u_570_005 | ceemdan_runtime_seconds | 0.205277 |
| u_570_005 | controlled_segment_end_index | 24953 |
| u_570_005 | controlled_segment_samples | 10000 |
| u_570_005 | controlled_segment_start_index | 14953 |
| u_570_005 | frequency_ordering_score | 0.875 |
| u_570_005 | imf_metrics.count | 9 |
| u_570_005 | maximum_adjacent_imf_correlation | 0.169345 |
| u_570_005 | maximum_adjacent_spectral_overlap | 0.882505 |
| u_570_005 | mean_adjacent_imf_correlation | 0.072798 |
| u_570_005 | number_of_imfs | 9 |
| u_570_005 | pairwise_absolute_orthogonality_index | 0.067947 |
| u_570_005 | preprocessing_parameters.detrend_before_filter | no |
| u_570_005 | preprocessing_parameters.detrend_type | linear |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_1/u_570_005/cutoff_search.csv | 1 | 23 |
| Stage_1/u_570_005/imf_metrics.csv | 9 | 6 |

- [Stage_1__u_570_005__04_psd_comparison.png](figures/Stage_1__u_570_005__04_psd_comparison.png)
- [Stage_1__u_570_005__06_ceemdan_decomposition.svg](figures/Stage_1__u_570_005__06_ceemdan_decomposition.svg)
- [Stage_1__u_570_005__09_imf_energy_distribution.png](figures/Stage_1__u_570_005__09_imf_energy_distribution.png)
- [Stage_1__u_570_005__02_selected_segment.png](figures/Stage_1__u_570_005__02_selected_segment.png)
- [Stage_1__u_570_005__07_individual_imfs.svg](figures/Stage_1__u_570_005__07_individual_imfs.svg)
- [Stage_1__u_570_005__05_cutoff_search.png](figures/Stage_1__u_570_005__05_cutoff_search.png)
- [Stage_1__u_570_005__05_cutoff_search.svg](figures/Stage_1__u_570_005__05_cutoff_search.svg)
- [Stage_1__u_570_005__15_time_frequency.png](figures/Stage_1__u_570_005__15_time_frequency.png)
- [Stage_1__u_570_005__09_imf_energy_distribution.svg](figures/Stage_1__u_570_005__09_imf_energy_distribution.svg)
- [Stage_1__u_570_005__07_individual_imfs.png](figures/Stage_1__u_570_005__07_individual_imfs.png)
- [Stage_1__u_570_005__03_preprocessing_comparison.png](figures/Stage_1__u_570_005__03_preprocessing_comparison.png)
- [Stage_1__u_570_005__12_adjacent_imf_correlation.svg](figures/Stage_1__u_570_005__12_adjacent_imf_correlation.svg)
- [Stage_1__u_570_005__13_seed_stability.svg](figures/Stage_1__u_570_005__13_seed_stability.svg)
- [Stage_1__u_570_005__10_imf_frequency_ordering.svg](figures/Stage_1__u_570_005__10_imf_frequency_ordering.svg)
- [Stage_1__u_570_005__06_ceemdan_decomposition.png](figures/Stage_1__u_570_005__06_ceemdan_decomposition.png)
- [Stage_1__u_570_005__01_raw_signal.png](figures/Stage_1__u_570_005__01_raw_signal.png)
- [Stage_1__u_570_005__14_reconstruction_error.svg](figures/Stage_1__u_570_005__14_reconstruction_error.svg)
- [Stage_1__u_570_005__03_preprocessing_comparison.svg](figures/Stage_1__u_570_005__03_preprocessing_comparison.svg)
- [Stage_1__u_570_005__13_seed_stability.png](figures/Stage_1__u_570_005__13_seed_stability.png)
- [Stage_1__u_570_005__14_reconstruction_error.png](figures/Stage_1__u_570_005__14_reconstruction_error.png)
- [Stage_1__u_570_005__02_selected_segment.svg](figures/Stage_1__u_570_005__02_selected_segment.svg)
- [Stage_1__u_570_005__08_residual.png](figures/Stage_1__u_570_005__08_residual.png)
- [Stage_1__u_570_005__11_imf_bandwidth.png](figures/Stage_1__u_570_005__11_imf_bandwidth.png)
- [Stage_1__u_570_005__12_adjacent_imf_correlation.png](figures/Stage_1__u_570_005__12_adjacent_imf_correlation.png)
- [Stage_1__u_570_005__04_psd_comparison.svg](figures/Stage_1__u_570_005__04_psd_comparison.svg)
- [Stage_1__u_570_005__15_time_frequency.svg](figures/Stage_1__u_570_005__15_time_frequency.svg)
- [Stage_1__u_570_005__01_raw_signal.svg](figures/Stage_1__u_570_005__01_raw_signal.svg)
- [Stage_1__u_570_005__10_imf_frequency_ordering.png](figures/Stage_1__u_570_005__10_imf_frequency_ordering.png)
- [Stage_1__u_570_005__11_imf_bandwidth.svg](figures/Stage_1__u_570_005__11_imf_bandwidth.svg)
- [Stage_1__u_570_005__08_residual.svg](figures/Stage_1__u_570_005__08_residual.svg)

## 5. Stage 2 IMF gating

| Recording | Metric | Measured value |
| --- | --- | --- |
| u_570_005 | chatter_band_retention | 0.0659777 |
| u_570_005 | correlation_with_source | 0.943953 |
| u_570_005 | energy_after | 119.42 |
| u_570_005 | energy_before | 1250.75 |
| u_570_005 | gate_normalisation | legacy_sum_normalised_baseline |
| u_570_005 | gate_sum | 1 |
| u_570_005 | gate_vector_stability.available | no |
| u_570_005 | gate_vector_stability.n_seeds | 3 |
| u_570_005 | gate_vector_stability.physical_imf_counts | 9, 9, 10 |
| u_570_005 | gate_vector_stability.reason | Gate vectors cannot be compared by row index because CEEMDAN returned different physical-IMF counts across seeds: [9, 9, 10]. |
| u_570_005 | gate_vector_stability.seeds | 42, 43, 44 |
| u_570_005 | gate_vector_stability.selection_threshold | 0.5 |
| u_570_005 | method | legacy_maiw_baseline |
| u_570_005 | out_of_band_attenuation | 0.901699 |
| u_570_005 | physical_imf_count | 9 |
| u_570_005 | reconstruction_runtime_seconds | 0.0142864 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_2/u_570_005/imf_gates.csv | 9 | 3 |
| Stage_2/u_570_005/imf_indicators.csv | 9 | 18 |

- [Stage_2__u_570_005__11_harmonic_markers.svg](figures/Stage_2__u_570_005__11_harmonic_markers.svg)
- [Stage_2__u_570_005__08_weighted_psd_comparison.svg](figures/Stage_2__u_570_005__08_weighted_psd_comparison.svg)
- [Stage_2__u_570_005__01_imf_gate_values.svg](figures/Stage_2__u_570_005__01_imf_gate_values.svg)
- [Stage_2__u_570_005__08_weighted_psd_comparison.png](figures/Stage_2__u_570_005__08_weighted_psd_comparison.png)
- [Stage_2__u_570_005__09_retained_suppressed_imfs.svg](figures/Stage_2__u_570_005__09_retained_suppressed_imfs.svg)
- [Stage_2__u_570_005__03_frequency_vs_gate.svg](figures/Stage_2__u_570_005__03_frequency_vs_gate.svg)
- [Stage_2__u_570_005__05_chatter_energy_vs_gate.png](figures/Stage_2__u_570_005__05_chatter_energy_vs_gate.png)
- [Stage_2__u_570_005__03_frequency_vs_gate.png](figures/Stage_2__u_570_005__03_frequency_vs_gate.png)
- [Stage_2__u_570_005__02_imf_indicator_comparison.svg](figures/Stage_2__u_570_005__02_imf_indicator_comparison.svg)
- [Stage_2__u_570_005__12_chatter_band_psd.png](figures/Stage_2__u_570_005__12_chatter_band_psd.png)
- [Stage_2__u_570_005__06_forced_harmonics_vs_gate.svg](figures/Stage_2__u_570_005__06_forced_harmonics_vs_gate.svg)
- [Stage_2__u_570_005__02_imf_indicator_comparison.png](figures/Stage_2__u_570_005__02_imf_indicator_comparison.png)
- [Stage_2__u_570_005__10_gate_stability.png](figures/Stage_2__u_570_005__10_gate_stability.png)
- [Stage_2__u_570_005__06_forced_harmonics_vs_gate.png](figures/Stage_2__u_570_005__06_forced_harmonics_vs_gate.png)
- [Stage_2__u_570_005__04_energy_vs_gate.png](figures/Stage_2__u_570_005__04_energy_vs_gate.png)
- [Stage_2__u_570_005__01_imf_gate_values.png](figures/Stage_2__u_570_005__01_imf_gate_values.png)
- [Stage_2__u_570_005__09_retained_suppressed_imfs.png](figures/Stage_2__u_570_005__09_retained_suppressed_imfs.png)
- [Stage_2__u_570_005__07_weighted_reconstruction.svg](figures/Stage_2__u_570_005__07_weighted_reconstruction.svg)
- [Stage_2__u_570_005__11_harmonic_markers.png](figures/Stage_2__u_570_005__11_harmonic_markers.png)
- [Stage_2__u_570_005__12_chatter_band_psd.svg](figures/Stage_2__u_570_005__12_chatter_band_psd.svg)
- [Stage_2__u_570_005__04_energy_vs_gate.svg](figures/Stage_2__u_570_005__04_energy_vs_gate.svg)
- [Stage_2__u_570_005__10_gate_stability.svg](figures/Stage_2__u_570_005__10_gate_stability.svg)
- [Stage_2__u_570_005__05_chatter_energy_vs_gate.svg](figures/Stage_2__u_570_005__05_chatter_energy_vs_gate.svg)
- [Stage_2__u_570_005__07_weighted_reconstruction.png](figures/Stage_2__u_570_005__07_weighted_reconstruction.png)

## 6. Stage 3 wavelet denoising

| Recording | Metric | Measured value |
| --- | --- | --- |
| u_570_005 | chatter_band_retention | 0.220895 |
| u_570_005 | correlation_after | 0.891228 |
| u_570_005 | correlation_before | 1 |
| u_570_005 | correlation_before_after | 0.891228 |
| u_570_005 | denoising_scope | reconstruction_level |
| u_570_005 | energy_after | 63.1152 |
| u_570_005 | energy_before | 119.42 |
| u_570_005 | estimated_noise_sigma | 0.0680289 |
| u_570_005 | input_output_correlation | 0.891228 |
| u_570_005 | input_stage | Stage_2 weighted reconstruction |
| u_570_005 | out_of_band_attenuation | 0.445904 |
| u_570_005 | requested_level | 4 |
| u_570_005 | resolved_level | 4 |
| u_570_005 | rms_after | 0.0794451 |
| u_570_005 | rms_before | 0.10928 |
| u_570_005 | runtime_seconds | 0.00063043 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_3/u_570_005/wavelet_thresholds.csv | 5 | 17 |

- [Stage_3__u_570_005__02_all_signal_stages.png](figures/Stage_3__u_570_005__02_all_signal_stages.png)
- [Stage_3__u_570_005__03_psd_before_after.png](figures/Stage_3__u_570_005__03_psd_before_after.png)
- [Stage_3__u_570_005__11_residual_noise_psd.png](figures/Stage_3__u_570_005__11_residual_noise_psd.png)
- [Stage_3__u_570_005__02_all_signal_stages.svg](figures/Stage_3__u_570_005__02_all_signal_stages.svg)
- [Stage_3__u_570_005__06_wavelet_subbands.svg](figures/Stage_3__u_570_005__06_wavelet_subbands.svg)
- [Stage_3__u_570_005__04_wavelet_level_energies.svg](figures/Stage_3__u_570_005__04_wavelet_level_energies.svg)
- [Stage_3__u_570_005__05_wavelet_thresholds.svg](figures/Stage_3__u_570_005__05_wavelet_thresholds.svg)
- [Stage_3__u_570_005__09_synthetic_recovery.svg](figures/Stage_3__u_570_005__09_synthetic_recovery.svg)
- [Stage_3__u_570_005__06_wavelet_subbands.png](figures/Stage_3__u_570_005__06_wavelet_subbands.png)
- [Stage_3__u_570_005__11_residual_noise_psd.svg](figures/Stage_3__u_570_005__11_residual_noise_psd.svg)
- [Stage_3__u_570_005__05_wavelet_thresholds.png](figures/Stage_3__u_570_005__05_wavelet_thresholds.png)
- [Stage_3__u_570_005__12_time_frequency_energy.svg](figures/Stage_3__u_570_005__12_time_frequency_energy.svg)
- [Stage_3__u_570_005__08_spectrogram_comparison.svg](figures/Stage_3__u_570_005__08_spectrogram_comparison.svg)
- [Stage_3__u_570_005__10_residual_noise.png](figures/Stage_3__u_570_005__10_residual_noise.png)
- [Stage_3__u_570_005__04_wavelet_level_energies.png](figures/Stage_3__u_570_005__04_wavelet_level_energies.png)
- [Stage_3__u_570_005__01_weighted_vs_denoised.svg](figures/Stage_3__u_570_005__01_weighted_vs_denoised.svg)
- [Stage_3__u_570_005__10_residual_noise.svg](figures/Stage_3__u_570_005__10_residual_noise.svg)
- [Stage_3__u_570_005__08_spectrogram_comparison.png](figures/Stage_3__u_570_005__08_spectrogram_comparison.png)
- [Stage_3__u_570_005__09_synthetic_recovery.png](figures/Stage_3__u_570_005__09_synthetic_recovery.png)
- [Stage_3__u_570_005__07_chatter_band_overlap.svg](figures/Stage_3__u_570_005__07_chatter_band_overlap.svg)
- [Stage_3__u_570_005__03_psd_before_after.svg](figures/Stage_3__u_570_005__03_psd_before_after.svg)
- [Stage_3__u_570_005__01_weighted_vs_denoised.png](figures/Stage_3__u_570_005__01_weighted_vs_denoised.png)
- [Stage_3__u_570_005__12_time_frequency_energy.png](figures/Stage_3__u_570_005__12_time_frequency_energy.png)
- [Stage_3__u_570_005__07_chatter_band_overlap.png](figures/Stage_3__u_570_005__07_chatter_band_overlap.png)

## 7. Stage 4 feature extraction

| Recording | Metric | Measured value |
| --- | --- | --- |
| u_570_005 | all_defined_values_finite | yes |
| u_570_005 | decisions_generated | no |
| u_570_005 | defined_feature_values | 120 |
| u_570_005 | defined_fraction | 0.916031 |
| u_570_005 | feature_schema_version | 1.0.0 |
| u_570_005 | feature_selection_performed | no |
| u_570_005 | model_training_performed | no |
| u_570_005 | physics_metadata_valid_for_all_windows | no |
| u_570_005 | probabilities_generated | no |
| u_570_005 | repeat_extraction_stability.absolute_tolerance | 1e-12 |
| u_570_005 | repeat_extraction_stability.all_values_within_tolerance | yes |
| u_570_005 | repeat_extraction_stability.deterministic | yes |
| u_570_005 | repeat_extraction_stability.exact_value_match | yes |
| u_570_005 | repeat_extraction_stability.feature_comparison_count.early_chatter_band_energy_growth | 1 |
| u_570_005 | repeat_extraction_stability.feature_comparison_count.early_energy_growth_rate | 1 |
| u_570_005 | repeat_extraction_stability.feature_comparison_count.early_hegr | 1 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_4/aggregate/all_recording_features.csv | 1 | 139 |
| Stage_4/aggregate/feature_correlations.csv | 122 | 123 |
| Stage_4/aggregate/feature_missingness.csv | 139 | 4 |
| Stage_4/aggregate/feature_repeatability.csv | 131 | 5 |
| Stage_4/aggregate/feature_summary.csv | 122 | 9 |
| Stage_4/u_570_005/window_features.csv | 1 | 138 |

- [Stage_4__u_570_005__01_rms_timeline.png](figures/Stage_4__u_570_005__01_rms_timeline.png)
- [Stage_4__u_570_005__04_chatter_energy_timeline.png](figures/Stage_4__u_570_005__04_chatter_energy_timeline.png)
- [Stage_4__u_570_005__07_instantaneous_energy_timeline.png](figures/Stage_4__u_570_005__07_instantaneous_energy_timeline.png)
- [Stage_4__u_570_005__10_feature_family_summary.png](figures/Stage_4__u_570_005__10_feature_family_summary.png)
- [Stage_4__u_570_005__06_hegr_timeline.svg](figures/Stage_4__u_570_005__06_hegr_timeline.svg)
- [Stage_4__u_570_005__03_spectral_entropy_timeline.svg](figures/Stage_4__u_570_005__03_spectral_entropy_timeline.svg)
- [Stage_4__u_570_005__08_imf_gate_values.svg](figures/Stage_4__u_570_005__08_imf_gate_values.svg)
- [Stage_4__u_570_005__10_feature_family_summary.svg](figures/Stage_4__u_570_005__10_feature_family_summary.svg)
- [Stage_4__u_570_005__08_imf_gate_values.png](figures/Stage_4__u_570_005__08_imf_gate_values.png)
- [Stage_4__u_570_005__02_kurtosis_timeline.svg](figures/Stage_4__u_570_005__02_kurtosis_timeline.svg)
- [Stage_4__u_570_005__07_instantaneous_energy_timeline.svg](figures/Stage_4__u_570_005__07_instantaneous_energy_timeline.svg)
- [Stage_4__u_570_005__05_harmonic_energy_timeline.png](figures/Stage_4__u_570_005__05_harmonic_energy_timeline.png)
- [Stage_4__u_570_005__03_spectral_entropy_timeline.png](figures/Stage_4__u_570_005__03_spectral_entropy_timeline.png)
- [Stage_4__u_570_005__04_chatter_energy_timeline.svg](figures/Stage_4__u_570_005__04_chatter_energy_timeline.svg)
- [Stage_4__u_570_005__02_kurtosis_timeline.png](figures/Stage_4__u_570_005__02_kurtosis_timeline.png)
- [Stage_4__u_570_005__09_wavelet_energy_ratios.svg](figures/Stage_4__u_570_005__09_wavelet_energy_ratios.svg)
- [Stage_4__u_570_005__09_wavelet_energy_ratios.png](figures/Stage_4__u_570_005__09_wavelet_energy_ratios.png)
- [Stage_4__u_570_005__06_hegr_timeline.png](figures/Stage_4__u_570_005__06_hegr_timeline.png)
- [Stage_4__u_570_005__05_harmonic_energy_timeline.svg](figures/Stage_4__u_570_005__05_harmonic_energy_timeline.svg)
- [Stage_4__u_570_005__01_rms_timeline.svg](figures/Stage_4__u_570_005__01_rms_timeline.svg)
- [Stage_4__aggregate__aggregate_feature_variance.svg](figures/Stage_4__aggregate__aggregate_feature_variance.svg)
- [Stage_4__aggregate__aggregate_feature_variance.png](figures/Stage_4__aggregate__aggregate_feature_variance.png)
- [Stage_4__aggregate__aggregate_feature_correlation_heatmap.svg](figures/Stage_4__aggregate__aggregate_feature_correlation_heatmap.svg)
- [Stage_4__aggregate__aggregate_feature_stability.png](figures/Stage_4__aggregate__aggregate_feature_stability.png)
- [Stage_4__aggregate__aggregate_feature_missingness.png](figures/Stage_4__aggregate__aggregate_feature_missingness.png)
- [Stage_4__aggregate__aggregate_feature_correlation_heatmap.png](figures/Stage_4__aggregate__aggregate_feature_correlation_heatmap.png)
- [Stage_4__aggregate__aggregate_feature_values_by_recording.svg](figures/Stage_4__aggregate__aggregate_feature_values_by_recording.svg)
- [Stage_4__aggregate__aggregate_feature_distributions.svg](figures/Stage_4__aggregate__aggregate_feature_distributions.svg)
- [Stage_4__aggregate__aggregate_feature_values_by_stickout.png](figures/Stage_4__aggregate__aggregate_feature_values_by_stickout.png)
- [Stage_4__aggregate__aggregate_feature_distributions.png](figures/Stage_4__aggregate__aggregate_feature_distributions.png)
- [Stage_4__aggregate__aggregate_feature_values_by_rpm.png](figures/Stage_4__aggregate__aggregate_feature_values_by_rpm.png)
- [Stage_4__aggregate__aggregate_feature_values_by_depth_of_cut.svg](figures/Stage_4__aggregate__aggregate_feature_values_by_depth_of_cut.svg)
- [Stage_4__aggregate__aggregate_feature_values_by_depth_of_cut.png](figures/Stage_4__aggregate__aggregate_feature_values_by_depth_of_cut.png)
- [Stage_4__aggregate__aggregate_feature_stability.svg](figures/Stage_4__aggregate__aggregate_feature_stability.svg)
- [Stage_4__aggregate__aggregate_feature_values_by_stickout.svg](figures/Stage_4__aggregate__aggregate_feature_values_by_stickout.svg)
- [Stage_4__aggregate__aggregate_feature_missingness.svg](figures/Stage_4__aggregate__aggregate_feature_missingness.svg)
- [Stage_4__aggregate__aggregate_feature_values_by_recording.png](figures/Stage_4__aggregate__aggregate_feature_values_by_recording.png)
- [Stage_4__aggregate__aggregate_feature_values_by_rpm.svg](figures/Stage_4__aggregate__aggregate_feature_values_by_rpm.svg)

## 8. Stage scorecard

| Stage | Score | Raw score | Passed | Failed | Cap reasons |
| --- | --- | --- | --- | --- | --- |
| Stage_1 | 100 | 100 | 56 | 0 |  |
| Stage_2 | 99.38 | 99.38 | 56 | 1 |  |
| Stage_3 | 100 | 100 | 49 | 0 |  |
| Stage_4 | 100 | 100 | 51 | 0 |  |

- [stage_scorecard.png](figures/stage_scorecard.png)
- [stage_progress.png](figures/stage_progress.png)

## 9. Warnings and failures

- Warning: Physics-guided gating is disabled; Stage 2 uses the configured MAIW baseline and metadata-dependent features may be undefined.
- Stage_2 failed checks: metric_selected_imf_consistency

## 10. Limitations

- The production workflow intentionally ends after Stage 4 feature extraction.
- Real-signal SNR improvement is not reported without a known clean reference.
