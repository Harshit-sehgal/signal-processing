# PG-AMCD Stage 1–4 Pipeline Report

## 1. Run overview

| Manifest field | Recorded value |
| --- | --- |
| run_id | af81a2c706b53d0411f52a53188ffc3bbcdad15030b3e31011683df11b2a8d93 |
| git_commit | f920df6c8701442a8a3eda5d81478fc4ea636b6b |
| git_dirty | yes |
| git_worktree_sha256 | e5eae1855e4e1bf5f125d44e94c075e02ef32dba0ac7fafcfb3fadb9a0441390 |
| status | completed |
| start_timestamp | 2026-07-14T01:31:45.832654+00:00 |
| end_timestamp | 2026-07-14T01:32:00.302596+00:00 |
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
| acceptance_sample | absolute_orthogonality_index | 0.245175 |
| acceptance_sample | adjacent_imf_correlations | 0.147004, 0.183302 |
| acceptance_sample | adjacent_spectral_overlaps | 0.0377752, 0.310413 |
| acceptance_sample | ceemdan_runtime_seconds | 0.016781 |
| acceptance_sample | controlled_segment_end_index | 605 |
| acceptance_sample | controlled_segment_samples | 512 |
| acceptance_sample | controlled_segment_start_index | 93 |
| acceptance_sample | frequency_ordering_score | 1 |
| acceptance_sample | imf_metrics.count | 3 |
| acceptance_sample | maximum_adjacent_imf_correlation | 0.183302 |
| acceptance_sample | maximum_adjacent_spectral_overlap | 0.310413 |
| acceptance_sample | mean_adjacent_imf_correlation | 0.165153 |
| acceptance_sample | number_of_imfs | 3 |
| acceptance_sample | pairwise_absolute_orthogonality_index | 0.245175 |
| acceptance_sample | preprocessing_parameters.detrend_before_filter | no |
| acceptance_sample | preprocessing_parameters.detrend_type | linear |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_1/acceptance_sample/cutoff_search.csv | 1 | 23 |
| Stage_1/acceptance_sample/imf_metrics.csv | 3 | 6 |

- [Stage_1__acceptance_sample__04_psd_comparison.png](figures/Stage_1__acceptance_sample__04_psd_comparison.png)
- [Stage_1__acceptance_sample__06_ceemdan_decomposition.svg](figures/Stage_1__acceptance_sample__06_ceemdan_decomposition.svg)
- [Stage_1__acceptance_sample__09_imf_energy_distribution.png](figures/Stage_1__acceptance_sample__09_imf_energy_distribution.png)
- [Stage_1__acceptance_sample__02_selected_segment.png](figures/Stage_1__acceptance_sample__02_selected_segment.png)
- [Stage_1__acceptance_sample__07_individual_imfs.svg](figures/Stage_1__acceptance_sample__07_individual_imfs.svg)
- [Stage_1__acceptance_sample__05_cutoff_search.png](figures/Stage_1__acceptance_sample__05_cutoff_search.png)
- [Stage_1__acceptance_sample__05_cutoff_search.svg](figures/Stage_1__acceptance_sample__05_cutoff_search.svg)
- [Stage_1__acceptance_sample__15_time_frequency.png](figures/Stage_1__acceptance_sample__15_time_frequency.png)
- [Stage_1__acceptance_sample__09_imf_energy_distribution.svg](figures/Stage_1__acceptance_sample__09_imf_energy_distribution.svg)
- [Stage_1__acceptance_sample__07_individual_imfs.png](figures/Stage_1__acceptance_sample__07_individual_imfs.png)
- [Stage_1__acceptance_sample__03_preprocessing_comparison.png](figures/Stage_1__acceptance_sample__03_preprocessing_comparison.png)
- [Stage_1__acceptance_sample__12_adjacent_imf_correlation.svg](figures/Stage_1__acceptance_sample__12_adjacent_imf_correlation.svg)
- [Stage_1__acceptance_sample__13_seed_stability.svg](figures/Stage_1__acceptance_sample__13_seed_stability.svg)
- [Stage_1__acceptance_sample__10_imf_frequency_ordering.svg](figures/Stage_1__acceptance_sample__10_imf_frequency_ordering.svg)
- [Stage_1__acceptance_sample__06_ceemdan_decomposition.png](figures/Stage_1__acceptance_sample__06_ceemdan_decomposition.png)
- [Stage_1__acceptance_sample__01_raw_signal.png](figures/Stage_1__acceptance_sample__01_raw_signal.png)
- [Stage_1__acceptance_sample__14_reconstruction_error.svg](figures/Stage_1__acceptance_sample__14_reconstruction_error.svg)
- [Stage_1__acceptance_sample__03_preprocessing_comparison.svg](figures/Stage_1__acceptance_sample__03_preprocessing_comparison.svg)
- [Stage_1__acceptance_sample__13_seed_stability.png](figures/Stage_1__acceptance_sample__13_seed_stability.png)
- [Stage_1__acceptance_sample__14_reconstruction_error.png](figures/Stage_1__acceptance_sample__14_reconstruction_error.png)
- [Stage_1__acceptance_sample__02_selected_segment.svg](figures/Stage_1__acceptance_sample__02_selected_segment.svg)
- [Stage_1__acceptance_sample__08_residual.png](figures/Stage_1__acceptance_sample__08_residual.png)
- [Stage_1__acceptance_sample__11_imf_bandwidth.png](figures/Stage_1__acceptance_sample__11_imf_bandwidth.png)
- [Stage_1__acceptance_sample__12_adjacent_imf_correlation.png](figures/Stage_1__acceptance_sample__12_adjacent_imf_correlation.png)
- [Stage_1__acceptance_sample__04_psd_comparison.svg](figures/Stage_1__acceptance_sample__04_psd_comparison.svg)
- [Stage_1__acceptance_sample__15_time_frequency.svg](figures/Stage_1__acceptance_sample__15_time_frequency.svg)
- [Stage_1__acceptance_sample__01_raw_signal.svg](figures/Stage_1__acceptance_sample__01_raw_signal.svg)
- [Stage_1__acceptance_sample__10_imf_frequency_ordering.png](figures/Stage_1__acceptance_sample__10_imf_frequency_ordering.png)
- [Stage_1__acceptance_sample__11_imf_bandwidth.svg](figures/Stage_1__acceptance_sample__11_imf_bandwidth.svg)
- [Stage_1__acceptance_sample__08_residual.svg](figures/Stage_1__acceptance_sample__08_residual.svg)

## 4. Stage 1 decomposition

| Recording | Metric | Measured value |
| --- | --- | --- |
| acceptance_sample | absolute_orthogonality_index | 0.245175 |
| acceptance_sample | adjacent_imf_correlations | 0.147004, 0.183302 |
| acceptance_sample | adjacent_spectral_overlaps | 0.0377752, 0.310413 |
| acceptance_sample | ceemdan_runtime_seconds | 0.016781 |
| acceptance_sample | controlled_segment_end_index | 605 |
| acceptance_sample | controlled_segment_samples | 512 |
| acceptance_sample | controlled_segment_start_index | 93 |
| acceptance_sample | frequency_ordering_score | 1 |
| acceptance_sample | imf_metrics.count | 3 |
| acceptance_sample | maximum_adjacent_imf_correlation | 0.183302 |
| acceptance_sample | maximum_adjacent_spectral_overlap | 0.310413 |
| acceptance_sample | mean_adjacent_imf_correlation | 0.165153 |
| acceptance_sample | number_of_imfs | 3 |
| acceptance_sample | pairwise_absolute_orthogonality_index | 0.245175 |
| acceptance_sample | preprocessing_parameters.detrend_before_filter | no |
| acceptance_sample | preprocessing_parameters.detrend_type | linear |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_1/acceptance_sample/cutoff_search.csv | 1 | 23 |
| Stage_1/acceptance_sample/imf_metrics.csv | 3 | 6 |

- [Stage_1__acceptance_sample__04_psd_comparison.png](figures/Stage_1__acceptance_sample__04_psd_comparison.png)
- [Stage_1__acceptance_sample__06_ceemdan_decomposition.svg](figures/Stage_1__acceptance_sample__06_ceemdan_decomposition.svg)
- [Stage_1__acceptance_sample__09_imf_energy_distribution.png](figures/Stage_1__acceptance_sample__09_imf_energy_distribution.png)
- [Stage_1__acceptance_sample__02_selected_segment.png](figures/Stage_1__acceptance_sample__02_selected_segment.png)
- [Stage_1__acceptance_sample__07_individual_imfs.svg](figures/Stage_1__acceptance_sample__07_individual_imfs.svg)
- [Stage_1__acceptance_sample__05_cutoff_search.png](figures/Stage_1__acceptance_sample__05_cutoff_search.png)
- [Stage_1__acceptance_sample__05_cutoff_search.svg](figures/Stage_1__acceptance_sample__05_cutoff_search.svg)
- [Stage_1__acceptance_sample__15_time_frequency.png](figures/Stage_1__acceptance_sample__15_time_frequency.png)
- [Stage_1__acceptance_sample__09_imf_energy_distribution.svg](figures/Stage_1__acceptance_sample__09_imf_energy_distribution.svg)
- [Stage_1__acceptance_sample__07_individual_imfs.png](figures/Stage_1__acceptance_sample__07_individual_imfs.png)
- [Stage_1__acceptance_sample__03_preprocessing_comparison.png](figures/Stage_1__acceptance_sample__03_preprocessing_comparison.png)
- [Stage_1__acceptance_sample__12_adjacent_imf_correlation.svg](figures/Stage_1__acceptance_sample__12_adjacent_imf_correlation.svg)
- [Stage_1__acceptance_sample__13_seed_stability.svg](figures/Stage_1__acceptance_sample__13_seed_stability.svg)
- [Stage_1__acceptance_sample__10_imf_frequency_ordering.svg](figures/Stage_1__acceptance_sample__10_imf_frequency_ordering.svg)
- [Stage_1__acceptance_sample__06_ceemdan_decomposition.png](figures/Stage_1__acceptance_sample__06_ceemdan_decomposition.png)
- [Stage_1__acceptance_sample__01_raw_signal.png](figures/Stage_1__acceptance_sample__01_raw_signal.png)
- [Stage_1__acceptance_sample__14_reconstruction_error.svg](figures/Stage_1__acceptance_sample__14_reconstruction_error.svg)
- [Stage_1__acceptance_sample__03_preprocessing_comparison.svg](figures/Stage_1__acceptance_sample__03_preprocessing_comparison.svg)
- [Stage_1__acceptance_sample__13_seed_stability.png](figures/Stage_1__acceptance_sample__13_seed_stability.png)
- [Stage_1__acceptance_sample__14_reconstruction_error.png](figures/Stage_1__acceptance_sample__14_reconstruction_error.png)
- [Stage_1__acceptance_sample__02_selected_segment.svg](figures/Stage_1__acceptance_sample__02_selected_segment.svg)
- [Stage_1__acceptance_sample__08_residual.png](figures/Stage_1__acceptance_sample__08_residual.png)
- [Stage_1__acceptance_sample__11_imf_bandwidth.png](figures/Stage_1__acceptance_sample__11_imf_bandwidth.png)
- [Stage_1__acceptance_sample__12_adjacent_imf_correlation.png](figures/Stage_1__acceptance_sample__12_adjacent_imf_correlation.png)
- [Stage_1__acceptance_sample__04_psd_comparison.svg](figures/Stage_1__acceptance_sample__04_psd_comparison.svg)
- [Stage_1__acceptance_sample__15_time_frequency.svg](figures/Stage_1__acceptance_sample__15_time_frequency.svg)
- [Stage_1__acceptance_sample__01_raw_signal.svg](figures/Stage_1__acceptance_sample__01_raw_signal.svg)
- [Stage_1__acceptance_sample__10_imf_frequency_ordering.png](figures/Stage_1__acceptance_sample__10_imf_frequency_ordering.png)
- [Stage_1__acceptance_sample__11_imf_bandwidth.svg](figures/Stage_1__acceptance_sample__11_imf_bandwidth.svg)
- [Stage_1__acceptance_sample__08_residual.svg](figures/Stage_1__acceptance_sample__08_residual.svg)

## 5. Stage 2 IMF gating

| Recording | Metric | Measured value |
| --- | --- | --- |
| acceptance_sample | chatter_band_retention | 0.964126 |
| acceptance_sample | correlation_with_source | 0.518267 |
| acceptance_sample | energy_after | 41.4061 |
| acceptance_sample | energy_before | 101.045 |
| acceptance_sample | gate_normalisation | independent_not_sum_normalised |
| acceptance_sample | gate_sum | 0.990459 |
| acceptance_sample | gate_vector_stability.available | yes |
| acceptance_sample | gate_vector_stability.max_gate_std | 6.34839e-05 |
| acceptance_sample | gate_vector_stability.mean_gate_by_imf | 0.981512, 0.00575972, 0.00219576 |
| acceptance_sample | gate_vector_stability.mean_gate_std | 4.29611e-05 |
| acceptance_sample | gate_vector_stability.mean_gates | 0.981512, 0.00575972, 0.00219576 |
| acceptance_sample | gate_vector_stability.n_seeds | 2 |
| acceptance_sample | gate_vector_stability.physical_imf_count | 3 |
| acceptance_sample | gate_vector_stability.physical_imf_counts | 3, 3 |
| acceptance_sample | gate_vector_stability.seeds | 42, 43 |
| acceptance_sample | gate_vector_stability.selected_count_by_seed | 1, 1 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_2/acceptance_sample/imf_gates.csv | 3 | 3 |
| Stage_2/acceptance_sample/imf_indicators.csv | 3 | 18 |

- [Stage_2__acceptance_sample__11_harmonic_markers.svg](figures/Stage_2__acceptance_sample__11_harmonic_markers.svg)
- [Stage_2__acceptance_sample__08_weighted_psd_comparison.svg](figures/Stage_2__acceptance_sample__08_weighted_psd_comparison.svg)
- [Stage_2__acceptance_sample__01_imf_gate_values.svg](figures/Stage_2__acceptance_sample__01_imf_gate_values.svg)
- [Stage_2__acceptance_sample__08_weighted_psd_comparison.png](figures/Stage_2__acceptance_sample__08_weighted_psd_comparison.png)
- [Stage_2__acceptance_sample__09_retained_suppressed_imfs.svg](figures/Stage_2__acceptance_sample__09_retained_suppressed_imfs.svg)
- [Stage_2__acceptance_sample__03_frequency_vs_gate.svg](figures/Stage_2__acceptance_sample__03_frequency_vs_gate.svg)
- [Stage_2__acceptance_sample__05_chatter_energy_vs_gate.png](figures/Stage_2__acceptance_sample__05_chatter_energy_vs_gate.png)
- [Stage_2__acceptance_sample__03_frequency_vs_gate.png](figures/Stage_2__acceptance_sample__03_frequency_vs_gate.png)
- [Stage_2__acceptance_sample__02_imf_indicator_comparison.svg](figures/Stage_2__acceptance_sample__02_imf_indicator_comparison.svg)
- [Stage_2__acceptance_sample__12_chatter_band_psd.png](figures/Stage_2__acceptance_sample__12_chatter_band_psd.png)
- [Stage_2__acceptance_sample__06_forced_harmonics_vs_gate.svg](figures/Stage_2__acceptance_sample__06_forced_harmonics_vs_gate.svg)
- [Stage_2__acceptance_sample__02_imf_indicator_comparison.png](figures/Stage_2__acceptance_sample__02_imf_indicator_comparison.png)
- [Stage_2__acceptance_sample__10_gate_stability.png](figures/Stage_2__acceptance_sample__10_gate_stability.png)
- [Stage_2__acceptance_sample__06_forced_harmonics_vs_gate.png](figures/Stage_2__acceptance_sample__06_forced_harmonics_vs_gate.png)
- [Stage_2__acceptance_sample__04_energy_vs_gate.png](figures/Stage_2__acceptance_sample__04_energy_vs_gate.png)
- [Stage_2__acceptance_sample__01_imf_gate_values.png](figures/Stage_2__acceptance_sample__01_imf_gate_values.png)
- [Stage_2__acceptance_sample__09_retained_suppressed_imfs.png](figures/Stage_2__acceptance_sample__09_retained_suppressed_imfs.png)
- [Stage_2__acceptance_sample__07_weighted_reconstruction.svg](figures/Stage_2__acceptance_sample__07_weighted_reconstruction.svg)
- [Stage_2__acceptance_sample__11_harmonic_markers.png](figures/Stage_2__acceptance_sample__11_harmonic_markers.png)
- [Stage_2__acceptance_sample__12_chatter_band_psd.svg](figures/Stage_2__acceptance_sample__12_chatter_band_psd.svg)
- [Stage_2__acceptance_sample__04_energy_vs_gate.svg](figures/Stage_2__acceptance_sample__04_energy_vs_gate.svg)
- [Stage_2__acceptance_sample__10_gate_stability.svg](figures/Stage_2__acceptance_sample__10_gate_stability.svg)
- [Stage_2__acceptance_sample__05_chatter_energy_vs_gate.svg](figures/Stage_2__acceptance_sample__05_chatter_energy_vs_gate.svg)
- [Stage_2__acceptance_sample__07_weighted_reconstruction.png](figures/Stage_2__acceptance_sample__07_weighted_reconstruction.png)

## 6. Stage 3 wavelet denoising

| Recording | Metric | Measured value |
| --- | --- | --- |
| acceptance_sample | chatter_band_retention | 0.00063004 |
| acceptance_sample | correlation_after | 0.232056 |
| acceptance_sample | correlation_before | 1 |
| acceptance_sample | correlation_before_after | 0.232056 |
| acceptance_sample | denoising_scope | reconstruction_level |
| acceptance_sample | energy_after | 2.16123 |
| acceptance_sample | energy_before | 41.4061 |
| acceptance_sample | estimated_noise_sigma | 0.476436 |
| acceptance_sample | input_output_correlation | 0.232056 |
| acceptance_sample | input_stage | Stage_2 weighted reconstruction |
| acceptance_sample | out_of_band_attenuation | 0.431672 |
| acceptance_sample | requested_level | 2 |
| acceptance_sample | resolved_level | 2 |
| acceptance_sample | rms_after | 0.0649704 |
| acceptance_sample | rms_before | 0.284379 |
| acceptance_sample | runtime_seconds | 0.000223672 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_3/acceptance_sample/wavelet_thresholds.csv | 3 | 17 |

- [Stage_3__acceptance_sample__02_all_signal_stages.png](figures/Stage_3__acceptance_sample__02_all_signal_stages.png)
- [Stage_3__acceptance_sample__03_psd_before_after.png](figures/Stage_3__acceptance_sample__03_psd_before_after.png)
- [Stage_3__acceptance_sample__11_residual_noise_psd.png](figures/Stage_3__acceptance_sample__11_residual_noise_psd.png)
- [Stage_3__acceptance_sample__02_all_signal_stages.svg](figures/Stage_3__acceptance_sample__02_all_signal_stages.svg)
- [Stage_3__acceptance_sample__06_wavelet_subbands.svg](figures/Stage_3__acceptance_sample__06_wavelet_subbands.svg)
- [Stage_3__acceptance_sample__04_wavelet_level_energies.svg](figures/Stage_3__acceptance_sample__04_wavelet_level_energies.svg)
- [Stage_3__acceptance_sample__05_wavelet_thresholds.svg](figures/Stage_3__acceptance_sample__05_wavelet_thresholds.svg)
- [Stage_3__acceptance_sample__09_synthetic_recovery.svg](figures/Stage_3__acceptance_sample__09_synthetic_recovery.svg)
- [Stage_3__acceptance_sample__06_wavelet_subbands.png](figures/Stage_3__acceptance_sample__06_wavelet_subbands.png)
- [Stage_3__acceptance_sample__11_residual_noise_psd.svg](figures/Stage_3__acceptance_sample__11_residual_noise_psd.svg)
- [Stage_3__acceptance_sample__05_wavelet_thresholds.png](figures/Stage_3__acceptance_sample__05_wavelet_thresholds.png)
- [Stage_3__acceptance_sample__12_time_frequency_energy.svg](figures/Stage_3__acceptance_sample__12_time_frequency_energy.svg)
- [Stage_3__acceptance_sample__08_spectrogram_comparison.svg](figures/Stage_3__acceptance_sample__08_spectrogram_comparison.svg)
- [Stage_3__acceptance_sample__10_residual_noise.png](figures/Stage_3__acceptance_sample__10_residual_noise.png)
- [Stage_3__acceptance_sample__04_wavelet_level_energies.png](figures/Stage_3__acceptance_sample__04_wavelet_level_energies.png)
- [Stage_3__acceptance_sample__01_weighted_vs_denoised.svg](figures/Stage_3__acceptance_sample__01_weighted_vs_denoised.svg)
- [Stage_3__acceptance_sample__10_residual_noise.svg](figures/Stage_3__acceptance_sample__10_residual_noise.svg)
- [Stage_3__acceptance_sample__08_spectrogram_comparison.png](figures/Stage_3__acceptance_sample__08_spectrogram_comparison.png)
- [Stage_3__acceptance_sample__09_synthetic_recovery.png](figures/Stage_3__acceptance_sample__09_synthetic_recovery.png)
- [Stage_3__acceptance_sample__07_chatter_band_overlap.svg](figures/Stage_3__acceptance_sample__07_chatter_band_overlap.svg)
- [Stage_3__acceptance_sample__03_psd_before_after.svg](figures/Stage_3__acceptance_sample__03_psd_before_after.svg)
- [Stage_3__acceptance_sample__01_weighted_vs_denoised.png](figures/Stage_3__acceptance_sample__01_weighted_vs_denoised.png)
- [Stage_3__acceptance_sample__12_time_frequency_energy.png](figures/Stage_3__acceptance_sample__12_time_frequency_energy.png)
- [Stage_3__acceptance_sample__07_chatter_band_overlap.png](figures/Stage_3__acceptance_sample__07_chatter_band_overlap.png)

## 7. Stage 4 feature extraction

| Recording | Metric | Measured value |
| --- | --- | --- |
| acceptance_sample | all_defined_values_finite | yes |
| acceptance_sample | decisions_generated | no |
| acceptance_sample | defined_feature_values | 255 |
| acceptance_sample | defined_fraction | 1 |
| acceptance_sample | feature_schema_version | 1.0.0 |
| acceptance_sample | feature_selection_performed | no |
| acceptance_sample | model_training_performed | no |
| acceptance_sample | physics_metadata_valid_for_all_windows | yes |
| acceptance_sample | probabilities_generated | no |
| acceptance_sample | repeat_extraction_stability.absolute_tolerance | 1e-12 |
| acceptance_sample | repeat_extraction_stability.all_values_within_tolerance | yes |
| acceptance_sample | repeat_extraction_stability.deterministic | yes |
| acceptance_sample | repeat_extraction_stability.exact_value_match | yes |
| acceptance_sample | repeat_extraction_stability.feature_comparison_count.early_chatter_band_energy_growth | 3 |
| acceptance_sample | repeat_extraction_stability.feature_comparison_count.early_energy_growth_rate | 3 |
| acceptance_sample | repeat_extraction_stability.feature_comparison_count.early_hegr | 3 |

| CSV artifact | Data rows | Columns |
| --- | --- | --- |
| Stage_4/acceptance_sample/window_features.csv | 3 | 92 |
| Stage_4/aggregate/all_recording_features.csv | 3 | 100 |
| Stage_4/aggregate/feature_correlations.csv | 87 | 88 |
| Stage_4/aggregate/feature_missingness.csv | 100 | 4 |
| Stage_4/aggregate/feature_repeatability.csv | 85 | 5 |
| Stage_4/aggregate/feature_summary.csv | 87 | 9 |

- [Stage_4__acceptance_sample__01_rms_timeline.png](figures/Stage_4__acceptance_sample__01_rms_timeline.png)
- [Stage_4__acceptance_sample__04_chatter_energy_timeline.png](figures/Stage_4__acceptance_sample__04_chatter_energy_timeline.png)
- [Stage_4__acceptance_sample__07_instantaneous_energy_timeline.png](figures/Stage_4__acceptance_sample__07_instantaneous_energy_timeline.png)
- [Stage_4__acceptance_sample__10_feature_family_summary.png](figures/Stage_4__acceptance_sample__10_feature_family_summary.png)
- [Stage_4__acceptance_sample__06_hegr_timeline.svg](figures/Stage_4__acceptance_sample__06_hegr_timeline.svg)
- [Stage_4__acceptance_sample__03_spectral_entropy_timeline.svg](figures/Stage_4__acceptance_sample__03_spectral_entropy_timeline.svg)
- [Stage_4__acceptance_sample__08_imf_gate_values.svg](figures/Stage_4__acceptance_sample__08_imf_gate_values.svg)
- [Stage_4__acceptance_sample__10_feature_family_summary.svg](figures/Stage_4__acceptance_sample__10_feature_family_summary.svg)
- [Stage_4__acceptance_sample__08_imf_gate_values.png](figures/Stage_4__acceptance_sample__08_imf_gate_values.png)
- [Stage_4__acceptance_sample__02_kurtosis_timeline.svg](figures/Stage_4__acceptance_sample__02_kurtosis_timeline.svg)
- [Stage_4__acceptance_sample__07_instantaneous_energy_timeline.svg](figures/Stage_4__acceptance_sample__07_instantaneous_energy_timeline.svg)
- [Stage_4__acceptance_sample__05_harmonic_energy_timeline.png](figures/Stage_4__acceptance_sample__05_harmonic_energy_timeline.png)
- [Stage_4__acceptance_sample__03_spectral_entropy_timeline.png](figures/Stage_4__acceptance_sample__03_spectral_entropy_timeline.png)
- [Stage_4__acceptance_sample__04_chatter_energy_timeline.svg](figures/Stage_4__acceptance_sample__04_chatter_energy_timeline.svg)
- [Stage_4__acceptance_sample__02_kurtosis_timeline.png](figures/Stage_4__acceptance_sample__02_kurtosis_timeline.png)
- [Stage_4__acceptance_sample__09_wavelet_energy_ratios.svg](figures/Stage_4__acceptance_sample__09_wavelet_energy_ratios.svg)
- [Stage_4__acceptance_sample__09_wavelet_energy_ratios.png](figures/Stage_4__acceptance_sample__09_wavelet_energy_ratios.png)
- [Stage_4__acceptance_sample__06_hegr_timeline.png](figures/Stage_4__acceptance_sample__06_hegr_timeline.png)
- [Stage_4__acceptance_sample__05_harmonic_energy_timeline.svg](figures/Stage_4__acceptance_sample__05_harmonic_energy_timeline.svg)
- [Stage_4__acceptance_sample__01_rms_timeline.svg](figures/Stage_4__acceptance_sample__01_rms_timeline.svg)
- [Stage_4__aggregate__aggregate_features_grouped_by_label.png](figures/Stage_4__aggregate__aggregate_features_grouped_by_label.png)
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
- [Stage_4__aggregate__aggregate_features_grouped_by_label.svg](figures/Stage_4__aggregate__aggregate_features_grouped_by_label.svg)
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
| Stage_2 | 100 | 100 | 57 | 0 |  |
| Stage_3 | 100 | 100 | 49 | 0 |  |
| Stage_4 | 100 | 100 | 52 | 0 |  |

- [stage_scorecard.png](figures/stage_scorecard.png)
- [stage_progress.png](figures/stage_progress.png)

## 9. Warnings and failures

_None recorded._

## 10. Limitations

- The production workflow intentionally ends after Stage 4 feature extraction.
- Real-signal SNR improvement is not reported without a known clean reference.
