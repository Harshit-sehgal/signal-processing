"""Traceable Stage 1--4 scorecards derived from completed run artifacts.

The scorer is intentionally filesystem-driven.  It never accepts caller-supplied
scores or measured values: every check reads ``run_manifest.json`` or an artifact
inside ``Stage_1`` through ``Stage_4``.  Test evidence belongs in the manifest,
preferably under ``stage_evidence.<Stage_N>.tests`` with boolean ``unit``,
``synthetic`` and ``integration`` fields.
"""

from __future__ import annotations

import csv
import json
import math
import re
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ("Stage_1", "Stage_2", "Stage_3", "Stage_4")
RUBRIC: dict[str, float] = {
    "algorithmic_correctness": 20.0,
    "input_output_validation": 15.0,
    "quantitative_metrics": 15.0,
    "automated_tests": 15.0,
    "required_artifacts": 15.0,
    "visualisations": 10.0,
    "reproducibility_provenance": 5.0,
    "documentation_accuracy": 5.0,
}

STAGE_METRICS_FILE = {
    "Stage_1": "stage_1_metrics.json",
    "Stage_2": "stage_2_metrics.json",
    "Stage_3": "stage_3_metrics.json",
    "Stage_4": "stage_4_metrics.json",
}

PER_RECORDING_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "Stage_1": (
        "preprocessed_physical.npz",
        "preprocessed_scaled.npz",
        "decomposition.npz",
        "imf_metrics.csv",
        "cutoff_search.csv",
        "stage_1_metrics.json",
        "stage_1_summary.md",
        "stage_1_config.json",
    ),
    "Stage_2": (
        "imf_indicators.csv",
        "imf_gates.csv",
        "weighted_reconstruction_scaled.npz",
        "weighted_reconstruction_physical.npz",
        "stage_2_metrics.json",
        "stage_2_summary.md",
        "stage_2_config.json",
    ),
    "Stage_3": (
        "wavelet_coefficients.npz",
        "wavelet_thresholds.csv",
        "denoised_scaled.npz",
        "denoised_physical.npz",
        "stage_3_metrics.json",
        "stage_3_summary.md",
        "stage_3_config.json",
    ),
    "Stage_4": (
        "window_features.csv",
        "window_features.json",
        "feature_schema.json",
        "feature_quality.json",
        "stage_4_metrics.json",
        "stage_4_summary.md",
        "stage_4_config.json",
    ),
}

STAGE_4_AGGREGATE_ARTIFACTS = (
    "all_recording_features.csv",
    "feature_summary.csv",
    "feature_missingness.csv",
    "feature_correlations.csv",
    "feature_schema.json",
)

# Each item is (traceable check id, accepted descriptive filename fragments).
VISUAL_REQUIREMENTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "Stage_1": (
        ("raw_signal", ("raw_signal", "raw_full_signal")),
        ("selected_segment", ("selected_segment",)),
        ("preprocessing_comparison", ("preprocessing_comparison", "raw_vs_preprocessed")),
        ("psd_comparison", ("psd_comparison", "raw_preprocessed_psd")),
        ("cutoff_search", ("cutoff_search", "cutoff_objective")),
        ("ceemdan_decomposition", ("ceemdan_decomposition", "decomposition_plot")),
        ("individual_imfs", ("individual_imfs", "individual_imf", "imf_01")),
        ("residual", ("residual_plot", "residual")),
        ("imf_energy_distribution", ("imf_energy_distribution",)),
        ("imf_centre_frequency", ("imf_frequency_ordering", "imf_centre_frequency")),
        ("imf_bandwidth", ("imf_bandwidth",)),
        ("adjacent_imf_correlation", ("adjacent_imf_correlation", "imf_correlation_heatmap")),
        ("seed_stability", ("seed_stability",)),
        ("reconstruction_error", ("reconstruction_error",)),
        ("time_frequency", ("time_frequency", "spectrogram", "scalogram")),
    ),
    "Stage_2": (
        ("imf_gate_values", ("imf_gate_values", "gate_values")),
        ("imf_indicator_comparison", ("imf_indicator_comparison", "indicator_comparison")),
        ("frequency_vs_gate", ("frequency_vs_gate", "centre_frequency_vs_gate")),
        ("energy_vs_gate", ("energy_vs_gate",)),
        ("chatter_energy_vs_gate", ("chatter_energy_vs_gate", "chatter_band_energy_vs_gate")),
        ("forced_harmonic_vs_gate", ("forced_harmonic", "harmonic_energy_vs_gate")),
        ("weighted_reconstruction", ("weighted_reconstruction", "before_after_waveform")),
        ("weighted_psd_comparison", ("weighted_psd_comparison", "before_after_psd")),
        ("retained_suppressed_imfs", ("retained_suppressed", "retained_vs_suppressed")),
        ("gate_stability", ("gate_stability",)),
        ("harmonic_markers", ("harmonic_markers",)),
        ("chatter_band_highlight", ("chatter_band_highlight", "chatter_band_psd")),
    ),
    "Stage_3": (
        ("weighted_vs_denoised", ("weighted_vs_denoised",)),
        ("all_signal_stages", ("all_signal_stages", "signal_stages")),
        ("psd_before_after", ("psd_before_after", "denoised_psd")),
        ("wavelet_level_energies", ("wavelet_level_energies", "coefficient_energy")),
        ("wavelet_thresholds", ("wavelet_thresholds", "threshold_values")),
        ("wavelet_subbands", ("wavelet_subband", "subband_frequency")),
        ("chatter_band_overlap", ("chatter_band_overlap",)),
        ("spectrogram_comparison", ("spectrogram_comparison", "spectrogram_before_after")),
        ("synthetic_recovery", ("synthetic_recovery",)),
        ("residual_noise_waveform", ("residual_noise_waveform", "residual_noise")),
        ("residual_noise_psd", ("residual_noise_psd",)),
        ("time_frequency_energy", ("time_frequency_energy",)),
        ("cumulative_retention", ("cumulative_retention", "stage_retention")),
    ),
    "Stage_4": (
        ("rms_timeline", ("rms_timeline",)),
        ("kurtosis_timeline", ("kurtosis_timeline",)),
        ("spectral_entropy_timeline", ("spectral_entropy_timeline",)),
        ("chatter_energy_timeline", ("chatter_energy_timeline",)),
        ("harmonic_energy_timeline", ("harmonic_energy_timeline",)),
        ("hegr_timeline", ("hegr_timeline",)),
        ("instantaneous_energy_timeline", ("instantaneous_energy",)),
        ("imf_gate_values", ("imf_gate_values", "gate_values")),
        ("wavelet_energy_ratios", ("wavelet_energy_ratios",)),
        ("feature_family_summary", ("feature_family_summary",)),
    ),
}

STAGE_4_AGGREGATE_VISUALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("feature_distributions", ("aggregate_feature_distributions", "feature_distributions")),
    ("feature_missingness", ("aggregate_feature_missingness", "feature_missingness")),
    ("feature_correlations", ("aggregate_feature_correlation_heatmap", "correlation_heatmap")),
    ("feature_variance", ("aggregate_feature_variance", "feature_variance")),
    (
        "grouped_by_recording",
        (
            "aggregate_feature_values_by_recording",
            "feature_values_by_recording",
            "grouped_by_recording",
        ),
    ),
    (
        "grouped_by_rpm",
        (
            "aggregate_feature_values_by_rpm",
            "feature_values_by_rpm",
            "grouped_by_rpm",
        ),
    ),
    (
        "grouped_by_stickout",
        (
            "aggregate_feature_values_by_stickout",
            "feature_values_by_stickout",
            "grouped_by_stickout",
        ),
    ),
    (
        "grouped_by_depth_of_cut",
        (
            "aggregate_feature_values_by_depth_of_cut",
            "feature_values_by_depth",
            "grouped_by_depth",
        ),
    ),
    ("feature_stability", ("aggregate_feature_stability", "feature_stability")),
)

METRIC_REQUIREMENTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "Stage_1": (
        ("number_of_imfs", ("number_of_imfs", "num_imfs", "imf_count")),
        ("reconstruction_nrmse", ("reconstruction_nrmse", "nrmse")),
        ("absolute_orthogonality_index", ("absolute_orthogonality_index", "absolute_oi")),
        ("signed_orthogonality_index", ("signed_orthogonality_index", "signed_oi")),
        (
            "mean_adjacent_imf_correlation",
            ("mean_adjacent_imf_correlation", "mean_adjacent_correlation"),
        ),
        (
            "maximum_adjacent_imf_correlation",
            ("maximum_adjacent_imf_correlation", "max_adjacent_correlation"),
        ),
        ("spectral_overlap", ("spectral_overlap",)),
        ("frequency_ordering_score", ("frequency_ordering_score", "frequency_ordering_index")),
        ("per_imf_energy_percentage", ("energy_percentage", "energy_percentages")),
        ("per_imf_centre_frequency", ("centre_frequency", "center_frequency")),
        ("per_imf_bandwidth", ("bandwidth",)),
        ("per_imf_spectral_entropy", ("spectral_entropy",)),
        (
            "seed_centre_frequency_stability",
            ("seed_centre_frequency", "centre_frequency_stability", "centre_frequency_instability"),
        ),
        (
            "seed_energy_distribution_stability",
            ("seed_energy_distribution", "energy_distribution_stability", "energy_distribution_l1"),
        ),
        ("seed_matched_imf_correlation", ("matched_imf_correlation", "matched_imf_correlations")),
        (
            "seed_spectral_overlap_variation",
            ("spectral_overlap_variation", "spectral_overlap_standard_deviation"),
        ),
        (
            "seed_imf_count_variation",
            ("imf_count_variation", "imf_count_range", "imf_count_mismatch_fraction"),
        ),
        ("cutoff_search", ("cutoff_search", "selected_cutoff")),
        ("ceemdan_runtime", ("ceemdan_runtime", "runtime_seconds")),
    ),
    "Stage_2": (
        ("gate_per_imf", ("gate", "gate_value", "weight")),
        ("imf_source_correlation", ("source_correlation", "correlation_with_source")),
        ("imf_relative_energy", ("relative_energy", "energy_ratio")),
        ("imf_kurtosis", ("kurtosis",)),
        ("imf_chatter_band_energy_ratio", ("chatter_band_energy_ratio",)),
        ("imf_spindle_harmonic_energy_ratio", ("spindle_harmonic_energy_ratio",)),
        (
            "imf_tooth_harmonic_energy_ratio",
            ("tooth_harmonic_energy_ratio", "tooth_passing_harmonic_energy_ratio"),
        ),
        ("imf_frequency_proximity", ("frequency_proximity",)),
        ("imf_centre_frequency", ("centre_frequency", "center_frequency")),
        ("imf_bandwidth", ("bandwidth",)),
        ("imf_spectral_entropy", ("spectral_entropy",)),
        ("chatter_band_retention", ("chatter_band_retention",)),
        ("spindle_harmonic_attenuation", ("spindle_harmonic_attenuation",)),
        ("tooth_harmonic_attenuation", ("tooth_harmonic_attenuation",)),
        ("out_of_band_attenuation", ("out_of_band_attenuation",)),
        ("rms_before", ("rms_before",)),
        ("rms_after", ("rms_after",)),
        ("energy_before", ("energy_before",)),
        ("energy_after", ("energy_after",)),
        ("source_correlation", ("source_correlation", "correlation_with_preprocessed")),
        ("spectral_distortion", ("spectral_distortion",)),
        ("gate_vector_stability", ("gate_vector_stability", "gate_stability")),
        ("selected_imf_consistency", ("selected_imf_consistency",)),
        ("reconstruction_runtime", ("reconstruction_runtime", "runtime_seconds")),
    ),
    "Stage_3": (
        ("rms_before", ("rms_before",)),
        ("rms_after", ("rms_after",)),
        ("energy_before", ("energy_before",)),
        ("energy_after", ("energy_after",)),
        ("chatter_band_retention", ("chatter_band_retention",)),
        ("out_of_band_attenuation", ("out_of_band_attenuation",)),
        ("spectral_distortion", ("spectral_distortion",)),
        ("correlation_before", ("correlation_before", "input_correlation")),
        ("correlation_after", ("correlation_after", "input_output_correlation")),
        ("wavelet_coefficient_energy", ("coefficient_energy", "wavelet_energy", "input_energy")),
        ("threshold_by_level", ("threshold", "threshold_by_level")),
        ("noise_sigma", ("noise_sigma", "estimated_noise_sigma")),
        ("transient_preservation", ("transient_preservation",)),
        ("synthetic_reference_rmse", ("synthetic_reference_rmse", "synthetic_rmse")),
        ("synthetic_reference_snr", ("synthetic_reference_snr", "synthetic_snr")),
        ("runtime", ("runtime", "runtime_seconds")),
    ),
}

FEATURE_FAMILIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "time_domain": (
        ("rms",),
        ("variance",),
        ("standard_deviation", "std"),
        ("peak_to_peak",),
        ("mean_absolute",),
        ("crest_factor",),
        ("kurtosis",),
        ("skewness",),
        ("impulse_factor",),
        ("shape_factor",),
        ("clearance_factor",),
        ("zero_crossing",),
    ),
    "frequency_domain": (
        ("spectral_centroid", "freq_centroid"),
        ("spectral_spread", "freq_spread"),
        ("spectral_entropy", "freq_entropy"),
        ("peak_frequency", "freq_peak"),
        ("chatter_band_energy_ratio", "freq_chatter_band_ratio"),
        ("spindle_harmonic_energy_ratio", "freq_spindle_harmonic_ratio"),
        ("tooth", "harmonic_energy_ratio"),
        ("sideband_energy_ratio", "sideband_ratio"),
        ("spectral_kurtosis",),
        ("band_energy_ratio",),
    ),
    "imf_domain": (
        ("imf_count",),
        ("imf_energy_ratio",),
        ("imf_centre_frequency", "imf_center_frequency"),
        ("imf_bandwidth",),
        ("imf_entropy",),
        ("imf_kurtosis",),
        ("imf_source_correlation",),
        ("imf_gate",),
        ("mode_mixing",),
        ("maximum_adjacent_correlation", "max_adjacent_correlation"),
        ("orthogonality_index",),
        ("frequency_ordering",),
        ("selected_imf_count", "imf_selected_count"),
    ),
    "wavelet_time_frequency": (
        ("wavelet_energy",),
        ("wavelet_energy_ratio",),
        ("wavelet_entropy",),
        ("high_frequency_coefficient_ratio", "high_freq_ratio"),
        ("time_frequency_concentration",),
        ("dominant_time_frequency_ridge", "wavelet_dominant_ridge"),
    ),
    "early_chatter": (
        ("instantaneous_amplitude",),
        ("instantaneous_energy",),
        ("energy_growth_rate",),
        ("hegr",),
        ("chatter_band_energy_growth",),
        ("short_term_spectral_energy_growth",),
    ),
    "physics_guided": (
        ("chatter_band_energy",),
        ("frequency_proximity",),
        ("spindle_frequency",),
        ("tooth_passing_frequency",),
        ("harmonic_distance",),
        ("sideband_strength",),
        ("forced_vibration_energy",),
        ("chatter_to_harmonic_energy_ratio",),
    ),
}


@dataclass(frozen=True)
class CheckResult:
    """One reproducible scoring check."""

    check_id: str
    description: str
    points: float
    passed: bool
    evidence: str


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _recording_dirs(stage_dir: Path) -> list[Path]:
    if not stage_dir.is_dir():
        return []
    return sorted(
        path
        for path in stage_dir.iterdir()
        if path.is_dir() and path.name not in {"aggregate", "figures", "report"}
    )


def _flatten_items(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            norm_key = _normalise(str(key))
            path = f"{prefix}_{norm_key}" if prefix else norm_key
            flattened[path] = child
            flattened.update(_flatten_items(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flattened.update(_flatten_items(child, f"{prefix}_{index}"))
    return flattened


def _key_matches(keys: Iterable[str], aliases: Sequence[str]) -> bool:
    normalised_aliases = tuple(_normalise(alias) for alias in aliases)

    def ordered_tokens(alias: str, key: str) -> bool:
        alias_tokens = alias.split("_")
        key_tokens = key.split("_")
        position = 0
        for token in alias_tokens:
            try:
                position = key_tokens.index(token, position) + 1
            except ValueError:
                return False
        return True

    return any(
        any(alias in key or ordered_tokens(alias, key) for alias in normalised_aliases)
        for key in keys
    )


def _stage_sources(manifest: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for parent_key in ("stage_evidence", "quality_gates", "stages", "stage_status"):
        parent = manifest.get(parent_key)
        if isinstance(parent, Mapping):
            child = parent.get(stage)
            if isinstance(child, Mapping):
                sources.append(child)
    # Keep stage-neutral top-level evidence available without allowing a lookup
    # for one stage to borrow nested evidence from a different stage.
    stage_containers = {"stage_evidence", "quality_gates", "stages", "stage_status"}
    sources.append({key: value for key, value in manifest.items() if key not in stage_containers})
    return sources


def _evidence_bool(manifest: Mapping[str, Any], stage: str, aliases: Sequence[str]) -> bool | None:
    wanted = tuple(_normalise(alias) for alias in aliases)
    for source in _stage_sources(manifest, stage):
        for key, value in _flatten_items(source).items():
            if any(key == alias or key.endswith(f"_{alias}") for alias in wanted):
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)) and value in (0, 1):
                    return bool(value)
                if isinstance(value, str):
                    lowered = value.lower().strip()
                    if lowered in {"pass", "passed", "success", "true", "completed"}:
                        return True
                    if lowered in {"fail", "failed", "false", "error"}:
                        return False
    return None


def _weighted_checks(
    category: str,
    outcomes: Sequence[tuple[str, str, bool, str]],
) -> list[CheckResult]:
    if not outcomes:
        raise ValueError(f"Category {category} has no scoring checks")
    points = RUBRIC[category] / len(outcomes)
    return [
        CheckResult(check_id, description, points, passed, evidence)
        for check_id, description, passed, evidence in outcomes
    ]


def _stage_has_failure(manifest: Mapping[str, Any], stage: str) -> bool:
    failures = manifest.get("failures", [])
    if not isinstance(failures, list):
        return False
    return any(
        isinstance(item, Mapping) and _normalise(str(item.get("stage", ""))) == _normalise(stage)
        for item in failures
    )


def _stage_completed(
    manifest: Mapping[str, Any], stage: str, records: Sequence[Path]
) -> tuple[bool, str]:
    for source in _stage_sources(manifest, stage):
        status = source.get("status")
        if isinstance(status, str):
            passed = status.lower() in {"completed", "success", "succeeded", "passed"}
            return passed and not _stage_has_failure(manifest, stage), f"recorded status={status!r}"
    global_status = str(manifest.get("status", "")).lower()
    passed = bool(records) and global_status in {"completed", "success", "succeeded"}
    passed = passed and not _stage_has_failure(manifest, stage)
    return passed, f"global status={global_status or 'missing'}; recordings={len(records)}"


def _artifact_exists_for_all(records: Sequence[Path], filename: str) -> tuple[bool, str]:
    missing = [
        record.name
        for record in records
        if not (record / filename).is_file() or (record / filename).stat().st_size == 0
    ]
    passed = bool(records) and not missing
    return (
        passed,
        "all recordings"
        if passed
        else f"missing/empty for: {', '.join(missing) or 'no recordings'}",
    )


def _parseable(path: Path) -> bool:
    try:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                return next(csv.reader(handle), None) is not None
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return bool(data.files)
        else:
            return path.stat().st_size > 0
    except (OSError, ValueError, json.JSONDecodeError, csv.Error):
        return False
    return True


def _machine_outputs_parseable(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    names = [
        name
        for name in PER_RECORDING_ARTIFACTS[stage]
        if Path(name).suffix in {".json", ".csv", ".npz"}
    ]
    paths = [record / name for record in records for name in names]
    if stage == "Stage_4":
        paths.extend(
            (records[0].parent / "aggregate" / name) for name in STAGE_4_AGGREGATE_ARTIFACTS
        ) if records else None
    failed = [
        str(path.relative_to(records[0].parent.parent)) if records else str(path)
        for path in paths
        if not path.is_file() or not _parseable(path)
    ]
    return (
        bool(paths) and not failed,
        "all machine outputs parse" if not failed else f"unreadable: {', '.join(failed)}",
    )


def _nonfinite_fields(value: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            bad.extend(_nonfinite_fields(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            bad.extend(_nonfinite_fields(child, f"{prefix}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        bad.append(prefix or "value")
    return bad


def _finite_or_explained(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    bad: list[str] = []
    for record in records:
        for path in record.iterdir():
            try:
                if path.suffix == ".json":
                    bad.extend(
                        f"{record.name}/{path.name}:{field}"
                        for field in _nonfinite_fields(json.loads(path.read_text(encoding="utf-8")))
                    )
                elif path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as data:
                        for key in data.files:
                            array = np.asarray(data[key])
                            if np.issubdtype(array.dtype, np.number) and not np.all(
                                np.isfinite(array)
                            ):
                                bad.append(f"{record.name}/{path.name}:{key}")
                elif path.suffix == ".csv":
                    text = path.read_text(encoding="utf-8").lower()
                    if re.search(r"(^|[,\s])(nan|[-+]?inf)(?=$|[,\s])", text):
                        # Stage 4 may explicitly document undefined features.
                        quality = (
                            _safe_json(record / "feature_quality.json")
                            if stage == "Stage_4"
                            else None
                        )
                        if not quality or not any(
                            token in json.dumps(quality).lower()
                            for token in ("undefined", "missing", "reason")
                        ):
                            bad.append(f"{record.name}/{path.name}")
            except (OSError, ValueError, json.JSONDecodeError):
                bad.append(f"{record.name}/{path.name}:unreadable")
    return not bad and bool(
        records
    ), "finite or explicitly explained" if not bad else f"non-finite: {', '.join(bad)}"


def _metric_keys(record: Path, stage: str) -> set[str]:
    keys: set[str] = set()
    metrics = _safe_json(record / STAGE_METRICS_FILE[stage]) or {}
    keys.update(_flatten_items(metrics).keys())
    for path in record.glob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
            keys.update(_normalise(value) for value in header)
        except (OSError, csv.Error):
            continue
    return keys


def _metric_outcomes(stage: str, records: Sequence[Path]) -> list[tuple[str, str, bool, str]]:
    if stage != "Stage_4":
        outcomes = []
        for metric_id, aliases in METRIC_REQUIREMENTS[stage]:
            missing = [
                record.name
                for record in records
                if not _key_matches(_metric_keys(record, stage), aliases)
            ]
            outcomes.append(
                (
                    f"metric_{metric_id}",
                    f"Required metric is recorded: {metric_id}",
                    bool(records) and not missing,
                    "present for all recordings"
                    if not missing
                    else f"missing for: {', '.join(missing)}",
                )
            )
        return outcomes

    available: set[str] = set()
    for record in records:
        schema = _safe_json(record / "feature_schema.json") or {}
        available.update(_flatten_items(schema).keys())
        try:
            with (record / "window_features.csv").open(newline="", encoding="utf-8") as handle:
                available.update(_normalise(item) for item in next(csv.reader(handle), []))
        except (OSError, csv.Error):
            pass
    outcomes = []
    for family, requirements in FEATURE_FAMILIES.items():
        missing = [
            "/".join(aliases) for aliases in requirements if not _key_matches(available, aliases)
        ]
        outcomes.append(
            (
                f"feature_family_{family}",
                f"Required {family.replace('_', ' ')} features are represented",
                bool(records) and not missing,
                "complete" if not missing else f"missing: {', '.join(missing)}",
            )
        )
    return outcomes


def _find_numeric(mapping: Mapping[str, Any], aliases: Sequence[str]) -> list[float]:
    values: list[float] = []
    wanted = tuple(_normalise(alias) for alias in aliases)

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalised = _normalise(str(key))
                # Numeric sanity checks need the named value itself.  A loose
                # substring match would, for example, treat
                # ``imf_count_variation`` as the actual IMF count.
                if any(normalised == alias or normalised.endswith(f"_{alias}") for alias in wanted):
                    candidates = child if isinstance(child, list) else [child]
                    for candidate in candidates:
                        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
                            values.append(float(candidate))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(mapping)
    return values


def _first_numeric_array(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                array = np.asarray(data[key])
                if np.issubdtype(array.dtype, np.number) and array.ndim >= 1:
                    return array
    except (OSError, ValueError):
        return None
    return None


def _core_scientific_sanity(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    if not records:
        return False, "no recording outputs"
    if stage == "Stage_1":
        for record in records:
            metrics = _safe_json(record / STAGE_METRICS_FILE[stage]) or {}
            counts = _find_numeric(metrics, ("number_of_imfs", "num_imfs", "imf_count"))
            nrmse = _find_numeric(metrics, ("reconstruction_nrmse", "nrmse"))
            ordering = _find_numeric(
                metrics, ("frequency_ordering_score", "frequency_ordering_index")
            )
            if not counts or min(counts) < 1 or not nrmse or min(nrmse) < 0:
                return False, f"invalid IMF count or NRMSE in {record.name}"
            if not ordering or any(value < 0 or value > 1 for value in ordering):
                return False, f"invalid frequency-ordering score in {record.name}"
        return True, "IMF count, NRMSE and frequency ordering are physically bounded"
    if stage == "Stage_2":
        for record in records:
            try:
                with (record / "imf_gates.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except OSError:
                return False, f"gate table missing for {record.name}"
            gate_key = next(
                (
                    key
                    for key in (rows[0] if rows else {})
                    if "gate" in _normalise(key) or "weight" in _normalise(key)
                ),
                None,
            )
            if not rows or gate_key is None:
                return False, f"gate values missing for {record.name}"
            try:
                gates = [float(row[gate_key]) for row in rows]
            except (KeyError, TypeError, ValueError):
                return False, f"invalid gate values for {record.name}"
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in gates):
                return False, f"out-of-bounds gate in {record.name}"
            decomposition = record.parent.parent / "Stage_1" / record.name / "decomposition.npz"
            try:
                with np.load(decomposition, allow_pickle=False) as data:
                    imfs = np.asarray(data["imfs"])
                if imfs.ndim != 2 or len(gates) != imfs.shape[0]:
                    return False, f"gate/IMF count mismatch in {record.name}"
            except (OSError, ValueError, KeyError):
                return False, f"Stage 1 IMF traceability missing for {record.name}"
        return True, "all gates are finite, bounded and match physical IMF counts"
    if stage == "Stage_3":
        for record in records:
            denoised = _first_numeric_array(record / "denoised_scaled.npz")
            weighted = _first_numeric_array(
                record.parent.parent
                / "Stage_2"
                / record.name
                / "weighted_reconstruction_scaled.npz"
            )
            if denoised is None or weighted is None or denoised.shape[-1] != weighted.shape[-1]:
                return False, f"output length mismatch in {record.name}"
            try:
                with (record / "wavelet_thresholds.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    rows = list(csv.DictReader(handle))
            except OSError:
                rows = []
            if not rows:
                return False, f"no stored thresholds for {record.name}"
        return True, "output lengths match and level thresholds are stored"

    for record in records:
        schema = _safe_json(record / "feature_schema.json") or {}
        metrics = _safe_json(record / STAGE_METRICS_FILE[stage]) or {}
        version = schema.get("feature_schema_version") or schema.get("version")
        repeatability = metrics.get("repeat_extraction_stability")
        try:
            with (record / "window_features.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            rows = []
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version) or not rows:
            return False, f"versioned schema or window rows missing for {record.name}"
        if not isinstance(repeatability, Mapping) or repeatability.get("deterministic") is not True:
            return False, f"repeated-extraction stability missing or failing for {record.name}"
    return True, "versioned schemas, window rows and repeated-extraction stability are present"


def _semantic_contract(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    texts: list[str] = []
    for record in records:
        for path in record.glob("*.md"):
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        for path in record.glob("*_config.json"):
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    combined = "\n".join(texts).lower()
    if stage == "Stage_1":
        passed = "ceemdan" in combined and "iceemdan" not in combined
        return (
            passed,
            "CEEMDAN is named accurately"
            if passed
            else "summary/config must say CEEMDAN, not ICEEMDAN",
        )
    if stage == "Stage_2":
        passed = "residual" in combined and any(
            token in combined for token in ("exclude", "include", "explicit")
        )
        return (
            passed,
            "residual handling is explicit" if passed else "residual handling is not documented",
        )
    if stage == "Stage_3":
        passed = any(
            token in combined
            for token in (
                "reconstruction-level",
                "reconstruction level",
                "imf-specific",
                "imf specific",
            )
        )
        return (
            passed,
            "denoising scope is explicit" if passed else "denoising scope is not documented",
        )
    passed = all(_schema_documented(record / "feature_schema.json") for record in records)
    return (
        passed and bool(records),
        "feature formulas and units are documented"
        if passed
        else "feature schema documentation is incomplete",
    )


def _schema_entries(schema: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("features", "schema", "fields"):
        value = schema.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            entries: list[Mapping[str, Any]] = []
            for name, item in value.items():
                if isinstance(item, Mapping):
                    entries.append({"name": name, **item})
            return entries
    return []


def _schema_documented(path: Path) -> bool:
    schema = _safe_json(path) or {}
    entries = _schema_entries(schema)
    required_groups = (
        ("name", "feature_name"),
        ("family", "feature_family"),
        ("description", "formula"),
        ("unit", "units"),
        ("required_source_stage", "source_stage"),
        ("metadata_required", "requires_metadata"),
        ("dimensionless", "is_dimensionless"),
        ("invalid_handling", "undefined_handling", "handling"),
    )
    return bool(entries) and all(
        all(any(alias in entry for alias in aliases) for aliases in required_groups)
        for entry in entries
    )


def _physical_traceability(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    if not records:
        return False, "no recordings"
    if stage == "Stage_1":
        for record in records:
            try:
                with np.load(record / "decomposition.npz", allow_pickle=False) as data:
                    keys = {_normalise(key) for key in data.files}
                    has_scale = _key_matches(keys, ("scale_factor",))
                    has_physical = _key_matches(keys, ("physical_source", "source_physical"))
                    has_scaled = _key_matches(keys, ("scaled_source", "source_scaled"))
                if not (has_scale and has_physical and has_scaled):
                    return False, f"scale/source traceability missing in {record.name}"
            except (OSError, ValueError):
                return False, f"decomposition unreadable in {record.name}"
        return True, "scaled and physical sources plus scale factors are stored"
    if stage in {"Stage_2", "Stage_3"}:
        prefix = "weighted_reconstruction" if stage == "Stage_2" else "denoised"
        for record in records:
            scaled = _first_numeric_array(record / f"{prefix}_scaled.npz")
            physical = _first_numeric_array(record / f"{prefix}_physical.npz")
            if scaled is None or physical is None or scaled.shape != physical.shape:
                return False, f"scaled/physical pair invalid in {record.name}"
        return True, "scaled and physical outputs have matching shapes"
    passed = all(_schema_documented(record / "feature_schema.json") for record in records)
    return (
        passed,
        "feature units are schema-traceable"
        if passed
        else "feature units are not fully documented",
    )


def _no_out_of_scope_outputs(run_dir: Path) -> tuple[bool, str]:
    forbidden_dirs = [
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and path.name in {"Stage_5", "Stage_6", "Stage_7"}
    ]
    forbidden_files = [
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
        and any(
            token in _normalise(path.name)
            for token in ("classifier_model", "chatter_probability", "final_decision")
        )
    ]
    passed = not forbidden_dirs and not forbidden_files
    evidence = (
        "no Stage 5--7 outputs"
        if passed
        else f"found: {', '.join(forbidden_dirs + forbidden_files)}"
    )
    return passed, evidence


def _stage_input_validated(
    stage: str, manifest: Mapping[str, Any], records: Sequence[Path]
) -> tuple[bool, str]:
    if stage == "Stage_1":
        evidence = _evidence_bool(
            manifest, stage, ("input_validation_passed", "input_contract_passed")
        )
        validation = manifest.get("input_validation")
        if evidence is None and isinstance(validation, Mapping):
            invalid = validation.get("n_invalid", validation.get("invalid_count"))
            evidence = invalid == 0 if isinstance(invalid, int) else None
        return evidence is True, f"manifest input-validation evidence={evidence}"
    if stage == "Stage_2":
        passed = _metadata_validated(records)
        physics_enabled = any(
            (_safe_json(record / "stage_2_config.json") or {}).get(
                "use_physics_gating", True
            )
            is not False
            for record in records
        )
        return (
            passed,
            (
                "RPM and tooth count validated"
                if physics_enabled
                else "non-physics MAIW baseline is explicitly configured"
            )
            if passed
            else "physics metadata is missing or invalid",
        )
    if stage == "Stage_3":
        required = {
            "wavelet_name",
            "level",
            "thresholding_mode",
            "chatter_band_threshold_multiplier",
            "noise_band_threshold_multiplier",
            "minimum_noise_sigma",
            "band_aware",
        }
        missing_records = []
        for record in records:
            config = _safe_json(record / "stage_3_config.json") or {}
            keys = set(_flatten_items(config))
            if not all(_key_matches(keys, (item,)) for item in required):
                missing_records.append(record.name)
        passed = bool(records) and not missing_records
        return (
            passed,
            "wavelet configuration validated"
            if passed
            else f"incomplete config: {', '.join(missing_records)}",
        )
    passed = all(_safe_json(record / "feature_quality.json") is not None for record in records)
    return (
        passed and bool(records),
        "feature quality/undefined handling recorded"
        if passed
        else "feature quality evidence missing",
    )


def _metadata_validated(records: Sequence[Path]) -> bool:
    if not records:
        return False
    for record in records:
        config = _safe_json(record / "stage_2_config.json") or {}
        flat = _flatten_items(config)
        physics_values = [
            value
            for key, value in flat.items()
            if key.endswith("use_physics_gating") or key.endswith("physics_guided")
        ]
        physics_enabled = not physics_values or any(
            value is True or "physics" in str(value).lower() for value in physics_values
        )
        if not physics_enabled:
            continue
        rpm = _find_numeric(config, ("rpm",))
        tooth = _find_numeric(config, ("tooth_count", "toothcount"))
        if not rpm or not tooth or min(rpm) <= 0 or min(tooth) < 1:
            return False
    return True


def _shape_validation(stage: str, records: Sequence[Path]) -> tuple[bool, str]:
    if not records:
        return False, "no recordings"
    for record in records:
        arrays: list[np.ndarray] = []
        for path in record.glob("*.npz"):
            try:
                with np.load(path, allow_pickle=False) as data:
                    arrays.extend(
                        np.asarray(data[key])
                        for key in data.files
                        if np.asarray(data[key]).ndim >= 1
                    )
            except (OSError, ValueError):
                return False, f"unreadable NPZ: {record.name}/{path.name}"
        if arrays and any(array.size == 0 for array in arrays):
            return False, f"empty array in {record.name}"
        if stage != "Stage_4" and not arrays:
            return False, f"no arrays in {record.name}"
    return True, "stored arrays/tables are non-empty and shape-valid"


def _figure_files(path: Path) -> list[Path]:
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".png", ".svg"}
    ]


def _visual_present(path: Path, aliases: Sequence[str]) -> bool:
    figures = _figure_files(path)
    return any(
        _key_matches((_normalise(figure.stem),), aliases) and figure.stat().st_size > 100
        for figure in figures
    )


def _visual_outcomes(
    stage: str, records: Sequence[Path], manifest: Mapping[str, Any]
) -> list[tuple[str, str, bool, str]]:
    outcomes = []
    for visual_id, aliases in VISUAL_REQUIREMENTS[stage]:
        # A residual plot is not applicable only when every recording explicitly says no residual.
        if stage == "Stage_1" and visual_id == "residual":
            residual_flags = [
                _evidence_bool(
                    _safe_json(record / "stage_1_metrics.json") or {}, stage, ("residual_present",)
                )
                for record in records
            ]
            if residual_flags and all(flag is False for flag in residual_flags):
                continue
        missing = [record.name for record in records if not _visual_present(record, aliases)]
        outcomes.append(
            (
                f"visual_{visual_id}",
                f"Required visual is generated: {visual_id}",
                bool(records) and not missing,
                "present for all recordings"
                if not missing
                else f"missing for: {', '.join(missing) or 'no recordings'}",
            )
        )
    if stage == "Stage_4":
        aggregate = records[0].parent / "aggregate" if records else Path("__missing__")
        for visual_id, aliases in STAGE_4_AGGREGATE_VISUALS:
            passed = aggregate.is_dir() and _visual_present(aggregate, aliases)
            outcomes.append(
                (
                    f"visual_aggregate_{visual_id}",
                    f"Required aggregate visual is generated: {visual_id}",
                    passed,
                    "present" if passed else "missing",
                )
            )
        labels_available = _evidence_bool(manifest, stage, ("labels_available",))
        if labels_available is True:
            passed = _visual_present(aggregate, ("label_group_comparison", "grouped_by_label"))
            outcomes.append(
                (
                    "visual_aggregate_label_groups",
                    "Label-group comparison is generated",
                    passed,
                    "present" if passed else "missing despite available labels",
                )
            )
    return outcomes


def _manifest_identity(manifest: Mapping[str, Any]) -> tuple[bool, str]:
    aliases = (
        ("run_id",),
        ("git_commit",),
        ("git_dirty",),
        ("start_timestamp", "start_iso"),
        ("end_timestamp", "end_iso"),
        ("cli_command", "command_line"),
        ("python_version", "env_info"),
        ("operating_system", "os", "env_info"),
        ("dependency_versions", "packages"),
        ("resolved_config",),
        ("input_files", "files_processed"),
        ("metadata_checksum",),
        ("pipeline_version",),
        ("feature_schema_version",),
    )
    flat = set(_flatten_items(manifest)) | {_normalise(str(key)) for key in manifest}
    missing = ["/".join(group) for group in aliases if not _key_matches(flat, group)]
    if manifest.get("git_dirty") is True:
        worktree_digest = manifest.get("git_worktree_sha256")
        if not isinstance(worktree_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", worktree_digest
        ):
            missing.append("git_worktree_sha256 for dirty checkout")
    return not missing, "complete run identity" if not missing else f"missing: {', '.join(missing)}"


def _stage_provenance(
    manifest: Mapping[str, Any], stage: str, records: Sequence[Path]
) -> tuple[bool, str]:
    runtime = False
    for key in ("per_stage_runtime", "stage_runtime", "stage_runtimes"):
        value = manifest.get(key)
        if isinstance(value, Mapping) and stage in value:
            runtime = True
    if not runtime:
        for source in _stage_sources(manifest, stage):
            runtime = runtime or _key_matches(
                _flatten_items(source), ("runtime", "runtime_seconds")
            )
    checksums = manifest.get("output_checksums")
    has_checksums = isinstance(checksums, Mapping) and bool(checksums)
    configs = all(any(record.glob("stage_*_config.json")) for record in records)
    passed = runtime and has_checksums and configs and bool(records)
    return passed, f"runtime={runtime}, output_checksums={has_checksums}, configs={configs}"


def _documentation_outcomes(
    stage: str, records: Sequence[Path]
) -> list[tuple[str, str, bool, str]]:
    summaries = [next(record.glob("stage_*_summary.md"), None) for record in records]
    summary_ok = bool(records) and all(
        path is not None and path.stat().st_size >= 80 for path in summaries
    )
    semantic_ok, semantic_evidence = _semantic_contract(stage, records)
    return [
        (
            "substantive_stage_summaries",
            "Human-readable stage summaries are substantive",
            summary_ok,
            "all summaries >=80 bytes" if summary_ok else "missing or placeholder summary",
        ),
        (
            "documented_stage_contract",
            "Algorithm/formula scope is documented accurately",
            semantic_ok,
            semantic_evidence,
        ),
    ]


def _artifact_outcomes(stage: str, records: Sequence[Path]) -> list[tuple[str, str, bool, str]]:
    outcomes = []
    for filename in PER_RECORDING_ARTIFACTS[stage]:
        passed, evidence = _artifact_exists_for_all(records, filename)
        outcomes.append(
            (
                f"artifact_{_normalise(filename)}",
                f"Required artifact exists: {filename}",
                passed,
                evidence,
            )
        )
    if stage == "Stage_4":
        aggregate = records[0].parent / "aggregate" if records else Path("__missing__")
        for filename in STAGE_4_AGGREGATE_ARTIFACTS:
            path = aggregate / filename
            passed = path.is_file() and path.stat().st_size > 0
            outcomes.append(
                (
                    f"artifact_aggregate_{_normalise(filename)}",
                    f"Required aggregate artifact exists: {filename}",
                    passed,
                    "present" if passed else "missing/empty",
                )
            )
    return outcomes


def _algorithmic_outcomes(
    run_dir: Path, stage: str, manifest: Mapping[str, Any], records: Sequence[Path]
) -> list[tuple[str, str, bool, str]]:
    completed, completed_evidence = _stage_completed(manifest, stage, records)
    sanity, sanity_evidence = _core_scientific_sanity(stage, records)
    traceability, traceability_evidence = _physical_traceability(stage, records)
    scoped, scoped_evidence = _no_out_of_scope_outputs(run_dir)
    return [
        (
            "stage_completed",
            "Stage completed without recorded failures",
            completed,
            completed_evidence,
        ),
        ("scientific_sanity", "Core scientific invariants hold", sanity, sanity_evidence),
        (
            "physical_traceability",
            "Physical/scaled values or units are traceable",
            traceability,
            traceability_evidence,
        ),
        ("stage_4_scope_only", "No Stage 5--7 output is active", scoped, scoped_evidence),
    ]


def _validation_outcomes(
    stage: str, manifest: Mapping[str, Any], records: Sequence[Path]
) -> list[tuple[str, str, bool, str]]:
    validated, validated_evidence = _stage_input_validated(stage, manifest, records)
    shapes, shape_evidence = _shape_validation(stage, records)
    finite, finite_evidence = _finite_or_explained(stage, records)
    return [
        (
            "stage_inputs_validated",
            "Stage-specific inputs/configuration are validated",
            validated,
            validated_evidence,
        ),
        (
            "output_shapes_valid",
            "Machine outputs have valid non-empty shapes",
            shapes,
            shape_evidence,
        ),
        (
            "finite_or_explained",
            "Numeric values are finite or explicitly undefined",
            finite,
            finite_evidence,
        ),
    ]


def _test_outcomes(manifest: Mapping[str, Any], stage: str) -> list[tuple[str, str, bool, str]]:
    aliases = {
        "unit_tests": ("unit", "unit_tests_passed", "stage_unit_tests_passed"),
        "synthetic_tests": ("synthetic", "synthetic_tests_passed", "synthetic_validation_passed"),
        "integration_tests": (
            "integration",
            "integration_tests_passed",
            "stage_integration_tests_passed",
        ),
    }
    outcomes = []
    for check_id, names in aliases.items():
        evidence = _evidence_bool(manifest, stage, names)
        outcomes.append(
            (
                check_id,
                f"{check_id.replace('_', ' ').title()} passed",
                evidence is True,
                f"manifest evidence={evidence}",
            )
        )
    return outcomes


def _explicit_cap_flags(manifest: Mapping[str, Any], stage: str) -> list[str]:
    reasons = []
    flags = (
        ("known P0 correctness issue", ("p0_correctness_issue", "known_p0_issue"), True),
        ("fabricated metrics", ("fabricated_metrics",), True),
        ("multiple active implementations", ("multiple_active_implementations",), True),
    )
    for reason, aliases, bad_value in flags:
        if _evidence_bool(manifest, stage, aliases) is bad_value:
            reasons.append(reason)
    if _evidence_bool(manifest, stage, ("integration", "integration_tests_passed")) is False:
        reasons.append("failing integration test")
    return reasons


def _score_stage(run_dir: Path, stage: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = run_dir / stage
    records = _recording_dirs(stage_dir)
    categories: dict[str, list[CheckResult]] = {
        "algorithmic_correctness": _weighted_checks(
            "algorithmic_correctness", _algorithmic_outcomes(run_dir, stage, manifest, records)
        ),
        "input_output_validation": _weighted_checks(
            "input_output_validation", _validation_outcomes(stage, manifest, records)
        ),
        "quantitative_metrics": _weighted_checks(
            "quantitative_metrics", _metric_outcomes(stage, records)
        ),
        "automated_tests": _weighted_checks("automated_tests", _test_outcomes(manifest, stage)),
        "required_artifacts": _weighted_checks(
            "required_artifacts", _artifact_outcomes(stage, records)
        ),
        "visualisations": _weighted_checks(
            "visualisations", _visual_outcomes(stage, records, manifest)
        ),
        "reproducibility_provenance": _weighted_checks(
            "reproducibility_provenance",
            [
                (
                    "complete_run_identity",
                    "Run identity contains required provenance",
                    *_manifest_identity(manifest),
                ),
                (
                    "stage_runtime_and_checksums",
                    "Stage runtime, config and output checksums are recorded",
                    *_stage_provenance(manifest, stage, records),
                ),
            ],
        ),
        "documentation_accuracy": _weighted_checks(
            "documentation_accuracy", _documentation_outcomes(stage, records)
        ),
    }

    all_checks = [check for checks in categories.values() for check in checks]
    raw_score = sum(check.points for check in all_checks if check.passed)

    # Engineering vs scientific sub-scores.
    engineering_categories = {
        "input_output_validation",
        "automated_tests",
        "required_artifacts",
        "reproducibility_provenance",
        "documentation_accuracy",
    }
    scientific_categories = {
        "algorithmic_correctness",
        "quantitative_metrics",
        "visualisations",
    }
    engineering_score = sum(
        sum(check.points for check in categories[cat] if check.passed)
        for cat in engineering_categories
    )
    scientific_score = sum(
        sum(check.points for check in categories[cat] if check.passed)
        for cat in scientific_categories
    )

    cap_reasons = _explicit_cap_flags(manifest, stage)
    artifact_checks = categories["required_artifacts"]
    visual_checks = categories["visualisations"]
    test_checks = categories["automated_tests"]
    if any(not check.passed for check in test_checks):
        cap_reasons.append("automated tests not verified or failing")
    if any(not check.passed for check in artifact_checks):
        cap_reasons.append("missing required output files")
    if any(not check.passed for check in visual_checks):
        cap_reasons.append("missing required visualisations")
    if stage == "Stage_2" and not _metadata_validated(records):
        cap_reasons.append("unvalidated metadata assumptions")
    cap_reasons = list(dict.fromkeys(cap_reasons))
    score = min(raw_score, 89.0) if cap_reasons else raw_score

    category_payload: dict[str, Any] = {}
    for name, checks in categories.items():
        category_payload[name] = {
            "earned": round(sum(check.points for check in checks if check.passed), 2),
            "possible": RUBRIC[name],
            "checks": [asdict(check) for check in checks],
        }
    failed = [check.check_id for check in all_checks if not check.passed]
    warnings = [f"Score capped at 89: {reason}" for reason in cap_reasons]
    return {
        "score": round(score, 2),
        "raw_score": round(raw_score, 2),
        "engineering_score": round(engineering_score, 2),
        "scientific_score": round(scientific_score, 2),
        "passed": [check.check_id for check in all_checks if check.passed],
        "failed": failed,
        "warnings": warnings,
        "cap_reasons": cap_reasons,
        "recording_count": len(records),
        "categories": category_payload,
    }


def calculate_stage_scorecard(run_dir: str | Path) -> dict[str, Any]:
    """Calculate, but do not persist, a Stage 1--4 scorecard for ``run_dir``."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    manifest_path = root / "run_manifest.json"
    manifest = _safe_json(manifest_path) or {}
    stages = {stage: _score_stage(root, stage, manifest) for stage in STAGES}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id", root.name),
        "rubric": RUBRIC,
        **stages,
    }


def _stage_score(payload: Mapping[str, Any], stage: str, key: str) -> float:
    return float(payload[stage].get(key, payload[stage]["score"]))


def _plot_scorecard(payload: Mapping[str, Any], output_path: Path) -> None:
    scores = [float(payload[stage]["score"]) for stage in STAGES]
    engineering = [_stage_score(payload, stage, "engineering_score") for stage in STAGES]
    scientific = [_stage_score(payload, stage, "scientific_score") for stage in STAGES]
    colors = [
        "#2e7d32" if score >= 90 else "#ef6c00" if score >= 70 else "#c62828" for score in scores
    ]
    fig = plt.figure(figsize=(18, 7))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.3, 1.3, 1.4))

    axis = fig.add_subplot(grid[0, 0])
    bars = axis.barh(STAGES, scores, color=colors)
    axis.set_xlim(0, 100)
    axis.axvline(90, color="#1565c0", linestyle="--", linewidth=1.5, label="90 target")
    axis.set_xlabel("Traceable score / 100")
    axis.set_title("PG-AMCD Stage 1–4 scorecard\n(Pipeline Completeness Score)")
    axis.legend(loc="lower right")
    for bar, score in zip(bars, scores):
        axis.text(
            min(score + 1, 97),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            fontweight="bold",
        )

    axis = fig.add_subplot(grid[0, 1])
    y = np.arange(len(STAGES))
    width = 0.35
    axis.barh(y - width / 2, engineering, width, label="Engineering", color="#2878B5")
    axis.barh(y + width / 2, scientific, width, label="Scientific", color="#E15759")
    axis.set_yticks(y, STAGES)
    axis.set_xlim(0, 100)
    axis.axvline(90, color="#1565c0", linestyle="--", linewidth=1.5, label="90 target")
    axis.set_xlabel("Score / 100")
    axis.set_title("Engineering vs Scientific score")
    axis.legend(loc="lower right")

    text_axis = fig.add_subplot(grid[0, 2])
    text_axis.axis("off")
    lines = ["Failed criteria (first 8 per stage)", ""]
    for stage in STAGES:
        failed = list(payload[stage].get("failed", []))
        suffix = f" (+{len(failed) - 8} more)" if len(failed) > 8 else ""
        summary = ", ".join(failed[:8]) if failed else "None"
        lines.append(f"{stage} [{len(failed)} failed]{suffix}")
        lines.extend(textwrap.wrap(summary, width=70) or [""])
        lines.append("")
    text_axis.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_progress(payload: Mapping[str, Any], output_path: Path) -> None:
    scores = np.asarray([float(payload[stage]["score"]) for stage in STAGES])
    engineering = np.asarray([_stage_score(payload, stage, "engineering_score") for stage in STAGES])
    scientific = np.asarray([_stage_score(payload, stage, "scientific_score") for stage in STAGES])
    passed = np.asarray([len(payload[stage].get("passed", [])) for stage in STAGES])
    failed = np.asarray([len(payload[stage].get("failed", [])) for stage in STAGES])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    colors = [
        "#2e7d32" if score >= 90 else "#ef6c00" if score >= 70 else "#c62828" for score in scores
    ]
    axes[0].bar(STAGES, scores, color=colors)
    axes[0].axhline(90, color="#1565c0", linestyle="--", label="90 target")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Pipeline Completeness Score")
    axes[0].legend()

    x = np.arange(len(STAGES))
    width = 0.35
    axes[1].bar(x - width / 2, engineering, width, label="Engineering", color="#2878B5")
    axes[1].bar(x + width / 2, scientific, width, label="Scientific", color="#E15759")
    axes[1].set_xticks(x, STAGES)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Engineering vs Scientific score")
    axes[1].legend()

    axes[2].bar(STAGES, passed, color="#2e7d32", label="passed checks")
    axes[2].bar(STAGES, failed, bottom=passed, color="#c62828", label="failed checks")
    axes[2].set_ylabel("Traceable checks")
    axes[2].set_title("Check progress")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_stage_scorecard(run_dir: str | Path) -> dict[str, str]:
    """Write ``stage_scorecard.json/png`` and ``stage_progress.png``.

    Returns string paths so callers can serialize the result directly into a
    manifest if desired.
    """

    root = Path(run_dir).resolve()
    payload = calculate_stage_scorecard(root)
    json_path = root / "stage_scorecard.json"
    scorecard_png = root / "stage_scorecard.png"
    progress_png = root / "stage_progress.png"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    _plot_scorecard(payload, scorecard_png)
    _plot_progress(payload, progress_png)
    return {
        "json": str(json_path),
        "scorecard_png": str(scorecard_png),
        "progress_png": str(progress_png),
    }


__all__ = [
    "PER_RECORDING_ARTIFACTS",
    "RUBRIC",
    "STAGES",
    "STAGE_4_AGGREGATE_ARTIFACTS",
    "calculate_stage_scorecard",
    "generate_stage_scorecard",
]
