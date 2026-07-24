"""Shared data structures for the PG-AMCD pipeline.

The pipeline's typed results previously lived in ``pipeline.py``. They were
relocated here to establish a single source of truth (architectural objective:
"one source of truth"); both ``pipeline.py`` and ``cli.py`` import from this
module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Stage1Output:
    """Canonical preprocessing and CEEMDAN output for one controlled segment."""

    time: np.ndarray
    raw_signal: np.ndarray
    preprocessed_physical: np.ndarray
    preprocessed_scaled: np.ndarray
    segment_time: np.ndarray
    segment_raw: np.ndarray
    segment_physical: np.ndarray
    segment_scaled: np.ndarray
    imfs_scaled: np.ndarray
    residual_scaled: np.ndarray
    imfs_physical: np.ndarray
    residual_physical: np.ndarray
    start_index: int
    end_index: int
    sampling_rate: float
    scale_factor: float
    selected_cutoff: float
    random_seed: int
    ceemdan_parameters: Dict[str, Any]
    cutoff_search: List[Dict[str, Any]]
    imf_metrics: List[Dict[str, Any]]
    seed_stability: Dict[str, Any]
    metrics: Dict[str, Any]
    runtime_seconds: float


@dataclass
class Stage2Output:
    """Canonical physics-guided IMF gating output."""

    indicators: List[Dict[str, Any]]
    gates: np.ndarray
    weighted_scaled: np.ndarray
    weighted_physical: np.ndarray
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    runtime_seconds: float


@dataclass
class Stage3Output:
    """Canonical reconstruction-level wavelet-denoising output."""

    coefficients: List[np.ndarray]
    threshold_rows: List[Dict[str, Any]]
    denoised_scaled: np.ndarray
    denoised_physical: np.ndarray
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    runtime_seconds: float
    synthetic_signals: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Stage4Output:
    """Canonical per-window feature extraction output."""

    feature_rows: List[Dict[str, Any]]
    feature_records: List[Dict[str, Any]]
    feature_schema: Dict[str, Any]
    feature_quality: Dict[str, Any]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    runtime_seconds: float


@dataclass
class WindowResult:
    """Per-window analysis result."""

    time_segment: np.ndarray
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    features: Dict[str, float]
    chatter_probability: float
    predicted_label: str
    selected_imfs: List[int]
    confidence: float
    imfs: np.ndarray  # (num_layers, N)
    maiw_reconstructed: np.ndarray  # (N,)
    denoised_clean: np.ndarray  # (N,)
    gates: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


@dataclass
class PipelineResult:
    """Aggregated result of :func:`pg_amcd.pipeline.process_recording`."""

    raw_signal: np.ndarray
    physical_preprocessed_signal: np.ndarray
    scaled_preprocessed_signal: np.ndarray
    window_results: List[WindowResult]
    sampling_rate: float
    scale_factors: Dict[str, float]
    selected_parameters: Dict[str, Any]
    warnings: List[str]
    recording_id: str = "recording"
    input_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_1: Stage1Output | None = None
    stage_2: Stage2Output | None = None
    stage_3: Stage3Output | None = None
    stage_4: Stage4Output | None = None


@dataclass
class CutoffOptimizationResult:
    """Result of adaptive cutoff optimisation (Sprint 4 / Goal 5.1)."""

    selected_cutoff: float
    per_cutoff_metrics: List[Dict[str, Any]] = field(default_factory=list)
    best_score: float = 0.0
    second_best_score: float = 0.0
    gap_to_second_best: float = 0.0
    seed_consistency: Optional[float] = None
    per_seed_best: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecordingMetadata:
    """Recording-level metadata for the chatter-detection index (Goal 6.1)."""

    recording_id: str
    file_path: str
    experiment_run_id: str = ""
    stickout: float = 0.0
    rpm: float = 0.0
    depth_of_cut: float = 0.0
    feed_rate: float = 0.0
    tooth_count: int = 1
    tool_id: str = ""
    sensor_id: str = ""
    label: str = "unknown"
    chatter_onset_time: float = float("nan")


@dataclass
class RunProvenance:
    """Complete run provenance (Segment 4 / Goal 4.3)."""

    run_id: str
    config_sha256: str
    dataset_index_sha256: str
    git_commit: str
    git_dirty: bool
    command_line: str
    start_iso: str
    end_iso: str
    random_seeds: List[int] = field(default_factory=list)
    per_file_runtime: Dict[str, float] = field(default_factory=dict)
    total_runtime: float = 0.0
    selected_cutoffs: Dict[str, float] = field(default_factory=dict)
    output_checksums: Dict[str, str] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Per-window chatter detection outcome (Goal 6)."""

    recording_id: str
    window_start: float
    window_end: float
    chatter_probability: float
    predicted_label: str
    confidence: float
