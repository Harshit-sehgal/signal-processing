"""Shared data structures for the PG-AMCD pipeline.

The pipeline's typed results previously lived in ``pipeline.py``. They were
relocated here to establish a single source of truth (architectural objective:
"one source of truth"); both ``pipeline.py`` and ``cli.py`` import from this
module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np


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


@dataclass
class CutoffOptimizationResult:
    """Result of adaptive cutoff optimisation (Sprint 4 / Goal 5.1)."""

    selected_cutoff: float
    per_cutoff_metrics: List[Dict[str, Any]] = field(default_factory=list)
    best_score: float = 0.0


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
