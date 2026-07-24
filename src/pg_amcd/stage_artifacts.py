"""Writers and visualisations for the canonical Stage 1--4 output contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from pg_amcd.models import PipelineResult, Stage1Output, Stage2Output, Stage3Output, Stage4Output
from pg_amcd.visualization import (
    plot_adjacent_overlap_diagnostics,
    plot_cutoff_search,
    plot_harmonic_overlap_diagnostics,
    plot_seed_stability_per_imf,
)


STAGE_NAMES = ("Stage_1", "Stage_2", "Stage_3", "Stage_4")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame([_jsonable(row) for row in rows])
    frame.to_csv(path, index=False)


def _save_figure(fig: plt.Figure, png_path: Path, write_svg: bool, dpi: int) -> None:
    try:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if not png_path.is_file() or png_path.stat().st_size <= 100:
            raise RuntimeError(f"Figure writer produced an empty PNG: {png_path}")
        if write_svg:
            svg_path = png_path.with_suffix(".svg")
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            if not svg_path.is_file() or svg_path.stat().st_size <= 100:
                raise RuntimeError(f"Figure writer produced an empty SVG: {svg_path}")
    finally:
        plt.close(fig)


def _finite_vector(
    value: Any,
    name: str,
    *,
    expected_size: int | None = None,
    bounded: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return a finite, non-empty 1-D artifact array or fail before writing."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{name} has {array.size} samples; expected {expected_size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if bounded is not None and np.any((array < bounded[0]) | (array > bounded[1])):
        raise ValueError(f"{name} must be bounded in [{bounded[0]}, {bounded[1]}]")
    return array


def _normalise_stage_2_indicators(
    indicators: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve typed Stage 2 names and add stable artifact/scorer aliases."""

    rows: list[dict[str, Any]] = []
    for indicator in indicators:
        row = dict(indicator)
        if "kurtosis_score" in row:
            row.setdefault("kurtosis_normalized", row["kurtosis_score"])
        if "correlation" in row:
            row.setdefault("source_correlation", row["correlation"])
        rows.append(row)
    return rows


def _normalise_stage_2_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(metrics)
    if "correlation_with_source" in normalised:
        normalised.setdefault("source_correlation", normalised["correlation_with_source"])
    if "reconstruction_runtime_seconds" in normalised:
        normalised.setdefault("runtime_seconds", normalised["reconstruction_runtime_seconds"])
    return normalised


def _normalise_stage_3_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt typed wavelet diagnostics without discarding their canonical fields."""

    normalised: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        is_approximation = bool(row.get("is_approximation", False))
        row.setdefault("is_detail", not is_approximation)
        if "input_energy" in row:
            row.setdefault("coefficient_energy", row["input_energy"])
        if "chatter_overlap_fraction" in row:
            row.setdefault("chatter_band_overlap_fraction", row["chatter_overlap_fraction"])
        normalised.append(row)

    total_energy = sum(
        float(row.get("input_energy", row.get("coefficient_energy", 0.0)) or 0.0)
        for row in normalised
    )
    for row in normalised:
        energy = float(row.get("input_energy", row.get("coefficient_energy", 0.0)) or 0.0)
        row.setdefault("energy_ratio", energy / total_energy if total_energy > 0.0 else 0.0)
    return normalised


def _normalise_stage_3_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(metrics)
    if "correlation_before_after" in normalised:
        normalised.setdefault("input_output_correlation", normalised["correlation_before_after"])
        normalised.setdefault("correlation_before", 1.0)
        normalised.setdefault("correlation_after", normalised["correlation_before_after"])
    return normalised


def _line_figure(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for label, values in series:
        ax.plot(x[: len(values)], values, linewidth=0.9, label=label)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(loc="best")
    return fig


def _psd(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([0.0]), np.array([0.0])
    nperseg = min(values.size, 1024)
    frequencies, power = scipy_signal.welch(values, fs=fs, nperseg=nperseg)
    return frequencies, power


def _psd_figure(
    series: list[tuple[str, np.ndarray]],
    fs: float,
    title: str,
    chatter_band: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, values in series:
        frequencies, power = _psd(values, fs)
        ax.semilogy(frequencies, np.maximum(power, np.finfo(float).tiny), label=label)
    if chatter_band is not None:
        ax.axvspan(
            chatter_band[0], chatter_band[1], alpha=0.15, color="crimson", label="Chatter band"
        )
    ax.set(title=title, xlabel="Frequency (Hz)", ylabel="PSD (unit²/Hz)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def _bar_figure(
    labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#2878B5"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(values))
    ax.bar(positions, values, color=color)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.25)
    return fig


def _spectrogram_figure(values: np.ndarray, fs: float, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    nperseg = min(max(32, values.size // 16), 256, values.size)
    noverlap = max(0, nperseg // 2)
    frequencies, times, power = scipy_signal.spectrogram(
        values, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    mesh = ax.pcolormesh(
        times, frequencies, 10.0 * np.log10(power + np.finfo(float).tiny), shading="auto"
    )
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    return fig


def create_run_directories(run_dir: str | Path) -> dict[str, Path]:
    """Create exactly the canonical top-level directories for a run."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    directories = {name: root / name for name in STAGE_NAMES}
    directories["report"] = root / "report"
    directories["report_figures"] = root / "report" / "figures"
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _write_stage_1(
    root: Path,
    recording_id: str,
    stage: Stage1Output,
    config: dict[str, Any],
    write_svg: bool,
    dpi: int,
) -> list[Path]:
    directory = root / "Stage_1" / recording_id
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "preprocessed_physical.npz",
        time=stage.time,
        signal=stage.preprocessed_physical,
        sampling_rate=stage.sampling_rate,
        selected_cutoff=stage.selected_cutoff,
        start_index=stage.start_index,
        end_index=stage.end_index,
    )
    np.savez_compressed(
        directory / "preprocessed_scaled.npz",
        time=stage.time,
        signal=stage.preprocessed_scaled,
        scale_factor=stage.scale_factor,
        sampling_rate=stage.sampling_rate,
    )
    np.savez_compressed(
        directory / "decomposition.npz",
        time_segment=stage.segment_time,
        scaled_source_segment=stage.segment_scaled,
        physical_source_segment=stage.segment_physical,
        imfs=stage.imfs_scaled,
        imfs_scaled=stage.imfs_scaled,
        imfs_physical=stage.imfs_physical,
        residual_scaled=stage.residual_scaled,
        residual_physical=stage.residual_physical,
        start_index=stage.start_index,
        end_index=stage.end_index,
        sampling_rate=stage.sampling_rate,
        scale_factor=stage.scale_factor,
        selected_cutoff=stage.selected_cutoff,
        random_seed=stage.random_seed,
        ceemdan_parameters=json.dumps(_jsonable(stage.ceemdan_parameters), sort_keys=True),
        final_component_semantics="PyEMD CEEMDAN final returned row, verified as reconstruction residual",
    )
    _write_rows(directory / "imf_metrics.csv", stage.imf_metrics)
    _write_rows(directory / "cutoff_search.csv", stage.cutoff_search)
    write_json(
        directory / "stage_1_metrics.json",
        {**stage.metrics, "seed_stability": stage.seed_stability},
    )
    write_json(
        directory / "stage_1_config.json",
        {
            "preprocessing": config.get("preprocessing", {}),
            "ceemdan": stage.ceemdan_parameters,
            "selected_cutoff": stage.selected_cutoff,
            "source_window": [stage.start_index, stage.end_index],
            "scale_factor": stage.scale_factor,
            "sampling_rate": stage.sampling_rate,
        },
    )
    summary = (
        f"# Stage 1 — CEEMDAN decomposition\n\n"
        f"Recording: `{recording_id}`  \n"
        f"Selected high-pass cutoff: {stage.selected_cutoff:.6g} Hz  \n"
        f"Controlled source window: [{stage.start_index}, {stage.end_index})  \n"
        f"Physical scale factor: {stage.scale_factor:.8g}  \n"
        f"Physical IMF count: {stage.imfs_scaled.shape[0]}  \n"
        f"Reconstruction NRMSE: {float(stage.metrics.get('reconstruction_nrmse', 0.0)):.6g}  \n"
        "The final CEEMDAN component is stored separately as `residual_*`; it is not weighted as an IMF.\n"
    )
    (directory / "stage_1_summary.md").write_text(summary, encoding="utf-8")

    figures: list[tuple[str, plt.Figure]] = []
    figures.append(
        (
            "01_raw_signal.png",
            _line_figure(
                stage.time,
                [("Raw", stage.raw_signal)],
                "Raw full-signal waveform",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "02_selected_segment.png",
            _line_figure(
                stage.segment_time,
                [("Selected raw", stage.segment_raw)],
                "Controlled source segment",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "03_preprocessing_comparison.png",
            _line_figure(
                stage.time,
                [("Raw", stage.raw_signal), ("Preprocessed", stage.preprocessed_physical)],
                "Raw versus preprocessed signal",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "04_psd_comparison.png",
            _psd_figure(
                [("Raw", stage.raw_signal), ("Preprocessed", stage.preprocessed_physical)],
                stage.sampling_rate,
                "Raw versus preprocessed PSD",
            ),
        )
    )

    cutoff_fig = plot_cutoff_search(stage.cutoff_search, selected_cutoff=stage.selected_cutoff)
    if cutoff_fig is not None:
        figures.append(("05_cutoff_search.png", cutoff_fig))

    count = stage.imfs_scaled.shape[0]
    fig, axes = plt.subplots(count + 2, 1, figsize=(11, max(6, 1.5 * (count + 2))), sharex=True)
    axes[0].plot(stage.segment_time, stage.segment_scaled, color="black", linewidth=0.8)
    axes[0].set_ylabel("Source")
    for index, imf in enumerate(stage.imfs_scaled):
        axes[index + 1].plot(stage.segment_time, imf, linewidth=0.7)
        axes[index + 1].set_ylabel(f"IMF {index + 1}")
    axes[-1].plot(stage.segment_time, stage.residual_scaled, color="crimson", linewidth=0.8)
    axes[-1].set_ylabel("Residual")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("CEEMDAN decomposition (scaled processing units)")
    fig.tight_layout()
    figures.append(("06_ceemdan_decomposition.png", fig))

    fig, axes = plt.subplots(
        count, 1, figsize=(11, max(4, 1.5 * count)), sharex=True, squeeze=False
    )
    for index, imf in enumerate(stage.imfs_physical):
        axes[index, 0].plot(stage.segment_time, imf, linewidth=0.7)
        axes[index, 0].set_ylabel(f"IMF {index + 1}")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Individual physical-unit IMFs")
    fig.tight_layout()
    figures.append(("07_individual_imfs.png", fig))
    figures.append(
        (
            "08_residual.png",
            _line_figure(
                stage.segment_time,
                [("Residual", stage.residual_physical)],
                "CEEMDAN residual",
                "Amplitude (physical unit)",
            ),
        )
    )

    labels = [f"IMF {index + 1}" for index in range(count)]
    energies = [float(row.get("energy_percentage", 0.0)) for row in stage.imf_metrics]
    centres = [float(row.get("centre_frequency_hz", 0.0)) for row in stage.imf_metrics]
    bandwidths = [float(row.get("bandwidth_hz", 0.0)) for row in stage.imf_metrics]
    figures.append(
        (
            "09_imf_energy_distribution.png",
            _bar_figure(labels, energies, "IMF energy distribution", "Energy (%)"),
        )
    )
    figures.append(
        (
            "10_imf_frequency_ordering.png",
            _bar_figure(
                labels, centres, "IMF centre-frequency ordering", "Centre frequency (Hz)", "#F28E2B"
            ),
        )
    )
    figures.append(
        (
            "11_imf_bandwidth.png",
            _bar_figure(labels, bandwidths, "IMF bandwidth", "Bandwidth (Hz)", "#59A14F"),
        )
    )

    corr = np.corrcoef(stage.imfs_scaled) if count > 1 else np.ones((1, 1))
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set(title="IMF correlation heatmap", xlabel="IMF index", ylabel="IMF index")
    figures.append(("12_adjacent_imf_correlation.png", fig))

    # Adjacent overlap-pair diagnostics when per-pair overlap data is available.
    adjacent_overlaps = [
        float(row.get("adjacent_spectral_overlap", np.nan)) for row in stage.imf_metrics
    ][: count]
    if count > 1 and all(np.isfinite(adjacent_overlaps[: count - 1])):
        fig_overlap = plot_adjacent_overlap_diagnostics(
            labels,
            centres,
            bandwidths,
            adjacent_overlaps[: count - 1],
            [corr[i, i + 1] for i in range(count - 1)],
        )
        if fig_overlap is not None:
            figures.append(("13c_adjacent_overlap_diagnostics.png", fig_overlap))

    stability = stage.seed_stability
    stability_labels = [
        "Count mismatch",
        "Frequency instability",
        "Energy L1",
        "Overlap SD",
        "1-|corr|",
    ]
    stability_values = [
        float(stability.get("imf_count_mismatch_fraction", 0.0)),
        float(stability.get("centre_frequency_instability", 0.0)),
        float(stability.get("energy_distribution_l1", 0.0)),
        float(stability.get("spectral_overlap_standard_deviation", 0.0)),
        1.0 - float(stability.get("matched_imf_correlation_mean", 1.0)),
    ]
    figures.append(
        (
            "13_seed_stability.png",
            _bar_figure(
                stability_labels,
                stability_values,
                "Structural CEEMDAN seed stability",
                "Instability",
                "#B07AA1",
            ),
        )
    )

    # Per-IMF seed-stability diagnostics when per-seed arrays are present.
    per_seed_keys = {
        "centre_frequencies": ("per_imf_centre_frequency", "centre_frequencies"),
        "energy_percentages": ("per_imf_energy_percentage", "energy_percentages"),
        "matched_correlations": ("per_imf_matched_correlation", "matched_correlations"),
    }
    per_seed_data: dict[str, Any] = {}
    for alias, key_choices in per_seed_keys.items():
        value: Any = None
        for key in key_choices:
            if key in stability:
                value = stability[key]
                break
        if value is None and isinstance(stability.get("per_imf"), dict):
            for key in key_choices:
                value = stability["per_imf"].get(key)
                if value is not None:
                    break
        if value is not None:
            per_seed_data[alias] = value

    per_seed_ok = (
        "centre_frequencies" in per_seed_data
        and "energy_percentages" in per_seed_data
        and "matched_correlations" in per_seed_data
        and len(per_seed_data["centre_frequencies"]) == count
        and len(per_seed_data["energy_percentages"]) == count
        and len(per_seed_data["matched_correlations"]) == count
    )
    if per_seed_ok:
        cf = per_seed_data["centre_frequencies"]
        ep = per_seed_data["energy_percentages"]
        n_seed_values = {len(row) for row in cf} | {len(row) for row in ep}
        if len(n_seed_values) == 1 and 0 not in n_seed_values:
            fig_seed = plot_seed_stability_per_imf(
                labels,
                cf,
                ep,
                per_seed_data["matched_correlations"],
            )
            if fig_seed is not None:
                figures.append(("13b_seed_stability_per_imf.png", fig_seed))

    reconstruction = np.sum(stage.imfs_scaled, axis=0) + stage.residual_scaled
    error = stage.segment_scaled - reconstruction
    figures.append(
        (
            "14_reconstruction_error.png",
            _line_figure(
                stage.segment_time,
                [("Error", error)],
                "CEEMDAN reconstruction error",
                "Scaled amplitude",
            ),
        )
    )
    figures.append(
        (
            "15_time_frequency.png",
            _spectrogram_figure(
                stage.segment_physical,
                stage.sampling_rate,
                "Selected-segment time-frequency representation",
            ),
        )
    )

    paths = []
    for name, figure in figures:
        target = directory / name
        _save_figure(figure, target, write_svg, dpi)
        paths.append(target)
    return paths


def _indicator_values(indicators: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in indicators]


def _write_stage_2(
    root: Path,
    recording_id: str,
    stage_1: Stage1Output,
    stage: Stage2Output,
    write_svg: bool,
    dpi: int,
) -> list[Path]:
    directory = root / "Stage_2" / recording_id
    directory.mkdir(parents=True, exist_ok=True)
    sample_count = stage_1.segment_time.size
    gates = _finite_vector(stage.gates, "Stage 2 gates", bounded=(0.0, 1.0))
    weighted_scaled = _finite_vector(
        stage.weighted_scaled,
        "Stage 2 weighted scaled signal",
        expected_size=sample_count,
    )
    weighted_physical = _finite_vector(
        stage.weighted_physical,
        "Stage 2 weighted physical signal",
        expected_size=sample_count,
    )
    indicators = _normalise_stage_2_indicators(stage.indicators)
    if len(indicators) != gates.size or stage_1.imfs_scaled.shape[0] != gates.size:
        raise ValueError("Stage 2 indicator, gate, and physical IMF counts must match")
    metrics = _normalise_stage_2_metrics(stage.metrics)

    _write_rows(directory / "imf_indicators.csv", indicators)
    _write_rows(
        directory / "imf_gates.csv",
        [
            {
                "imf_index": index + 1,
                "gate": float(gate),
                "selected": bool(gate >= float(stage.config.get("selection_threshold", 0.5))),
            }
            for index, gate in enumerate(gates)
        ],
    )
    np.savez_compressed(
        directory / "weighted_reconstruction_scaled.npz",
        time=stage_1.segment_time,
        signal=weighted_scaled,
        gates=gates,
        residual_included=bool(stage.config.get("include_residual", False)),
    )
    np.savez_compressed(
        directory / "weighted_reconstruction_physical.npz",
        time=stage_1.segment_time,
        signal=weighted_physical,
        scale_factor=stage_1.scale_factor,
        unit_restoration="weighted_scaled * Stage_1 scale_factor",
    )
    write_json(directory / "stage_2_metrics.json", metrics)
    method = str(stage.config.get("method", "physics_guided_independent_gates"))
    physics_enabled = method == "physics_guided_independent_gates"
    write_json(
        directory / "stage_2_config.json",
        {
            "use_physics_gating": physics_enabled,
            "physics_metadata": stage.metadata,
            "gating": stage.config,
        },
    )
    selected = int(np.sum(gates >= float(stage.config.get("selection_threshold", 0.5))))
    title = "physics-guided independent IMF gating" if physics_enabled else "legacy MAIW baseline"
    gate_semantics = (
        "Gates are independent sigmoid relevance values and are deliberately not "
        "normalised to sum to one."
        if physics_enabled
        else "Weights are the explicitly requested legacy sum-normalised MAIW baseline; "
        "physics-guided gating was disabled."
    )
    (directory / "stage_2_summary.md").write_text(
        f"# Stage 2 — {title}\n\n"
        f"Recording: `{recording_id}`  \n"
        f"IMFs gated: {gates.size}  \n"
        f"IMFs above configured selection threshold: {selected}  \n"
        f"Residual included: {bool(stage.config.get('include_residual', False))}  \n"
        f"{gate_semantics}\n",
        encoding="utf-8",
    )

    labels = [f"IMF {index + 1}" for index in range(gates.size)]
    figures: list[tuple[str, plt.Figure]] = []
    figures.append(
        (
            "01_imf_gate_values.png",
            _bar_figure(labels, gates.tolist(), "Independent gate value by IMF", "Gate [0, 1]"),
        )
    )

    indicator_keys = [
        "correlation",
        "relative_energy",
        "kurtosis_normalized",
        "chatter_band_energy_ratio",
        "spindle_harmonic_energy_ratio",
        "tooth_harmonic_energy_ratio",
        "frequency_proximity",
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(gates.size)
    width = 0.8 / max(1, len(indicator_keys))
    for index, key in enumerate(indicator_keys):
        ax.bar(
            positions + index * width,
            _indicator_values(indicators, key),
            width=width,
            label=key.replace("_", " "),
        )
    ax.set_xticks(positions + width * (len(indicator_keys) - 1) / 2, labels)
    ax.set(title="IMF indicator comparison", ylabel="Indicator value")
    ax.legend(fontsize=7, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    figures.append(("02_imf_indicator_comparison.png", fig))

    def scatter_figure(key: str, title: str, xlabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = _indicator_values(indicators, key)
        points = ax.scatter(values, gates, c=np.arange(gates.size), cmap="viridis", s=65)
        fig.colorbar(points, ax=ax, label="IMF index")
        ax.set(title=title, xlabel=xlabel, ylabel="Gate")
        ax.grid(alpha=0.25)
        figures.append((filename, fig))

    scatter_figure(
        "centre_frequency_hz",
        "IMF centre frequency versus gate",
        "Centre frequency (Hz)",
        "03_frequency_vs_gate.png",
    )
    scatter_figure(
        "relative_energy", "IMF energy versus gate", "Relative energy", "04_energy_vs_gate.png"
    )
    scatter_figure(
        "chatter_band_energy_ratio",
        "Chatter-band energy versus gate",
        "Chatter-band ratio",
        "05_chatter_energy_vs_gate.png",
    )
    scatter_figure(
        "forced_harmonic_energy_ratio",
        "Forced-harmonic energy versus gate",
        "Forced-harmonic ratio",
        "06_forced_harmonics_vs_gate.png",
    )

    figures.append(
        (
            "07_weighted_reconstruction.png",
            _line_figure(
                stage_1.segment_time,
                [("Preprocessed", stage_1.segment_physical), ("Weighted", stage.weighted_physical)],
                "Before/after IMF gating",
                "Amplitude (physical unit)",
            ),
        )
    )
    center = float(stage.config.get("chatter_band_center", 0.0))
    spread = float(stage.config.get("chatter_band_spread", 0.0))
    band = (
        (max(0.0, center - spread), min(stage_1.sampling_rate / 2.0, center + spread))
        if center > 0.0
        else None
    )
    figures.append(
        (
            "08_weighted_psd_comparison.png",
            _psd_figure(
                [("Preprocessed", stage_1.segment_physical), ("Weighted", stage.weighted_physical)],
                stage_1.sampling_rate,
                "PSD before and after IMF gating",
                band,
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.arange(gates.size)
    for index, (imf, gate) in enumerate(zip(stage_1.imfs_physical, gates)):
        scaled = imf / (np.max(np.abs(imf)) + np.finfo(float).eps)
        ax.plot(
            stage_1.segment_time,
            scaled + 2.2 * index,
            color="#2A9D8F" if gate >= 0.5 else "#B0B0B0",
            linewidth=0.6,
        )
    ax.set(
        title="Retained versus suppressed IMFs", xlabel="Time (s)", ylabel="IMF (green=retained)"
    )
    ax.set_yticks(2.2 * offsets, labels)
    figures.append(("09_retained_suppressed_imfs.png", fig))

    stability = metrics.get("gate_vector_stability", {})
    if isinstance(stability, dict):
        mean_gate = stability.get("mean_gates", stability.get("mean_gate_by_imf", gates))
        gate_std = stability.get(
            "standard_deviation", stability.get("std_gate_by_imf", np.zeros_like(gates))
        )
    else:
        mean_gate, gate_std = gates, np.zeros_like(gates)
    mean_gate_array = _finite_vector(
        mean_gate,
        "Stage 2 mean gate stability",
        expected_size=gates.size,
        bounded=(0.0, 1.0),
    )
    gate_std_array = _finite_vector(
        gate_std,
        "Stage 2 gate stability standard deviation",
        expected_size=gates.size,
    )

    # Prefer structurally matched labels from gate stability; fall back to
    # indicator-derived labels if unavailable.
    if (
        isinstance(stability, dict)
        and "matched_labels" in stability
        and len(stability["matched_labels"]) == gates.size
    ):
        matched_labels = list(stability["matched_labels"])
    else:
        matched_labels = []
        for index in range(gates.size):
            cf = _indicator_values(
                [indicators[index]] if index < len(indicators) else [],
                "centre_frequency_hz",
            )
            if cf and cf[0] > 0:
                matched_labels.append(f"M{index + 1}\n{cf[0]:.0f} Hz")
            else:
                matched_labels.append(f"M{index + 1}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(
        np.arange(gates.size),
        mean_gate_array,
        yerr=gate_std_array,
        fmt="o",
        capsize=4,
    )
    ax.set_xticks(np.arange(gates.size), matched_labels)
    ax.set(title="Gate stability across CEEMDAN seeds (matched modes)", ylabel="Gate", ylim=(0.0, 1.0))
    ax.grid(alpha=0.25)
    figures.append(("10_gate_stability.png", fig))

    # Gate stability including unmatched-mode penalty if provided.
    unmatched_penalty = float(stability.get("unmatched_mode_penalty", 0.0))
    if unmatched_penalty > 0.0:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(["matched", "unmatched"], [1.0 - unmatched_penalty, unmatched_penalty])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Penalty / proportion")
        ax.set_title("Unmatched-mode stability penalty")
        ax.grid(axis="y", alpha=0.25)
        figures.append(("10b_unmatched_mode_penalty.png", fig))

    # Spindle/tooth harmonic overlap diagnostic when each ratio set contains
    # at least one positive value. Overlap/union are visual approximations
    # derived from the per-IMF ratios, not rigorous harmonic mask intersections.
    spindle_ratios = _indicator_values(indicators, "spindle_harmonic_energy_ratio")
    tooth_ratios = _indicator_values(indicators, "tooth_harmonic_energy_ratio")
    if any(v > 0 for v in spindle_ratios) and any(v > 0 for v in tooth_ratios):
        overlap = [min(s, t) for s, t in zip(spindle_ratios, tooth_ratios)]
        spindle_only = [s - o for s, o in zip(spindle_ratios, overlap)]
        tooth_only = [t - o for t, o in zip(tooth_ratios, overlap)]
        union = [max(s, t) for s, t in zip(spindle_ratios, tooth_ratios)]
        fig_harmonic = plot_harmonic_overlap_diagnostics(
            spindle_only, tooth_only, overlap, union, labels
        )
        if fig_harmonic is not None:
            figures.append(("11b_harmonic_overlap_diagnostics.png", fig_harmonic))

    frequencies, power = _psd(stage_1.segment_physical, stage_1.sampling_rate)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(frequencies, np.maximum(power, np.finfo(float).tiny))
    rpm = stage.metadata.get("rpm")
    teeth = stage.metadata.get("tooth_count")
    if rpm and teeth:
        spindle = float(rpm) / 60.0
        tooth = spindle * int(teeth)
        for harmonic in range(1, int(stage.config.get("harmonic_count", 5)) + 1):
            ax.axvline(harmonic * spindle, color="#4E79A7", alpha=0.45, linestyle=":")
            ax.axvline(harmonic * tooth, color="#E15759", alpha=0.45, linestyle="--")
    ax.set(
        title="Spindle and tooth-passing harmonic markers",
        xlabel="Frequency (Hz)",
        ylabel="PSD (unit²/Hz)",
    )
    ax.grid(alpha=0.2)
    figures.append(("11_harmonic_markers.png", fig))

    figures.append(
        (
            "12_chatter_band_psd.png",
            _psd_figure(
                [("Weighted", stage.weighted_physical)],
                stage_1.sampling_rate,
                "Weighted PSD with chatter band",
                band,
            ),
        )
    )

    paths = []
    for name, figure in figures:
        target = directory / name
        _save_figure(figure, target, write_svg, dpi)
        paths.append(target)
    return paths


def _write_stage_3(
    root: Path,
    recording_id: str,
    stage_1: Stage1Output,
    stage_2: Stage2Output,
    stage: Stage3Output,
    write_svg: bool,
    dpi: int,
) -> list[Path]:
    directory = root / "Stage_3" / recording_id
    directory.mkdir(parents=True, exist_ok=True)
    sample_count = stage_1.segment_time.size
    denoised_scaled = _finite_vector(
        stage.denoised_scaled,
        "Stage 3 denoised scaled signal",
        expected_size=sample_count,
    )
    denoised_physical = _finite_vector(
        stage.denoised_physical,
        "Stage 3 denoised physical signal",
        expected_size=sample_count,
    )
    if not stage.coefficients or not stage.threshold_rows:
        raise ValueError("Stage 3 coefficients and threshold diagnostics must be non-empty")
    if len(stage.coefficients) != len(stage.threshold_rows):
        raise ValueError("Stage 3 coefficient and threshold-row counts must match")
    threshold_rows = _normalise_stage_3_rows(stage.threshold_rows)
    coefficient_payload: dict[str, Any] = {
        f"coefficient_{index}": _finite_vector(coefficient, f"Stage 3 coefficient {index}")
        for index, coefficient in enumerate(stage.coefficients)
    }
    coefficient_payload["coefficient_order"] = np.asarray(
        [
            row.get("coefficient_name", f"coefficient_{index}")
            for index, row in enumerate(threshold_rows)
        ]
    )
    np.savez_compressed(directory / "wavelet_coefficients.npz", **coefficient_payload)
    _write_rows(directory / "wavelet_thresholds.csv", threshold_rows)
    np.savez_compressed(
        directory / "denoised_scaled.npz",
        time=stage_1.segment_time,
        signal=denoised_scaled,
        input_weighted=stage_2.weighted_scaled,
    )
    np.savez_compressed(
        directory / "denoised_physical.npz",
        time=stage_1.segment_time,
        signal=denoised_physical,
        scale_factor=stage_1.scale_factor,
        unit_restoration="denoised_scaled * Stage_1 scale_factor",
    )
    metrics = _normalise_stage_3_metrics(stage.metrics)
    write_json(directory / "stage_3_metrics.json", metrics)
    stage_3_config = {
        **stage.config,
        "level": stage.config.get(
            "level", stage.config.get("applied_level", metrics.get("resolved_level"))
        ),
        "thresholding_mode": stage.config.get(
            "thresholding_mode", stage.config.get("threshold_mode", "soft")
        ),
        "chatter_band_threshold_multiplier": stage.config.get(
            "chatter_band_threshold_multiplier",
            stage.config.get("chatter_threshold_scale"),
        ),
        "noise_band_threshold_multiplier": stage.config.get(
            "noise_band_threshold_multiplier",
            stage.config.get("noise_threshold_scale"),
        ),
        "minimum_noise_sigma": stage.config.get(
            "minimum_noise_sigma",
            stage.config.get("noise_sigma", metrics.get("estimated_noise_sigma")),
        ),
    }
    write_json(directory / "stage_3_config.json", stage_3_config)
    (directory / "stage_3_summary.md").write_text(
        "# Stage 3 — reconstruction-level adaptive wavelet denoising\n\n"
        f"Recording: `{recording_id}`  \n"
        f"Wavelet: `{stage.config.get('wavelet_name', 'unknown')}`  \n"
        f"Resolved decomposition level: {stage.metrics.get('resolved_level', stage.config.get('level', 'unknown'))}  \n"
        f"Thresholding mode: `{stage.config.get('threshold_mode', 'soft')}`  \n"
        "The canonical production approach denoises the Stage 2 weighted reconstruction; it is not described as IMF-specific.\n",
        encoding="utf-8",
    )

    fs = stage_1.sampling_rate
    center = float(stage.config.get("chatter_center", stage.config.get("chatter_band_center", 0.0)))
    spread = float(stage.config.get("chatter_spread", stage.config.get("chatter_band_spread", 0.0)))
    band = (max(0.0, center - spread), min(fs / 2.0, center + spread)) if center > 0 else None
    figures: list[tuple[str, plt.Figure]] = []
    figures.append(
        (
            "01_weighted_vs_denoised.png",
            _line_figure(
                stage_1.segment_time,
                [("Weighted", stage_2.weighted_physical), ("Denoised", stage.denoised_physical)],
                "Weighted versus denoised waveform",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "02_all_signal_stages.png",
            _line_figure(
                stage_1.segment_time,
                [
                    ("Raw", stage_1.segment_raw),
                    ("Preprocessed", stage_1.segment_physical),
                    ("Weighted", stage_2.weighted_physical),
                    ("Denoised", stage.denoised_physical),
                ],
                "Signal progression through Stage 3",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "03_psd_before_after.png",
            _psd_figure(
                [("Weighted", stage_2.weighted_physical), ("Denoised", stage.denoised_physical)],
                fs,
                "Wavelet denoising PSD comparison",
                band,
            ),
        )
    )

    detail_rows = [row for row in threshold_rows if bool(row.get("is_detail", False))]
    level_labels = [
        str(row.get("coefficient_name", row.get("level", index + 1)))
        for index, row in enumerate(detail_rows)
    ]
    coefficient_energy = [
        float(row.get("coefficient_energy", row.get("energy", 0.0))) for row in detail_rows
    ]
    threshold_values = [float(row.get("threshold", 0.0)) for row in detail_rows]
    figures.append(
        (
            "04_wavelet_level_energies.png",
            _bar_figure(
                level_labels, coefficient_energy, "Wavelet coefficient energy by level", "Energy"
            ),
        )
    )
    figures.append(
        (
            "05_wavelet_thresholds.png",
            _bar_figure(
                level_labels,
                threshold_values,
                "BayesShrink threshold by level",
                "Threshold",
                "#E15759",
            ),
        )
    )

    low_ranges = [float(row.get("frequency_low_hz", 0.0)) for row in detail_rows]
    high_ranges = [float(row.get("frequency_high_hz", 0.0)) for row in detail_rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(detail_rows))
    ax.barh(positions, np.asarray(high_ranges) - np.asarray(low_ranges), left=low_ranges)
    ax.set_yticks(positions, level_labels)
    ax.set(title="Wavelet detail-subband frequency ranges", xlabel="Frequency (Hz)")
    figures.append(("06_wavelet_subbands.png", fig))

    overlaps = [
        float(
            row.get(
                "chatter_band_overlap_fraction",
                row.get("chatter_overlap_fraction", row.get("chatter_overlap", 0.0)),
            )
        )
        for row in detail_rows
    ]
    figures.append(
        (
            "07_chatter_band_overlap.png",
            _bar_figure(
                level_labels,
                overlaps,
                "Chatter-band overlap by wavelet subband",
                "Overlap fraction",
                "#59A14F",
            ),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    for ax, values, title in zip(
        axes,
        (stage_2.weighted_physical, stage.denoised_physical),
        ("Before denoising", "After denoising"),
    ):
        nperseg = min(max(32, values.size // 16), 256, values.size)
        frequencies, times, power = scipy_signal.spectrogram(
            values, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
        )
        ax.pcolormesh(
            times, frequencies, 10.0 * np.log10(power + np.finfo(float).tiny), shading="auto"
        )
        ax.set(ylabel="Frequency (Hz)", title=title)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Spectrogram before and after wavelet denoising")
    fig.tight_layout()
    figures.append(("08_spectrogram_comparison.png", fig))

    synthetic = stage.synthetic_signals
    if {"clean", "noisy", "recovered"}.issubset(synthetic):
        time_axis = np.arange(len(synthetic["clean"])) / fs
        recovery_figure = _line_figure(
            time_axis,
            [
                ("Clean", synthetic["clean"]),
                ("Noisy", synthetic["noisy"]),
                ("Recovered", synthetic["recovered"]),
            ],
            "Synthetic ground-truth recovery",
            "Amplitude",
        )
    else:
        recovery_figure = _line_figure(
            stage_1.segment_time,
            [("Weighted input", stage_2.weighted_physical), ("Recovered", stage.denoised_physical)],
            "Recovery comparison (no clean real-signal reference)",
            "Amplitude (physical unit)",
        )
    figures.append(("09_synthetic_recovery.png", recovery_figure))

    residual_noise = stage_2.weighted_physical - stage.denoised_physical
    figures.append(
        (
            "10_residual_noise.png",
            _line_figure(
                stage_1.segment_time,
                [("Removed component", residual_noise)],
                "Residual-noise waveform",
                "Amplitude (physical unit)",
            ),
        )
    )
    figures.append(
        (
            "11_residual_noise_psd.png",
            _psd_figure([("Removed component", residual_noise)], fs, "Residual-noise PSD"),
        )
    )

    weighted_energy = np.abs(scipy_signal.hilbert(stage_2.weighted_physical)) ** 2
    denoised_energy = np.abs(scipy_signal.hilbert(stage.denoised_physical)) ** 2
    figures.append(
        (
            "12_time_frequency_energy.png",
            _line_figure(
                stage_1.segment_time,
                [
                    ("Weighted analytic energy", weighted_energy),
                    ("Denoised analytic energy", denoised_energy),
                ],
                "Time-frequency energy preservation",
                "Instantaneous energy",
            ),
        )
    )

    # Cumulative chatter-band retention across stages.
    def _chatter_band_energy(values: np.ndarray, fs: float, band: tuple[float, float]) -> float:
        frequencies, power = scipy_signal.welch(values, fs=fs, nperseg=min(values.size, 1024))
        mask = (frequencies >= band[0]) & (frequencies <= band[1])
        if not np.any(mask):
            return 0.0
        integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if integrate is None:
            raise RuntimeError("NumPy does not provide trapezoid or trapz integration")
        return float(integrate(power[mask], frequencies[mask]))

    retention_stages = ["Raw", "Stage 1", "Stage 2", "Stage 3"]
    if band is not None and band[0] < band[1]:
        raw_energy = _chatter_band_energy(stage_1.segment_raw, fs, band)
        prep_energy = _chatter_band_energy(stage_1.segment_physical, fs, band)
        weighted_energy_band = _chatter_band_energy(stage_2.weighted_physical, fs, band)
        denoised_energy_band = _chatter_band_energy(stage.denoised_physical, fs, band)
        reference = raw_energy if raw_energy > 0 else 1.0
        retention_values = [
            prep_energy / reference,
            weighted_energy_band / reference,
            denoised_energy_band / reference,
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ["#2878B5", "#59A14F", "#F28E2B", "#E15759"]
        bars = ax.bar(retention_stages[1:], retention_values, color=bar_colors)
        ax.set_ylabel("Cumulative retention")
        ax.set_ylim(0, 1)
        ax.set_title("Cumulative chatter-band retention (Stage 1 → Stage 3)")
        for bar, val in zip(bars, retention_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figures.append(("13_cumulative_retention.png", fig))

        # Persist cumulative retention as a metric and a small JSON record.
        retention_record = {
            "stages": retention_stages,
            "retention_values": [1.0] + [float(value) for value in retention_values],
            "chatter_band_hz": list(band),
        }
    else:
        retention_record = {
            "stages": retention_stages,
            "retention_values": [],
            "chatter_band_hz": [],
            "reason": "chatter band not configured or invalid",
        }

    stage.metrics["cumulative_chatter_band_retention"] = retention_record
    (directory / "cumulative_retention.json").write_text(
        json.dumps(retention_record, indent=2, default=str),
        encoding="utf-8",
    )

    paths = []
    for name, figure in figures:
        target = directory / name
        _save_figure(figure, target, write_svg, dpi)
        paths.append(target)
    return paths


def _feature_key(rows: list[dict[str, Any]], *candidates: str) -> str | None:
    keys = {key for row in rows for key in row}
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def _feature_timeline(
    rows: list[dict[str, Any]], candidates: tuple[str, ...], title: str, ylabel: str
) -> plt.Figure:
    key = _feature_key(rows, *candidates)
    times = np.asarray(
        [
            float(row.get("window_start_seconds", row.get("window_start", index)))
            for index, row in enumerate(rows)
        ],
        dtype=float,
    )
    if key is None:
        values = np.full(times.shape, np.nan)
        label = "Undefined"
    else:
        values = np.asarray([np.nan if row.get(key) is None else float(row[key]) for row in rows])
        label = key
    return _line_figure(times, [(label, values)], title, ylabel)


def _schema_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    entries = schema.get("features", schema.get("entries", []))
    if isinstance(entries, dict):
        return [
            {"name": key, **(value if isinstance(value, dict) else {})}
            for key, value in entries.items()
        ]
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _write_stage_4(
    root: Path,
    recording_id: str,
    stage_2: Stage2Output,
    stage_3: Stage3Output,
    stage: Stage4Output,
    write_svg: bool,
    dpi: int,
) -> list[Path]:
    directory = root / "Stage_4" / recording_id
    directory.mkdir(parents=True, exist_ok=True)
    if not stage.feature_rows:
        raise ValueError("Stage 4 requires at least one feature row")
    if not _schema_entries(stage.feature_schema):
        raise ValueError("Stage 4 requires a non-empty, documented feature schema")
    defined_numeric = [
        float(value)
        for row in stage.feature_rows
        for value in row.values()
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
    ]
    if not defined_numeric or not np.all(np.isfinite(defined_numeric)):
        raise ValueError("Stage 4 feature rows require finite, defined numeric values")
    _write_rows(directory / "window_features.csv", stage.feature_rows)
    write_json(directory / "window_features.json", stage.feature_records or stage.feature_rows)
    write_json(directory / "feature_schema.json", stage.feature_schema)
    write_json(directory / "feature_quality.json", stage.feature_quality)
    write_json(directory / "stage_4_metrics.json", stage.metrics)
    write_json(directory / "stage_4_config.json", stage.config)
    (directory / "stage_4_summary.md").write_text(
        "# Stage 4 — feature extraction\n\n"
        f"Recording: `{recording_id}`  \n"
        f"Feature schema: `{stage.feature_schema.get('feature_schema_version', stage.feature_schema.get('version', 'unknown'))}`  \n"
        f"Windows: {len(stage.feature_rows)}  \n"
        f"Defined feature values: {stage.metrics.get('defined_feature_values', 'see feature_quality.json')}  \n"
        f"Undefined feature values: {stage.metrics.get('undefined_feature_values', 'see feature_quality.json')}  \n"
        "This stage only extracts and validates features. It performs no feature selection, model training, probability generation, or decision output.\n",
        encoding="utf-8",
    )

    rows = stage.feature_rows
    figures: list[tuple[str, plt.Figure]] = [
        (
            "01_rms_timeline.png",
            _feature_timeline(rows, ("time_rms", "rms"), "RMS over time", "RMS (physical unit)"),
        ),
        (
            "02_kurtosis_timeline.png",
            _feature_timeline(
                rows, ("time_kurtosis", "kurtosis"), "Kurtosis over time", "Kurtosis"
            ),
        ),
        (
            "03_spectral_entropy_timeline.png",
            _feature_timeline(
                rows,
                ("freq_spectral_entropy", "freq_entropy", "spectral_entropy"),
                "Spectral entropy over time",
                "Normalised entropy",
            ),
        ),
        (
            "04_chatter_energy_timeline.png",
            _feature_timeline(
                rows,
                (
                    "physics_chatter_band_energy",
                    "freq_chatter_band_energy",
                    "freq_chatter_band_ratio",
                ),
                "Chatter-band energy over time",
                "Energy / ratio",
            ),
        ),
        (
            "05_harmonic_energy_timeline.png",
            _feature_timeline(
                rows,
                (
                    "freq_tooth_harmonic_ratio",
                    "freq_harmonics_ratio",
                    "physics_forced_vibration_energy",
                ),
                "Harmonic energy over time",
                "Energy ratio",
            ),
        ),
        (
            "06_hegr_timeline.png",
            _feature_timeline(
                rows,
                ("early_hegr", "hegr", "energy_growth_rate"),
                "Hilbert energy-growth rate (HEGR) over time",
                "Energy / s",
            ),
        ),
        (
            "07_instantaneous_energy_timeline.png",
            _feature_timeline(
                rows,
                ("early_instantaneous_energy_mean", "instantaneous_energy_mean"),
                "Instantaneous analytic energy over time",
                "Energy",
            ),
        ),
        (
            "08_imf_gate_values.png",
            _bar_figure(
                [f"IMF {index + 1}" for index in range(stage_2.gates.size)],
                stage_2.gates.astype(float).tolist(),
                "Stage 2 IMF gates used by Stage 4",
                "Gate",
            ),
        ),
    ]

    wavelet_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("wavelet_") and key.endswith("_energy_ratio")
        }
    )
    wavelet_values = []
    for key in wavelet_keys:
        defined = [float(row[key]) for row in rows if row.get(key) is not None]
        wavelet_values.append(float(np.mean(defined)) if defined else 0.0)
    if not wavelet_keys:
        threshold_rows = _normalise_stage_3_rows(stage_3.threshold_rows)
        wavelet_keys = [
            str(row.get("coefficient_name", row.get("level", index + 1)))
            for index, row in enumerate(threshold_rows)
        ]
        wavelet_values = [float(row.get("energy_ratio", 0.0)) for row in threshold_rows]
    figures.append(
        (
            "09_wavelet_energy_ratios.png",
            _bar_figure(
                wavelet_keys, wavelet_values, "Wavelet energy ratios", "Energy ratio", "#59A14F"
            ),
        )
    )

    schema_entries = _schema_entries(stage.feature_schema)
    family_values: dict[str, list[float]] = {}
    for entry in schema_entries:
        name = str(entry.get("name", ""))
        family = str(entry.get("family", "other"))
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None and isinstance(row.get(name), (int, float, np.number))
        ]
        if values:
            family_values.setdefault(family, []).extend(abs(value) for value in values)
    family_labels = sorted(family_values)
    family_medians = [float(np.median(family_values[label])) for label in family_labels]
    figures.append(
        (
            "10_feature_family_summary.png",
            _bar_figure(
                family_labels,
                family_medians,
                "Feature-family magnitude summary",
                "Median absolute value",
                "#B07AA1",
            ),
        )
    )

    paths = []
    for name, figure in figures:
        target = directory / name
        _save_figure(figure, target, write_svg, dpi)
        paths.append(target)
    return paths


def write_recording_artifacts(
    run_dir: str | Path,
    result: PipelineResult,
    config: dict[str, Any],
) -> dict[str, list[str]]:
    """Write all required per-recording Stage 1--4 files and figures."""

    create_run_directories(run_dir)
    if (
        result.stage_1 is None
        or result.stage_2 is None
        or result.stage_3 is None
        or result.stage_4 is None
    ):
        raise ValueError("A complete Stage 1--4 PipelineResult is required for artifact generation")
    output_cfg = config.get("output", {})
    write_svg = bool(output_cfg.get("write_svg", True))
    dpi = int(output_cfg.get("png_dpi", 140))
    root = Path(run_dir)
    recording_id = result.recording_id
    paths = {
        "Stage_1": _write_stage_1(root, recording_id, result.stage_1, config, write_svg, dpi),
        "Stage_2": _write_stage_2(
            root, recording_id, result.stage_1, result.stage_2, write_svg, dpi
        ),
        "Stage_3": _write_stage_3(
            root, recording_id, result.stage_1, result.stage_2, result.stage_3, write_svg, dpi
        ),
        "Stage_4": _write_stage_4(
            root, recording_id, result.stage_2, result.stage_3, result.stage_4, write_svg, dpi
        ),
    }
    return {stage: [str(path) for path in stage_paths] for stage, stage_paths in paths.items()}


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "recording_id",
        "relative_path",
        "window_index",
        "window_start_seconds",
        "window_end_seconds",
        "label",
        "tool_identifier",
        "rpm",
        "tooth_count",
        "stickout",
        "depth_of_cut",
        "feed_rate",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _group_heatmap(frame: pd.DataFrame, group: str, numeric: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(numeric)), 5))
    if group not in frame or frame[group].dropna().empty or not numeric:
        ax.text(0.5, 0.5, f"{group} metadata unavailable", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        return fig
    grouped = frame.groupby(group, dropna=True)[numeric].mean(numeric_only=True)
    values = grouped.to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    normalised = (values - medians) / scale
    image = ax.imshow(normalised, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    fig.colorbar(image, ax=ax, label="Within-feature z-score")
    ax.set_yticks(np.arange(grouped.shape[0]), [str(value) for value in grouped.index])
    ax.set_xticks(np.arange(len(numeric)), numeric, rotation=90, fontsize=6)
    ax.set(title=title, ylabel=group)
    return fig


def _feature_repeatability_figure(differences: dict[str, float], *, complete: bool) -> plt.Figure:
    """Render a readable repeated-extraction stability summary."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not differences:
        ax.text(0.5, 0.5, "Repeat-extraction evidence unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        maximum = max(differences.values())
        if complete and maximum <= np.finfo(float).eps:
            ax.text(
                0.5,
                0.58,
                "PASS",
                ha="center",
                va="center",
                fontsize=30,
                color="#2E7D32",
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.40,
                f"{len(differences)} features compared twice\nMaximum absolute difference: {maximum:.3g}",
                ha="center",
                va="center",
                fontsize=13,
            )
            ax.set_axis_off()
        else:
            ranked = sorted(differences.items(), key=lambda item: item[1], reverse=True)[:20]
            names = [item[0] for item in reversed(ranked)]
            values = [item[1] for item in reversed(ranked)]
            ax.barh(names, values, color="#76B7B2")
            ax.set_xlabel("Maximum absolute difference")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.25)
    ax.set_title("Feature stability across repeated Stage 4 extractions")
    fig.tight_layout()
    return fig


def write_aggregate_stage_4(
    run_dir: str | Path,
    results: list[PipelineResult],
    config: dict[str, Any],
) -> list[str]:
    """Write required run-level Stage 4 aggregate tables and visualisations."""

    if not results or any(
        result.stage_4 is None or not result.stage_4.feature_rows for result in results
    ):
        raise ValueError("At least one complete Stage 4 result is required for aggregation")
    directory = Path(run_dir) / "Stage_4" / "aggregate"
    directory.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, Any]] = []
    for result in results:
        assert result.stage_4 is not None
        for row in result.stage_4.feature_rows:
            combined_rows.append(
                {
                    "recording_id": result.recording_id,
                    "relative_path": result.input_path,
                    **result.metadata,
                    **row,
                }
            )
    frame = pd.DataFrame([_jsonable(row) for row in combined_rows])
    if frame.empty:
        raise ValueError("Stage 4 aggregate feature table must not be empty")
    frame.to_csv(directory / "all_recording_features.csv", index=False)
    numeric = [column for column in _numeric_feature_columns(frame) if frame[column].notna().any()]
    if not numeric:
        raise ValueError("Stage 4 aggregate requires at least one defined numeric feature")
    summary = (
        frame[numeric].describe(include="all").transpose().reset_index(names="feature")
        if numeric
        else pd.DataFrame(columns=["feature"])
    )
    summary.to_csv(directory / "feature_summary.csv", index=False)

    missing_rows = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        missing_rows.append(
            {
                "feature": column,
                "missing_count": missing,
                "total_count": int(len(frame)),
                "missing_fraction": float(missing / len(frame)) if len(frame) else 0.0,
            }
        )
    pd.DataFrame(missing_rows).to_csv(directory / "feature_missingness.csv", index=False)
    correlation = frame[numeric].corr() if numeric else pd.DataFrame()
    correlation.to_csv(directory / "feature_correlations.csv", index=True)
    schema = next(result.stage_4.feature_schema for result in results if result.stage_4 is not None)
    write_json(directory / "feature_schema.json", schema)

    output_cfg = config.get("output", {})
    write_svg = bool(output_cfg.get("write_svg", True))
    dpi = int(output_cfg.get("png_dpi", 140))
    figures: list[tuple[str, plt.Figure]] = []

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(numeric)), 5))
    if numeric:
        normalised = frame[numeric].copy()
        for column in numeric:
            median = normalised[column].median()
            spread = normalised[column].std()
            normalised[column] = (normalised[column] - median) / (
                spread if spread and np.isfinite(spread) else 1.0
            )
        ax.boxplot(
            [normalised[column].dropna().to_numpy() for column in numeric],
            tick_labels=numeric,
            showfliers=False,
        )
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.set_ylabel("Standardised value")
    ax.set_title("Aggregate feature distributions (all numeric features)")
    figures.append(("aggregate_feature_distributions.png", fig))

    missing_frame = pd.DataFrame(missing_rows)
    figures.append(
        (
            "aggregate_feature_missingness.png",
            _bar_figure(
                missing_frame["feature"].astype(str).tolist(),
                missing_frame["missing_fraction"].astype(float).tolist(),
                "Feature missingness",
                "Missing fraction",
                "#E15759",
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(numeric)), max(6, 0.28 * len(numeric))))
    if numeric:
        correlation_image = np.nan_to_num(
            correlation.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        image = ax.imshow(correlation_image, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        ax.set_xticks(np.arange(len(numeric)), numeric, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(len(numeric)), numeric, fontsize=5)
    ax.set_title("Aggregate feature correlation heatmap")
    figures.append(("aggregate_feature_correlation_heatmap.png", fig))

    variances = []
    for column in numeric:
        variance = float(frame[column].var(skipna=True))
        variances.append(variance if math.isfinite(variance) else 0.0)
    figures.append(
        (
            "aggregate_feature_variance.png",
            _bar_figure(numeric, variances, "Feature variance", "Variance", "#F28E2B"),
        )
    )
    figures.append(
        (
            "aggregate_feature_values_by_recording.png",
            _group_heatmap(frame, "recording_id", numeric, "Feature values grouped by recording"),
        )
    )
    figures.append(
        (
            "aggregate_feature_values_by_rpm.png",
            _group_heatmap(frame, "rpm", numeric, "Feature values grouped by RPM"),
        )
    )
    figures.append(
        (
            "aggregate_feature_values_by_stickout.png",
            _group_heatmap(frame, "stickout", numeric, "Feature values grouped by stickout"),
        )
    )
    figures.append(
        (
            "aggregate_feature_values_by_depth_of_cut.png",
            _group_heatmap(
                frame, "depth_of_cut", numeric, "Feature values grouped by depth of cut"
            ),
        )
    )
    if "label" in frame and not frame["label"].dropna().empty:
        figures.append(
            (
                "aggregate_features_grouped_by_label.png",
                _group_heatmap(
                    frame, "label", numeric, "Feature values grouped by available label"
                ),
            )
        )

    repeatability_differences: dict[str, float] = {}
    repeatability_counts: dict[str, int] = {}
    repeatability_complete = True
    for result in results:
        assert result.stage_4 is not None
        repeatability = result.stage_4.metrics.get("repeat_extraction_stability")
        if not isinstance(repeatability, dict) or repeatability.get("deterministic") is not True:
            repeatability_complete = False
            continue
        differences = repeatability.get("feature_maximum_absolute_difference")
        counts = repeatability.get("feature_comparison_count")
        if not isinstance(differences, dict) or not isinstance(counts, dict):
            repeatability_complete = False
            continue
        for name, raw_difference in differences.items():
            difference = float(raw_difference)
            if not math.isfinite(difference) or difference < 0.0:
                repeatability_complete = False
                continue
            repeatability_differences[str(name)] = max(
                repeatability_differences.get(str(name), 0.0), difference
            )
            repeatability_counts[str(name)] = repeatability_counts.get(str(name), 0) + int(
                counts.get(name, 0)
            )
    repeatability_names = sorted(repeatability_differences)
    repeatability_rows = [
        {
            "feature": name,
            "maximum_absolute_difference": repeatability_differences[name],
            "defined_value_comparisons": repeatability_counts.get(name, 0),
            "all_recordings_deterministic": repeatability_complete,
            "repeat_count_per_recording": 2,
        }
        for name in repeatability_names
    ]
    pd.DataFrame(repeatability_rows).to_csv(directory / "feature_repeatability.csv", index=False)
    figures.append(
        (
            "aggregate_feature_stability.png",
            _feature_repeatability_figure(
                repeatability_differences,
                complete=repeatability_complete,
            ),
        )
    )

    paths: list[str] = []
    for name, figure in figures:
        target = directory / name
        _save_figure(figure, target, write_svg, dpi)
        paths.append(str(target))
    return paths
