"""Negative and controlled edge contracts for Stage 1 cutoff optimisation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pg_amcd.optimization as optimization
from pg_amcd.decomposition import CEEMDANResult


def _signal(samples: int = 256, fs: float = 1000.0) -> np.ndarray:
    time = np.arange(samples) / fs
    return np.sin(2.0 * np.pi * 80.0 * time) + 0.3 * np.sin(2.0 * np.pi * 250.0 * time)


def _config() -> dict[str, Any]:
    return {
        "preprocessing": {
            "filter_order": 2,
            "low_pass_cutoff_hz": 400.0,
            "scale_percentile": 99.0,
            "padtype": "odd",
            "padlen": None,
        },
        "ceemdan": {
            "trials": 2,
            "search_trials": 1,
            "epsilon": 0.02,
            "noise_seed": 10,
            "search_sifting_iterations": 3,
            "parallel": False,
        },
        "maiw": {
            "chatter_band_center": 250.0,
            "chatter_band_spread": 40.0,
        },
    }


@pytest.mark.parametrize(
    ("segment", "message"),
    [
        (np.ones(8, dtype=complex) * (1.0 + 1.0j), "real-valued"),
        (np.ones((2, 8)), "one-dimensional"),
        (np.ones(2), "at least 3 samples"),
        (np.asarray([0.0, np.nan, 1.0]), "NaN or infinite"),
    ],
)
def test_raw_segment_validation(segment: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        optimization.optimize_cutoff(segment, [20.0], _config(), 1000.0)


def test_preprocessing_config_resolves_supported_aliases() -> None:
    resolved = optimization._resolve_preprocessing_config(
        {
            "preprocessing": {
                "upper_cutoff_hz": 350.0,
                "filter_order": 4,
                "detrend_type": "constant",
                "detrend_before_filter": True,
                "scale_percentile": 98.0,
                "padtype": None,
                "padlen": 0,
            }
        },
        1000.0,
    )
    assert resolved == {
        "lowpass_cutoff_hz": 350.0,
        "filter_order": 4,
        "detrend_type": "constant",
        "detrend_before_filter": True,
        "scale_percentile": 98.0,
        "padtype": None,
        "padlen": 0,
    }


@pytest.mark.parametrize("order", [True, "bad", 0, 1.5, np.nan])
def test_preprocessing_config_rejects_noninteger_filter_orders(order: object) -> None:
    config = {"preprocessing": {"filter_order": order}}
    with pytest.raises(ValueError, match="filter_order must be a positive integer"):
        optimization._resolve_preprocessing_config(config, 1000.0)


@pytest.mark.parametrize("padlen", [True, "bad", -1, 1.5, np.nan])
def test_preprocessing_config_rejects_invalid_pad_length(padlen: object) -> None:
    config = {"preprocessing": {"padlen": padlen}}
    with pytest.raises(ValueError, match="padlen must be a non-negative integer"):
        optimization._resolve_preprocessing_config(config, 1000.0)


@pytest.mark.parametrize(
    "weights",
    [
        {"spectral_overlap": -1.0},
        {"spectral_overlap": np.nan},
        {
            "spectral_overlap": 0.0,
            "maximum_adjacent_correlation": 0.0,
            "absolute_orthogonality": 0.0,
            "frequency_ordering_penalty": 0.0,
            "seed_instability": 0.0,
            "chatter_band_distortion": 0.0,
        },
    ],
)
def test_objective_weights_must_be_finite_nonnegative_and_nonzero(
    weights: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="weights|positive"):
        optimization._resolve_objective_weights({"cutoff_search": {"objective_weights": weights}})


def test_objective_weights_are_normalised() -> None:
    weights = optimization._resolve_objective_weights(
        {
            "cutoff_search": {
                "objective_weights": {
                    "spectral_overlap": 2.0,
                    "maximum_adjacent_correlation": 1.0,
                }
            }
        }
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["spectral_overlap"] > weights["maximum_adjacent_correlation"]


def test_band_energy_and_distortion_edges_are_explicit() -> None:
    signal = _signal()
    zeros = np.zeros_like(signal)
    assert optimization._band_energy(signal, 1000.0, 600.0, 700.0) == 0.0
    assert (
        optimization.calculate_chatter_band_distortion(signal, signal, 1000.0, 1000.0, 10.0) == 0.0
    )
    assert optimization.calculate_chatter_band_distortion(zeros, zeros, 1000.0, 250.0, 40.0) == 0.0
    assert optimization.calculate_chatter_band_distortion(zeros, signal, 1000.0, 250.0, 40.0) == 1.0


@pytest.mark.parametrize(
    ("candidate", "fs", "center", "spread", "message"),
    [
        (np.ones(255), 1000.0, 250.0, 40.0, "identical shapes"),
        (_signal(), 1000.0, -1.0, 40.0, "center_hz"),
        (_signal(), 1000.0, np.nan, 40.0, "center_hz"),
        (_signal(), 1000.0, 250.0, 0.0, "spread_hz"),
        (_signal(), 1000.0, 250.0, np.nan, "spread_hz"),
        (_signal(), 0.0, 250.0, 40.0, "fs must be finite and positive"),
    ],
)
def test_chatter_distortion_validates_shape_band_and_sampling_rate(
    candidate: np.ndarray,
    fs: float,
    center: float,
    spread: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        optimization.calculate_chatter_band_distortion(_signal(), candidate, fs, center, spread)


@pytest.mark.parametrize("components", [np.ones(32), np.empty((0, 32))])
def test_cutoff_objective_requires_at_least_one_component(
    components: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="at least one component"):
        optimization.compute_cutoff_objective(components, np.ones(32), 1000.0, _config())


def test_single_component_objective_is_finite_and_clipped() -> None:
    signal = _signal()
    score = optimization.compute_cutoff_objective(
        signal[np.newaxis, :],
        signal,
        1000.0,
        _config(),
        seed_instability=3.0,
        chatter_band_distortion=-2.0,
    )
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_seed_resolution_supports_explicit_configured_and_generated_values() -> None:
    assert optimization._resolve_seed_values({}, 2, [7, 8]) == [7, 8]
    assert optimization._resolve_seed_values({"search_seed_values": [3, 4]}, 9, None) == [3, 4]
    assert optimization._resolve_seed_values({"noise_seed": 20}, 3, None) == [
        20,
        21,
        22,
    ]


@pytest.mark.parametrize(
    ("n_seeds", "seeds", "message"),
    [
        (True, None, "positive integer"),
        (1.5, None, "positive integer"),
        (0, None, "positive integer"),
        (2, [], "At least one"),
        (2, [1, 1], "unique"),
    ],
)
def test_seed_resolution_rejects_invalid_cardinality(
    n_seeds: object, seeds: list[int] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        optimization._resolve_seed_values({}, n_seeds, seeds)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fs", "cutoffs", "message"),
    [
        (0.0, [20.0], "fs must be finite and positive"),
        (np.nan, [20.0], "fs must be finite and positive"),
        (1000.0, [], "non-empty sequence"),
        (1000.0, [np.nan], "must all be finite"),
        (1000.0, [20.0, 20.0], "must be unique"),
        (1000.0, [0.0], "0 < cutoff"),
        (1000.0, [400.0], "0 < cutoff"),
    ],
)
def test_cutoff_search_validates_sampling_rate_and_candidates(
    fs: float, cutoffs: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        optimization.optimize_cutoff(_signal(), cutoffs, _config(), fs)


def test_cutoff_search_forwards_ceemdan_options_and_explicit_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_decompose(
        signal: np.ndarray,
        trials: int,
        epsilon: float,
        seed: int,
        sifting_iterations: int,
        **options: Any,
    ) -> CEEMDANResult:
        calls.append(
            {
                "trials": trials,
                "epsilon": epsilon,
                "seed": seed,
                "sifting_iterations": sifting_iterations,
                **options,
            }
        )
        imfs = np.vstack((0.7 * signal, 0.2 * signal))
        residual = signal - np.sum(imfs, axis=0)
        return CEEMDANResult(
            imfs=imfs,
            residual=residual,
            components=np.vstack((imfs, residual)),
            parameters={"algorithm": "CEEMDAN"},
            runtime_seconds=0.01,
            residual_verified=True,
            residual_verification_nrmse=0.0,
            reconstruction_nrmse=0.0,
        )

    monkeypatch.setattr(optimization, "decompose_ceemdan", fake_decompose)
    config = _config()
    config["ceemdan"].update(
        {
            "processes": 2,
            "max_imf": 4,
            "noise_scale": 0.7,
            "noise_kind": "uniform",
            "range_threshold": 0.0,
            "total_power_threshold": 0.0,
            "beta_progress": False,
        }
    )

    result = optimization.optimize_cutoff(_signal(), [20.0], config, 1000.0, seeds=[31, 32])

    assert result.selected_cutoff == 20.0
    assert [call["seed"] for call in calls] == [31, 32]
    assert all(call["trials"] == 1 for call in calls)
    assert all(call["sifting_iterations"] == 3 for call in calls)
    assert all(call["parallel"] is False for call in calls)
    assert all(call["processes"] == 2 for call in calls)
    assert all(call["max_imf"] == 4 for call in calls)
    assert all(call["noise_scale"] == 0.7 for call in calls)
    assert all(call["noise_kind"] == "uniform" for call in calls)
    assert all(call["range_threshold"] == 0.0 for call in calls)
    assert all(call["total_power_threshold"] == 0.0 for call in calls)
    assert all(call["beta_progress"] is False for call in calls)
    assert result.per_cutoff_metrics[0]["ceemdan_seeds"] == [31, 32]


def test_scalar_seed_summary_rejects_empty_and_nonfinite_runs() -> None:
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        optimization.multi_seed_stability(float, [])
    with pytest.raises(ValueError, match="non-finite"):
        optimization.multi_seed_stability(lambda _seed: np.nan, [1])

    single = optimization.multi_seed_stability(float, [7])
    assert single["std"] == 0.0
    assert single["ci95_low"] == single["ci95_high"] == 7.0
