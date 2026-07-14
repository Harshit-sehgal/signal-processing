"""Validation and metric edge contracts for the canonical CEEMDAN core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pg_amcd.decomposition as decomposition


def _signal(samples: int = 128, fs: float = 512.0) -> np.ndarray:
    time = np.arange(samples) / fs
    return np.sin(2.0 * np.pi * 40.0 * time) + 0.3 * np.sin(2.0 * np.pi * 110.0 * time)


def _modes(frequencies: list[float], samples: int = 128, fs: float = 512.0) -> np.ndarray:
    time = np.arange(samples) / fs
    return np.stack([np.sin(2.0 * np.pi * frequency * time) for frequency in frequencies])


def _fake_ceemdan_class(
    output: Callable[[np.ndarray], np.ndarray] | None = None,
) -> type:
    class FakeCEEMDAN:
        instances: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.seed: int | None = None
            self.max_imf: int | None = None
            type(self).instances.append(self)

        def noise_seed(self, seed: int) -> None:
            self.seed = seed

        def __call__(self, signal: np.ndarray, *, max_imf: int) -> np.ndarray:
            self.max_imf = max_imf
            if output is not None:
                return output(signal)
            first = 0.65 * signal
            return np.vstack((first, signal - first))

    return FakeCEEMDAN


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.ones(8, dtype=complex) * (1.0 + 1.0j), "real-valued"),
        (np.ones((2, 8)), "one-dimensional"),
        (np.ones(2), "at least three samples"),
        (np.asarray([0.0, np.nan, 1.0]), "NaN or infinite"),
    ],
)
def test_signal_contract_rejects_invalid_inputs(value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decomposition.decompose_ceemdan(value, 1, 0.02, 1, parallel=False)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.ones((1, 8), dtype=complex) * (1.0 + 1.0j), "real-valued"),
        (np.ones(8), "two-dimensional"),
        (np.empty((0, 8)), "n_modes"),
        (np.ones((1, 2)), "n_samples"),
        (np.asarray([[0.0, np.inf, 1.0]]), "NaN or infinite"),
    ],
)
def test_imf_matrix_contract_rejects_invalid_inputs(value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decomposition.calculate_imf_metrics(value, 512.0)


def test_nrmse_zero_reference_and_shape_contracts() -> None:
    zeros = np.zeros(8)
    assert decomposition._nrmse(zeros, zeros) == 0.0
    assert decomposition._nrmse(zeros, np.ones(8)) == float("inf")
    with pytest.raises(ValueError, match="identical shapes"):
        decomposition._nrmse(np.zeros(8), np.zeros(7))


def test_unknown_emd_signal_version_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise decomposition.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(decomposition.importlib_metadata, "version", missing)
    assert decomposition._emd_signal_version() == "unknown"


@pytest.mark.parametrize(
    ("value", "kwargs", "message"),
    [
        ("not-an-int", {}, "must be an integer"),
        (True, {}, "must be an integer"),
        (1.5, {}, "must be an integer"),
        (np.nan, {}, "must be an integer"),
        (0, {}, "at least 1"),
        (-2, {"allow_minus_one": True}, "-1 or at least 1"),
    ],
)
def test_integer_options_are_strict(value: object, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decomposition._coerce_integer(value, name="option", **kwargs)
    assert decomposition._coerce_integer(-1, name="option", allow_minus_one=True) == -1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trials": 0}, "trials must be"),
        ({"epsilon": -0.1}, "epsilon must be finite and non-negative"),
        ({"epsilon": np.nan}, "epsilon must be finite and non-negative"),
        ({"sifting_iterations": 0}, "sifting_iterations must be"),
        ({"max_imf": 0}, "max_imf must be"),
        ({"noise_seed": 2**32}, r"smaller than 2\*\*32"),
        ({"processes": 0}, "processes must be"),
        ({"noise_kind": "triangular"}, "noise_kind must be"),
        ({"noise_scale": 0.0}, "noise_scale must be finite and positive"),
        ({"range_threshold": -1.0}, "range_threshold must be finite and non-negative"),
        (
            {"total_power_threshold": np.inf},
            "total_power_threshold must be finite and non-negative",
        ),
        ({"residual_rtol": -1.0}, "residual_rtol must be finite and non-negative"),
        ({"residual_atol": np.nan}, "residual_atol must be finite and non-negative"),
    ],
)
def test_ceemdan_options_fail_before_library_execution(
    kwargs: dict[str, object], message: str
) -> None:
    options: dict[str, object] = {
        "trials": 1,
        "epsilon": 0.02,
        "noise_seed": 1,
        "sifting_iterations": 2,
        "parallel": False,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        decomposition.decompose_ceemdan(_signal(), **options)  # type: ignore[arg-type]


def test_constant_signal_is_rejected_before_ceemdan() -> None:
    with pytest.raises(ValueError, match="constant or numerically flat"):
        decomposition.decompose_ceemdan(np.ones(128), 1, 0.02, 1, parallel=False)


def test_ceemdan_forwards_all_supported_options_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_ceemdan_class()
    monkeypatch.setattr(decomposition, "CEEMDAN", fake)

    result = decomposition.decompose_ceemdan(
        _signal(),
        trials=3,
        epsilon=0.03,
        noise_seed=7,
        sifting_iterations=4,
        parallel=False,
        processes=2,
        max_imf=2,
        noise_scale=0.8,
        noise_kind="uniform",
        range_threshold=0.0,
        total_power_threshold=0.0,
        beta_progress=False,
    )

    instance = fake.instances[-1]
    assert instance.seed == 7
    assert instance.max_imf == 2
    assert instance.kwargs == {
        "trials": 3,
        "epsilon": 0.03,
        "FIXE": 4,
        "parallel": False,
        "processes": 2,
        "noise_scale": 0.8,
        "noise_kind": "uniform",
        "range_thr": 0.0,
        "total_power_thr": 0.0,
        "beta_progress": False,
    }
    assert result.residual_verified is True
    assert result.reconstruction_nrmse < 1e-12
    assert result.parameters["algorithm"] == "CEEMDAN"
    assert result.parameters["processes"] == 2
    assert result.as_metadata()["number_of_imfs"] == 1
    np.testing.assert_allclose(result.reconstruction, _signal())

    wrapped = decomposition.run_ceemdan(_signal(), 1, 0.02, 8, 2, parallel=False)
    assert wrapped.shape == (2, _signal().size)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda signal: np.ones(signal.size), "unexpected component matrix shape"),
        (lambda signal: signal[np.newaxis, :], "fewer than two components"),
        (
            lambda signal: np.vstack((np.full(signal.size, np.nan), np.zeros(signal.size))),
            "non-finite component values",
        ),
        (
            lambda signal: np.vstack((0.6 * signal, 0.1 * signal)),
            "Unable to verify",
        ),
    ],
)
def test_ceemdan_library_output_is_verified(
    monkeypatch: pytest.MonkeyPatch,
    output: Callable[[np.ndarray], np.ndarray],
    message: str,
) -> None:
    monkeypatch.setattr(decomposition, "CEEMDAN", _fake_ceemdan_class(output))
    with pytest.raises(RuntimeError, match=message):
        decomposition.decompose_ceemdan(_signal(), 1, 0.02, 1, 2, parallel=False)


def test_reconstruction_metrics_validate_component_lengths() -> None:
    source = _signal(64)
    modes = np.vstack((source, np.zeros_like(source)))
    with pytest.raises(ValueError, match="IMF and source-signal lengths differ"):
        decomposition.calculate_reconstruction_nrmse(source, modes[:, :-1])
    with pytest.raises(ValueError, match="Residual and source-signal lengths differ"):
        decomposition.calculate_reconstruction_nrmse(source, modes, np.zeros(63))


def test_zero_energy_and_single_mode_metric_edges_are_defined() -> None:
    zeros = np.zeros((1, 64))
    metrics = decomposition.calculate_imf_metrics(zeros, 512.0)
    assert metrics[0].energy_percentage == 0.0
    assert metrics[0].centre_frequency_hz == 0.0
    assert metrics[0].bandwidth_hz == 0.0
    assert metrics[0].spectral_entropy == 0.0
    assert decomposition.calculate_orthogonality_metrics(zeros).signed_index == 0.0
    assert decomposition.calculate_adjacent_imf_correlations(zeros).size == 0
    assert decomposition.calculate_adjacent_spectral_overlaps(zeros, 512.0).size == 0
    np.testing.assert_array_equal(decomposition.calculate_imf_correlation_matrix(zeros), np.eye(1))
    assert decomposition.frequency_ordering_score_from_centres([40.0]) == 1.0

    legacy_components = np.vstack((zeros, zeros))
    assert decomposition.calculate_adjacent_imf_correlation(legacy_components) == (
        0.0,
        0.0,
    )
    assert decomposition.calculate_spectral_overlap(legacy_components, 512.0) == 0.0


def test_constant_modes_have_zero_pairwise_correlations() -> None:
    modes = np.vstack((np.ones(64), np.ones(64) * 2.0, np.arange(64)))
    np.testing.assert_array_equal(
        decomposition.calculate_adjacent_imf_correlations(modes), [0.0, 0.0]
    )
    matrix = decomposition.calculate_imf_correlation_matrix(modes)
    assert matrix[0, 1] == 0.0
    assert matrix[1, 2] == 0.0
    assert decomposition._absolute_correlation(np.ones(64), np.arange(64)) == 0.0


def test_frequency_metric_inputs_and_sampling_rate_are_validated() -> None:
    for centres in ([], [[1.0, 2.0]], [10.0, np.nan]):
        with pytest.raises(ValueError, match="non-empty finite vector"):
            decomposition.frequency_ordering_score_from_centres(centres)
    with pytest.raises(ValueError, match="Sampling rate must be finite and positive"):
        decomposition.calculate_imf_metrics(_modes([40.0]), 0.0)


def test_decomposition_metric_bundle_handles_explicit_and_absent_residuals() -> None:
    modes = _modes([120.0, 40.0], samples=128)
    residual = np.linspace(-0.05, 0.05, 128)
    source = np.sum(modes, axis=0) + residual
    result = decomposition.CEEMDANResult(
        imfs=modes,
        residual=residual,
        components=np.vstack((modes, residual)),
        parameters={"algorithm": "CEEMDAN"},
        runtime_seconds=0.25,
        residual_verified=True,
        residual_verification_nrmse=0.0,
        reconstruction_nrmse=0.0,
    )

    explicit = decomposition.calculate_decomposition_metrics(source, result, 512.0)
    assert explicit["ceemdan_runtime_seconds"] == 0.25
    assert explicit["reconstruction_nrmse"] < 1e-12

    raw = decomposition.calculate_decomposition_metrics(np.sum(modes, axis=0), modes, 512.0)
    assert "ceemdan_runtime_seconds" not in raw
    assert raw["reconstruction_nrmse"] < 1e-12

    with pytest.raises(ValueError, match="IMF and source-signal lengths differ"):
        decomposition.calculate_decomposition_metrics(source, modes[:, :-1], 512.0)
    with pytest.raises(ValueError, match="Residual and source-signal lengths differ"):
        decomposition.calculate_decomposition_metrics(source, modes, 512.0, residual=np.zeros(127))
    with pytest.raises(ValueError, match="IMF and source-signal lengths differ"):
        decomposition.calculate_orthogonality_index(modes[:, :-1], source)


def test_seed_stability_validates_cardinality_and_handles_one_seed() -> None:
    modes = _modes([120.0, 40.0], samples=128)
    with pytest.raises(ValueError, match="At least one decomposition"):
        decomposition.calculate_seed_stability([], 512.0)
    with pytest.raises(ValueError, match="same sample count"):
        decomposition.calculate_seed_stability([modes, _modes([120.0], samples=127)], 512.0)
    with pytest.raises(ValueError, match="seeds length"):
        decomposition.calculate_seed_stability([modes, modes], 512.0, seeds=[1])

    single = decomposition.calculate_seed_stability([modes], 512.0)
    assert single["seeds"] == [0]
    assert single["pairwise_comparisons"] == []
    assert single["matched_imf_correlation_mean"] == 1.0
    assert single["instability_score"] == 0.0


def test_seed_matching_accounts_for_unmatched_modes_in_both_directions() -> None:
    fewer = _modes([120.0, 40.0], samples=128)
    more = _modes([150.0, 80.0, 25.0], samples=128)

    forward = decomposition.calculate_seed_stability([fewer, more], 512.0, seeds=[1, 2])
    reverse = decomposition.calculate_seed_stability([more, fewer], 512.0, seeds=[2, 1])

    assert forward["imf_count_range"] == reverse["imf_count_range"] == 1
    assert forward["energy_distribution_l1"] > 0.0
    assert reverse["energy_distribution_l1"] > 0.0
    assert forward["pairwise_comparisons"][0]["second_imf_count"] == 3
    assert reverse["pairwise_comparisons"][0]["first_imf_count"] == 3


def test_legacy_composite_cutoff_score_is_finite() -> None:
    physical = _modes([120.0, 40.0], samples=128)
    residual = np.zeros(128)
    components = np.vstack((physical, residual))
    score = decomposition.calculate_composite_cutoff_score(
        components, np.sum(physical, axis=0), 512.0
    )
    assert np.isfinite(score)
    assert score >= 0.0
