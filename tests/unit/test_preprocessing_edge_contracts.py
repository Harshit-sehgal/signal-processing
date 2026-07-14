"""Negative and edge contracts for amplitude-traceable Stage 1 preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

import pg_amcd.preprocessing as preprocessing


def _signal(samples: int = 256, fs: float = 1000.0) -> np.ndarray:
    time = np.arange(samples) / fs
    return np.sin(2.0 * np.pi * 80.0 * time) + 0.2 * np.sin(2.0 * np.pi * 220.0 * time)


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        (np.ones(8, dtype=complex) * (1.0 + 1.0j), "real-valued"),
        (np.ones((2, 8)), "one-dimensional"),
        (np.asarray([]), "at least one sample"),
        (np.asarray([0.0, np.nan, 1.0]), "NaN or infinite"),
        (np.asarray([0.0, np.inf, 1.0]), "NaN or infinite"),
    ],
)
def test_signal_validation_rejects_nonphysical_arrays(signal: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preprocessing.butter_bandpass_filter_sos(signal, 20.0, 400.0, 1000.0)


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, np.nan, np.inf])
def test_filter_rejects_invalid_sampling_rate(sampling_rate: float) -> None:
    with pytest.raises(ValueError, match="Sampling rate must be a finite positive"):
        preprocessing.butter_bandpass_filter_sos(_signal(), 20.0, 400.0, sampling_rate)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (np.nan, 400.0, "cutoffs must be finite"),
        (20.0, np.inf, "cutoffs must be finite"),
        (20.0, 500.0, "Invalid band-pass bounds"),
        (400.0, 400.0, "Invalid band-pass bounds"),
    ],
)
def test_filter_rejects_invalid_band_edges(lower: float, upper: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preprocessing.butter_bandpass_filter_sos(_signal(), lower, upper, 1000.0)


@pytest.mark.parametrize("order", [True, "three", 2.5, np.nan, 0])
def test_filter_order_must_be_a_positive_integer(order: object) -> None:
    with pytest.raises(ValueError, match="Filter order must be a positive integer"):
        preprocessing.butter_bandpass_filter_sos(
            _signal(),
            20.0,
            400.0,
            1000.0,
            order=order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("padlen", [True, "many", -1, 2.5, np.nan])
def test_explicit_pad_length_must_be_a_nonnegative_integer(padlen: object) -> None:
    with pytest.raises(ValueError, match="padlen must be a non-negative integer"):
        preprocessing.butter_bandpass_filter_sos(
            _signal(),
            20.0,
            400.0,
            1000.0,
            padlen=padlen,  # type: ignore[arg-type]
        )


def test_explicit_pad_length_must_be_shorter_than_signal() -> None:
    with pytest.raises(ValueError, match="must be greater than configured padlen"):
        preprocessing.butter_bandpass_filter_sos(_signal(32), 20.0, 400.0, 1000.0, padlen=32)


def test_filter_rejects_nonfinite_scipy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preprocessing.scipy.signal,
        "sosfiltfilt",
        lambda *_args, **_kwargs: np.full(256, np.nan),
    )

    with pytest.raises(ValueError, match="SOS filtering produced non-finite"):
        preprocessing.butter_bandpass_filter_sos(_signal(), 20.0, 400.0, 1000.0)


@pytest.mark.parametrize("detrend_type", ["quadratic", "none", ""])
def test_preprocessing_rejects_unsupported_detrending(detrend_type: str) -> None:
    with pytest.raises(ValueError, match="detrend_type must be"):
        preprocessing.preprocess_signal_result(
            _signal(),
            20.0,
            400.0,
            1000.0,
            detrend_type=detrend_type,
        )


@pytest.mark.parametrize("percentile", [0.0, -1.0, 100.1, np.nan, np.inf])
def test_preprocessing_rejects_invalid_scale_percentiles(percentile: float) -> None:
    with pytest.raises(ValueError, match="scale_percentile must be in"):
        preprocessing.preprocess_signal_result(
            _signal(),
            20.0,
            400.0,
            1000.0,
            scale_percentile=percentile,
        )


def test_preprocessing_rejects_a_numerically_flat_filtered_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preprocessing,
        "butter_bandpass_filter_sos",
        lambda signal, *_args, **_kwargs: np.zeros_like(signal, dtype=float),
    )

    with pytest.raises(ValueError, match="no numerically meaningful amplitude"):
        preprocessing.preprocess_signal_result(
            _signal(), 20.0, 400.0, 1000.0, detrend_type="constant"
        )


def test_preprocessing_parameter_aliases_and_explicit_padding_are_traceable() -> None:
    result = preprocessing.preprocess_signal_result(
        _signal(),
        20.0,
        400.0,
        1000.0,
        order=2,
        detrend_type="constant",
        padtype=None,
        padlen=0,
    )
    parameters = result.parameters.as_dict()

    assert result.parameters.high_pass_cutoff_hz == 20.0
    assert result.parameters.low_pass_cutoff_hz == 400.0
    assert parameters["high_pass_cutoff_hz"] == 20.0
    assert parameters["low_pass_cutoff_hz"] == 400.0
    assert parameters["padtype"] is None
    assert parameters["padlen"] == 0
    np.testing.assert_allclose(
        result.restore_physical(result.scaled_signal), result.physical_signal
    )
