"""Controlled synthetic machining-vibration data (Sprint 4 / Goal 5.4).

Generates signals with known components so that denoising / detection quality
can be measured against ground truth (chatter reconstruction RMSE, SNR
improvement, spectral distortion, chatter-band retention, noise-band
attenuation, onset-detection error).
"""

from typing import Dict

import numpy as np


def generate_synthetic_signal(
    fs: float = 10_000.0,
    duration: float = 1.0,
    seed: int = 0,
    rpm: float = 600.0,
    tooth_count: int = 1,
    chatter_freq: float = 1250.0,
    chatter_onset: float = 0.5,
    snr_db: float = 20.0,
) -> tuple:
    """Generate a synthetic vibration signal with known ground-truth parts.

    Returns ``(t, signal, components)`` where ``components`` holds the
    individual ``forced``, ``chatter``, ``drift``, ``noise`` and ``clean``
    (forced + chatter + drift) arrays.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs

    f_tp = (rpm / 60.0) * tooth_count
    x_forced = 0.6 * np.sin(2 * np.pi * f_tp * t) + 0.3 * np.sin(2 * np.pi * 2 * f_tp * t)

    envelope = 1.0 / (1.0 + np.exp(-15.0 * (t - chatter_onset)))
    x_chatter = envelope * np.sin(2 * np.pi * chatter_freq * t)

    x_drift = 0.5 * (t ** 2)

    x_clean = x_forced + x_chatter + x_drift
    signal_power = float(np.var(x_clean))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    x_noise = rng.normal(0.0, np.sqrt(noise_power), n)

    signal = x_clean + x_noise
    components = {
        "forced": x_forced,
        "chatter": x_chatter,
        "drift": x_drift,
        "noise": x_noise,
        "clean": x_clean,
    }
    return t, signal, components


def evaluate_denoising_performance(
    clean_reference,
    denoised,
    fs: float,
    chatter_center: float,
    chatter_spread: float,
    chatter_onset: float = 0.5,
) -> Dict[str, float]:
    """Quantitative denoising-quality metrics against a known clean reference."""
    clean = np.asarray(clean_reference, dtype=float)
    den = np.asarray(denoised, dtype=float)
    if clean.shape != den.shape:
        m = min(clean.size, den.size)
        clean = clean[:m]
        den = den[:m]

    # Align amplitudes to remove scaling bias from RMSE and SNR
    den_energy = float(np.sum(den ** 2))
    if den_energy > 0:
        alpha = float(np.sum(clean * den) / den_energy)
        den = den * alpha

    rmse = float(np.sqrt(np.mean((clean - den) ** 2)))

    residual_power = float(np.var(clean - den))
    signal_power = float(np.var(clean))
    snr_db = 10.0 * np.log10(signal_power / residual_power) if residual_power > 0 else float("inf")

    freqs = np.fft.fftfreq(clean.size, d=1.0 / fs)
    pos = freqs >= 0
    spec_clean = np.abs(np.fft.fft(clean))[pos]
    spec_den = np.abs(np.fft.fft(den))[pos]
    sc = spec_clean / (np.sum(spec_clean) + 1e-12)
    sd = spec_den / (np.sum(spec_den) + 1e-12)
    spectral_distortion = float(np.sum(np.abs(sc - sd)))

    mask = (freqs[pos] >= chatter_center - chatter_spread) & (freqs[pos] <= chatter_center + chatter_spread)
    cb_clean = float(np.sum(sc[mask]))
    cb_den = float(np.sum(sd[mask]))
    chatter_band_retention = float(cb_den / cb_clean) if cb_clean > 0 else 0.0

    out_mask = ~mask
    nb_clean = float(np.sum(sc[out_mask]))
    nb_den = float(np.sum(sd[out_mask]))
    noise_band_attenuation = float(1.0 - nb_den / nb_clean) if nb_clean > 0 else 0.0

    # Best-effort onset detection from the denoised energy envelope.
    env = np.abs(den - np.mean(den))
    if env.size > 50:
        kernel = np.ones(50) / 50.0
        env = np.convolve(env, kernel, mode="same")
    lo, hi = float(env.min()), float(env.max())
    thresh = 0.5 * (lo + hi)
    crossings = np.where(np.diff(env > thresh))[0]
    est_onset = float(crossings[0] / fs) if crossings.size > 0 else float("nan")
    onset_detection_error = (
        abs(est_onset - chatter_onset) if not np.isnan(est_onset) else float("nan")
    )

    return {
        "rmse": rmse,
        "snr_db": snr_db,
        "spectral_distortion": spectral_distortion,
        "chatter_band_retention": chatter_band_retention,
        "noise_band_attenuation": noise_band_attenuation,
        "onset_detection_error": onset_detection_error,
    }
