"""Mathematical validation of EMD decompositions and reconstructions.

Implements the metric suite from the research validation plan (Research Goal
2): reconstruction error (NRMSE), orthogonality index (OI), mode-mixing index
(MMI), per-IMF energy distribution, and a frequency-ordering score. All
functions are pure (numpy only) and operate on the IMF matrix ``imfs`` of
shape ``(n_imfs, n_samples)``.
"""

from typing import Dict

import numpy as np


def reconstruction_nrmse(original: np.ndarray, imfs: np.ndarray) -> float:
    """Normalised root-mean-square reconstruction error.

    NRMSE = ||original - sum_i imfs_i||_2 / ||original||_2.

    A value near 0 indicates a faithful reconstruction.
    """
    original = np.asarray(original, dtype=float)
    reconstructed = np.sum(imfs, axis=0)
    denom = np.sqrt(np.mean(original ** 2))
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)) / denom)


def orthogonality_index(imfs: np.ndarray) -> float:
    """Orthogonality Index (OI) of an IMF set.

    OI = 2 * sum_{i<j} <imf_i, imf_j> / sum_k ||imf_k||^2.

    OI = 0 means the IMFs are perfectly orthogonal.
    """
    imfs = np.asarray(imfs, dtype=float)
    cross = 0.0
    for i in range(imfs.shape[0]):
        for j in range(i + 1, imfs.shape[0]):
            cross += float(np.sum(imfs[i] * imfs[j]))
    total_energy = float(np.sum(imfs ** 2))
    if total_energy == 0:
        return 0.0
    return 2.0 * cross / total_energy


def mode_mixing_index(imfs: np.ndarray) -> float:
    """Mode-Mixing Index (MMI): mean absolute adjacent-IMF correlation.

    Lower is better; high adjacent correlation indicates mode mixing.
    Returns 0.0 when fewer than two IMFs are present.
    """
    imfs = np.asarray(imfs, dtype=float)
    n = imfs.shape[0]
    if n < 2:
        return 0.0
    corrs = []
    for i in range(n - 1):
        a = imfs[i] - imfs[i].mean()
        b = imfs[i + 1] - imfs[i + 1].mean()
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        corrs.append(0.0 if denom == 0 else abs(float(np.sum(a * b) / denom)))
    return float(np.mean(corrs))


def energy_distribution(imfs: np.ndarray) -> np.ndarray:
    """Per-IMF energy as a percentage of total IMF energy."""
    imfs = np.asarray(imfs, dtype=float)
    energies = np.sum(imfs ** 2, axis=1)
    total = float(np.sum(energies))
    if total == 0:
        return np.zeros(imfs.shape[0])
    return energies / total * 100.0


def frequency_ordering_index(imfs: np.ndarray, fs: float) -> float:
    """Frequency-ordering score in [0, 1].

    1.0 means IMF mean frequencies are strictly decreasing with index (the
    ideal EMD ordering); 0.0 means fully inverted. Uses the Pearson
    correlation between IMF index and mean frequency, mapped from [-1, 1] to
    [0, 1].
    """
    imfs = np.asarray(imfs, dtype=float)
    n = imfs.shape[0]
    if n < 2:
        return 1.0
    n_samples = imfs.shape[1]
    freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)
    pos = freqs >= 0
    mean_freqs = []
    for i in range(n):
        spec = np.abs(np.fft.fft(imfs[i]))[pos]
        s = float(spec.sum())
        mean_freqs.append(float(np.sum(freqs[pos] * spec) / s) if s > 0 else 0.0)
    mean_freqs = np.asarray(mean_freqs)
    idx = np.arange(n)
    if np.std(mean_freqs) == 0 or np.std(idx) == 0:
        return 1.0 if np.all(np.diff(mean_freqs) <= 0) else 0.0
    corr = float(np.corrcoef(idx, mean_freqs)[0, 1])
    return (corr + 1.0) / 2.0


def validate_decomposition(original: np.ndarray, imfs: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute the full validation metric bundle for one decomposition."""
    return {
        "nrmse": reconstruction_nrmse(original, imfs),
        "orthogonality_index": orthogonality_index(imfs),
        "mode_mixing_index": mode_mixing_index(imfs),
        "frequency_ordering_index": frequency_ordering_index(imfs, fs),
    }
