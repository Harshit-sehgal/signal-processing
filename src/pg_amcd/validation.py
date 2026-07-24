"""Mathematical validation of EMD decompositions and reconstructions.

Implements the metric suite from the research validation plan (Research Goal
2): reconstruction error (NRMSE), orthogonality index (OI), mode-mixing index
(MMI), per-IMF energy distribution, and a frequency-ordering score. All
functions are pure (numpy only) and operate on the IMF matrix ``imfs`` of
shape ``(n_imfs, n_samples)``.
"""

from typing import Any, Dict, List

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
    mean_frequency_values: list[float] = []
    for i in range(n):
        spec = np.abs(np.fft.fft(imfs[i]))[pos]
        s = float(spec.sum())
        mean_frequency_values.append(
            float(np.sum(freqs[pos] * spec) / s) if s > 0 else 0.0
        )
    mean_freqs = np.asarray(mean_frequency_values)
    idx = np.arange(n)
    if np.std(mean_freqs) == 0 or np.std(idx) == 0:
        return 1.0 if np.all(np.diff(mean_freqs) <= 0) else 0.0
    corr = float(np.corrcoef(idx, mean_freqs)[0, 1])
    return (1.0 - corr) / 2.0


def validate_decomposition(original: np.ndarray, imfs: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute the full validation metric bundle for one decomposition."""
    return {
        "nrmse": reconstruction_nrmse(original, imfs),
        "orthogonality_index": orthogonality_index(imfs),
        "mode_mixing_index": mode_mixing_index(imfs),
        "frequency_ordering_index": frequency_ordering_index(imfs, fs),
    }


def split_dataset_by_group(
    index_table: List[Dict[str, Any]],
    group_key: str = "recording_id",
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split a dataset index into train/val/test groups without leakage.

    Splits are performed at the group level (e.g., ``recording_id``) so that
    all windows belonging to one recording stay in one split.  Class labels are
    read from each group's first row and used to balance the split.

    Parameters
    ----------
    index_table: list of row dictionaries. Must contain ``group_key`` and a
        ``label`` field on every row.
    group_key: column used to define a recording/group.
    test_frac: fraction of groups to reserve for testing.
    val_frac: fraction of groups to reserve for validation.
    random_state: seed for the random permutation.

    Returns
    -------
    dict with keys ``train``, ``val``, ``test``; each maps to a list of rows.
    Note that ``val`` or ``test`` may be empty when a class has too few
    groups to satisfy the requested fractions while preserving at least one
    group in ``train``.
    """
    if not index_table:
        raise ValueError("index_table must contain at least one row.")
    if test_frac < 0 or val_frac < 0 or (test_frac + val_frac) >= 1.0:
        raise ValueError("test_frac and val_frac must be non-negative and sum to less than 1.")

    rows_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in index_table:
        group = str(row[group_key])
        rows_by_group.setdefault(group, []).append(row)

    group_ids = list(rows_by_group.keys())
    labels = {gid: rows_by_group[gid][0].get("label", "unknown") for gid in group_ids}

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(group_ids))
    shuffled = [group_ids[int(i)] for i in perm]

    class_to_groups: Dict[str, List[str]] = {}
    for gid in shuffled:
        class_to_groups.setdefault(labels[gid], []).append(gid)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    for gids in class_to_groups.values():
        n = len(gids)
        n_test = max(0, int(round(n * test_frac)))
        n_val = max(0, int(round(n * val_frac)))
        # Ensure at least one group remains in train.
        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        test_ids.extend(gids[:n_test])
        val_ids.extend(gids[n_test : n_test + n_val])
        train_ids.extend(gids[n_test + n_val :])

    def _collect(gids: List[str]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for gid in gids:
            result.extend(rows_by_group[gid])
        return result

    return {
        "train": _collect(train_ids),
        "val": _collect(val_ids),
        "test": _collect(test_ids),
    }
