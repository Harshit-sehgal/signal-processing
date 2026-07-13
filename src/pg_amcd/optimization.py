"""Adaptive cutoff selection and multi-seed stability (Sprint 4 / Segment 5).

Implements Goal 5.1 (controlled cutoff optimisation over the *same* raw
segment), Goal 5.2 (five-component cutoff objective) and Goal 5.3 (multi-seed
stability reporting).
"""

from typing import Dict, List, Any, Optional, Sequence

import numpy as np

from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.decomposition import (
    run_ceemdan,
    calculate_adjacent_imf_correlation,
    calculate_spectral_overlap,
    calculate_orthogonality_index,
)
from pg_amcd.models import CutoffOptimizationResult


def _chatter_band_distortion(
    imfs: np.ndarray, original: np.ndarray, fs: float, center: float, spread: float
) -> float:
    """Fraction of chatter-band energy lost in the reconstruction (0 = none lost)."""
    reconstructed = np.sum(imfs, axis=0)
    n = len(original)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    pos = freqs >= 0
    spec_orig = np.abs(np.fft.fft(original))[pos] ** 2
    spec_recon = np.abs(np.fft.fft(reconstructed))[pos] ** 2
    mask = (freqs[pos] >= center - spread) & (freqs[pos] <= center + spread)
    e_orig = float(np.sum(spec_orig[mask]))
    e_recon = float(np.sum(spec_recon[mask]))
    if e_orig <= 0:
        return 0.0
    return float(max(0.0, 1.0 - e_recon / e_orig))


def compute_cutoff_objective(
    imfs: np.ndarray,
    original: np.ndarray,
    fs: float,
    config: Dict[str, Any],
    seed_instability: float = 0.0,
) -> float:
    """Five-component cutoff objective (Goal 5.2).

    Lower is better: this is a *badness* score combining spectral overlap,
    max adjacent-IMF correlation, |orthogonality index|, seed instability and
    chatter-band distortion. Normalised so each term lives in ``[0, 1]``.
    """
    maiw = config.get("maiw", {})
    center = maiw.get("chatter_band_center", 1250.0)
    spread = maiw.get("chatter_band_spread", 500.0)

    _, max_adj = calculate_adjacent_imf_correlation(imfs)
    spectral_overlap = calculate_spectral_overlap(imfs, fs)
    oi = calculate_orthogonality_index(imfs, original)
    cbd = _chatter_band_distortion(imfs, original, fs, center, spread)

    return (
        0.25 * min(1.0, spectral_overlap)
        + 0.20 * min(1.0, max_adj)
        + 0.20 * min(1.0, abs(oi))
        + 0.20 * min(1.0, seed_instability)
        + 0.15 * min(1.0, cbd)
    )


def optimize_cutoff(
    raw_segment: np.ndarray,
    candidate_cutoffs: Sequence[float],
    config: Dict[str, Any],
    fs: float,
    n_seeds: int = 2,
    seeds: Optional[Sequence[int]] = None,
) -> CutoffOptimizationResult:
    """Select the best preprocessing cutoff by optimising the same raw segment.

    Every candidate cutoff processes the *identical* ``raw_segment`` (Goal 5.1:
    never pick a different max-energy window per cutoff). The cutoff with the
    lowest :func:`compute_cutoff_objective` is selected.
    """
    if not candidate_cutoffs:
        raise ValueError("candidate_cutoffs must be a non-empty sequence.")
    ceemdan_cfg = config.get("ceemdan", {})
    if seeds is None:
        base = ceemdan_cfg.get("noise_seed", 42)
        seeds = [base + k for k in range(max(1, n_seeds))]
    search_trials = ceemdan_cfg.get("search_trials", ceemdan_cfg.get("trials", 50))
    search_sift = ceemdan_cfg.get("search_sifting_iterations", ceemdan_cfg.get("sifting_iterations", 16))
    epsilon = ceemdan_cfg.get("epsilon", 0.02)
    high_cutoff = min(4000.0, fs / 2.0 - 10.0)

    per_cutoff: List[Dict[str, Any]] = []
    for cut in candidate_cutoffs:
        objectives = []
        recon_nrmse = []
        for sd in seeds:
            _, scaled, _ = preprocess_signal(raw_segment, float(cut), high_cutoff, fs)
            imfs = run_ceemdan(scaled, search_trials, epsilon, int(sd), search_sift)
            objectives.append(compute_cutoff_objective(imfs, scaled, fs, config))
            reconstructed = np.sum(imfs, axis=0)
            denom = np.sqrt(np.mean(scaled ** 2))
            recon_nrmse.append(
                float(np.sqrt(np.mean((scaled - reconstructed) ** 2)) / denom)
                if denom > 0
                else 0.0
            )
        seed_instability = float(np.std(recon_nrmse)) if len(recon_nrmse) > 1 else 0.0
        mean_obj = float(np.mean(objectives))
        final = mean_obj + 0.20 * seed_instability
        per_cutoff.append(
            {
                "cutoff": float(cut),
                "objective": mean_obj,
                "seed_instability": seed_instability,
                "final_score": final,
                "mean_reconstruction_nrmse": float(np.mean(recon_nrmse)),
            }
        )

    best = min(per_cutoff, key=lambda d: d["final_score"])
    return CutoffOptimizationResult(
        selected_cutoff=best["cutoff"],
        per_cutoff_metrics=per_cutoff,
        best_score=best["final_score"],
    )


def multi_seed_stability(func, seeds: Sequence[int]) -> Dict[str, float]:
    """Run ``func(seed)`` across seeds and return summary statistics (Goal 5.3)."""
    values = np.array([float(func(int(sd))) for sd in seeds])
    mean = float(np.mean(values))
    std = float(np.std(values))
    n = len(values)
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": n,
    }
