"""Generate diagnostic figures for the PG-AMCD paper (Segment 7 figure set).

Builds the core signal-processing diagnostics directly from real
``Vibration_Clean`` recordings through the full pipeline: raw vs denoised
signal, STFT time-frequency, CEEMDAN IMF modes, chatter-vs-stable feature
distributions, and the corpus-wide per-window detection probability with its
temporally smoothed version. Each figure is generated defensively (one failure
never aborts the rest). Degrades gracefully if a sample file is missing.
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
VIB = os.path.join(ROOT, "Vibration_Clean")


def _load_mat(path):
    d = loadmat(path)
    ts = d["tsDS"]
    sig = ts[:, 1].astype(float)
    t = ts[:, 0].astype(float)
    if t[-1] <= t[0]:
        t = np.arange(len(sig)) / 10_000.0
    return t, sig


def _process(path, cfg):
    from pg_amcd.pipeline import process_recording
    t, sig = _load_mat(path)
    res = process_recording(t, sig, cfg, metadata={"rpm": 600, "tooth_count": 1},
                            mode="exploratory")
    return res


def _sample(prefix, stickout, fname):
    p = os.path.join(VIB, stickout, fname)
    return p if os.path.exists(p) else None


def fig_signal_denoised(cfg, fig_dir):
    samples = [
        ("chatter", _sample("c", "2p5inch_stickout", "c_570_014.mat")),
        ("stable", _sample("s", "2p5inch_stickout", "s_570_015.mat")),
    ]
    samples = [(lab, p) for lab, p in samples if p]
    if len(samples) < 2:
        return None
    fig, axes = plt.subplots(len(samples), 1, figsize=(9, 5), sharex=True)
    for ax, (lab, p) in zip(axes, samples):
        res = _process(p, cfg)
        wr = res.window_results[0]
        raw = res.raw_signal[wr.start_idx:wr.end_idx]
        den = wr.denoised_clean
        tt = wr.time_segment - wr.time_segment[0]
        ax.plot(tt, raw, color="gray", alpha=0.6, label="raw")
        ax.plot(tt, den, color="tab:blue", lw=1.2, label="PG-AMCD denoised")
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Raw vs denoised vibration signal")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_signal_denoised.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_stft(cfg, fig_dir):
    samples = [
        ("chatter", _sample("c", "3p5inch_stickout", "c_770_015.mat")),
        ("stable", _sample("s", "3p5inch_stickout", "s_770_010.mat")),
    ]
    samples = [(lab, p) for lab, p in samples if p]
    if len(samples) < 2:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, (lab, p) in zip(axes, samples):
        t, sig = _load_mat(p)
        f, ts, Z = stft(sig, fs=10_000.0, nperseg=256, noverlap=192)
        ax.pcolormesh(ts, f[:80], np.abs(Z[:80, :]), shading="auto", cmap="viridis")
        ax.set_ylabel(f"{lab}\n freq (Hz)", fontsize=8)
        ax.set_ylim(0, 4000)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("STFT magnitude spectrogram (chatter vs stable)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_stft.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_imfs(cfg, fig_dir):
    p = _sample("c", "2p5inch_stickout", "c_570_014.mat")
    if not p:
        return None
    res = _process(p, cfg)
    imfs = res.window_results[0].imfs
    n = min(5, imfs.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(9, 6), sharex=True)
    for i in range(n):
        axes[i].plot(imfs[i], lw=0.8)
        axes[i].set_ylabel(f"IMF {i+1}", fontsize=8)
    axes[-1].set_xlabel("sample")
    fig.suptitle("CEEMDAN intrinsic mode functions (chatter recording)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_imfs.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_feature_distributions(cfg, fig_dir):
    import json
    tpath = os.path.join(ROOT, "outputs", "temporal_results.json")
    if not os.path.exists(tpath):
        return None
    with open(tpath, "r", encoding="utf-8") as fh:
        tdata = json.load(fh)
    imp = tdata.get("feature_importances", {})
    if not imp:
        return None
    top = [k for k, _ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:6]]

    # Accumulate per-window features for a sample of chatter/stable recordings.
    chatter, stable = [], []
    for f in glob.glob(os.path.join(VIB, "**", "c_*.mat"), recursive=True)[:12]:
        try:
            res = _process(f, cfg)
            chatter.append([res.window_results[0].features[k] for k in top])
        except Exception:
            continue
    for f in glob.glob(os.path.join(VIB, "**", "s_*.mat"), recursive=True)[:12]:
        try:
            res = _process(f, cfg)
            stable.append([res.window_results[0].features[k] for k in top])
        except Exception:
            continue
    if not chatter or not stable:
        return None
    chatter = np.asarray(chatter, dtype=float)
    stable = np.asarray(stable, dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for j, k in enumerate(top):
        ax = axes[j // 3, j % 3]
        ax.boxplot([chatter[:, j], stable[:, j]], labels=["chatter", "stable"],
                   showmeans=True)
        ax.set_title(k, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Top-6 feature distributions (chatter vs stable)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_feature_distributions.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_detection_timeline(fig_dir):
    import json
    tpath = os.path.join(ROOT, "outputs", "temporal_results.json")
    if not os.path.exists(tpath):
        return None
    with open(tpath, "r", encoding="utf-8") as fh:
        tdata = json.load(fh)
    oof = tdata.get("oof", {})
    y_true = np.asarray(oof.get("y_true", []), dtype=int)
    best = tdata.get("best_model")
    proba = np.asarray(oof.get("proba", {}).get(best, []), dtype=float)
    smoothed = np.asarray(tdata.get("smoothed_proba", []), dtype=float)
    if len(y_true) == 0 or len(proba) == 0 or len(smoothed) == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(proba, color="gray", alpha=0.6, label="per-window proba")
    ax.plot(smoothed, color="tab:red", lw=1.3, label="temporally smoothed")
    ax.scatter(np.where(y_true == 1)[0], np.ones(int((y_true == 1).sum())) * 0.05,
               color="tab:orange", s=20, label="true chatter window", marker="|")
    ax.set_xlabel("window index (grouped-CV out-of-fold)")
    ax.set_ylabel(f"{best} chatter probability")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Corpus-wide detection probability: per-window vs smoothed")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_detection_timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_imf_weights(cfg, fig_dir):
    """IMF gating/weighting explanation: combined MAIW weights + indicators per IMF."""
    try:
        from pg_amcd.synthetic import generate_synthetic_signal
        from pg_amcd.decomposition import run_ceemdan
        from pg_amcd.weighting import calculate_maiw_weights
        fs = float(cfg.get("sampling_rate", 10000.0))
        t, sig, _ = generate_synthetic_signal(
            fs=fs, duration=0.4, seed=1, chatter_freq=1250.0, chatter_onset=0.5
        )
        imfs = run_ceemdan(sig, trials=2, epsilon=0.02, noise_seed=42, sifting_iterations=2)
        W, C, E, K, F = calculate_maiw_weights(imfs, sig, fs, cfg)
        n = W.shape[0]
        nC = C / (float(C.sum()) or 1.0)
        nE = E / (float(E.sum()) or 1.0)
        nK = K / (float(K.sum()) or 1.0)
        nF = F / (float(F.sum()) or 1.0)
        x = np.arange(n)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].bar(x, W, color="tab:blue")
        axes[0].set_title("Combined MAIW weight W per IMF")
        axes[0].set_xlabel("IMF index (1 = highest frequency)")
        axes[0].set_ylabel("weight")
        w = 0.2
        axes[1].bar(x - 1.5 * w, nC, w, label="corr")
        axes[1].bar(x - 0.5 * w, nE, w, label="energy")
        axes[1].bar(x + 0.5 * w, nK, w, label="kurtosis")
        axes[1].bar(x + 1.5 * w, nF, w, label="freq-prox")
        axes[1].set_title("Normalized MAIW indicators per IMF")
        axes[1].set_xlabel("IMF index")
        axes[1].legend(fontsize=8)
        fig.suptitle("How PG-AMCD weights/selects IMFs (chatter-band gating)")
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_imf_weights.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        print(f"imf_weights figure failed: {exc}")
        return None


def fig_runtime_comparison(cfg, fig_dir):
    """Runtime comparison of the major pipeline stages on one synthetic signal."""
    try:
        import time
        from pg_amcd.synthetic import generate_synthetic_signal
        from pg_amcd.decomposition import run_ceemdan
        from pg_amcd.preprocessing import preprocess_signal
        from pg_amcd.denoising import wavelet_denoise
        from pg_amcd.weighting import calculate_maiw_weights, reconstruct_weighted_signal
        from pg_amcd.pipeline import process_recording
        fs = float(cfg.get("sampling_rate", 10000.0))
        t, sig, _ = generate_synthetic_signal(
            fs=fs, duration=0.4, seed=2, chatter_freq=1250.0, chatter_onset=0.5
        )
        low, high = 50.0, min(4000.0, fs / 2.0 - 10.0)
        cc = cfg["maiw"]["chatter_band_center"]
        cs = cfg["maiw"]["chatter_band_spread"]
        reps = 3

        def _time(fn):
            best = float("inf")
            for _ in range(reps):
                s = time.perf_counter()
                r = fn()
                best = min(best, time.perf_counter() - s)
            return best, r

        d_ceem, imfs = _time(
            lambda: run_ceemdan(sig, trials=2, epsilon=0.02, noise_seed=42, sifting_iterations=2)
        )
        phys, _, _ = preprocess_signal(sig, low, high, fs)
        d_wav, _ = _time(
            lambda: wavelet_denoise(phys, wavelet_name="db8", level=4, fs=fs,
                                    chatter_center=cc, chatter_spread=cs)
        )

        def _maiw():
            W, *_ = calculate_maiw_weights(imfs, sig, fs, cfg)
            return reconstruct_weighted_signal(imfs, W)

        d_mai, _ = _time(_maiw)
        d_full, _ = _time(lambda: process_recording(t, sig, cfg, mode="exploratory"))
        stages = {
            "CEEMDAN\ndecompose": d_ceem,
            "Wavelet\ndenoise": d_wav,
            "MAIW\nweight": d_mai,
            "Full\npipeline": d_full,
        }
        names = list(stages.keys())
        vals = [stages[k] for k in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, vals, color="tab:green")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v * 1000:.0f} ms", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("mean wall-time (s)")
        ax.set_title("Pipeline stage runtime (synthetic 0.4 s signal, 3 reps)")
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_runtime_comparison.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        print(f"runtime figure failed: {exc}")
        return None


def fig_parameter_sensitivity(cfg, fig_dir):
    """Parameter sensitivity: full_proposed reconstruction RMSE vs wavelet level."""
    try:
        from pg_amcd.baselines import benchmark_denoising
        fs = float(cfg.get("sampling_rate", 10000.0))
        levels = list(range(1, 7))
        rmse = []
        for L in levels:
            agg = benchmark_denoising(n_signals=2, fs=fs, duration=0.4, seed=3,
                                      snr_db=20.0, wavelet_level=L)
            rmse.append(agg["full_proposed"]["rmse"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(levels, rmse, "o-", color="tab:purple")
        for L, r in zip(levels, rmse):
            ax.text(L, r, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("wavelet decomposition level")
        ax.set_ylabel("full_proposed RMSE (synthetic)")
        ax.set_title("Parameter sensitivity: denoising RMSE vs wavelet level")
        ax.set_xticks(levels)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_parameter_sensitivity.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        print(f"parameter sensitivity figure failed: {exc}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="PG-AMCD diagnostic figures")
    parser.add_argument(
        "--config", default=os.path.join(ROOT, "configs", "research_fast.json")
    )
    parser.add_argument("--figure-dir", default=FIG_DIR)
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from pg_amcd.config import load_pipeline_config
    cfg = load_pipeline_config(args.config)
    fig_dir = args.figure_dir
    os.makedirs(fig_dir, exist_ok=True)

    makers = [
        ("signal/denoised", lambda: fig_signal_denoised(cfg, fig_dir)),
        ("STFT", lambda: fig_stft(cfg, fig_dir)),
        ("IMFs", lambda: fig_imfs(cfg, fig_dir)),
        ("feature distributions", lambda: fig_feature_distributions(cfg, fig_dir)),
        ("detection timeline", lambda: fig_detection_timeline(fig_dir)),
        ("IMF weights", lambda: fig_imf_weights(cfg, fig_dir)),
        ("runtime comparison", lambda: fig_runtime_comparison(cfg, fig_dir)),
        ("parameter sensitivity", lambda: fig_parameter_sensitivity(cfg, fig_dir)),
    ]
    written = 0
    for name, fn in makers:
        try:
            p = fn()
            if p:
                print(f"Wrote {name} figure: {p}")
                written += 1
            else:
                print(f"Skipped {name} figure (missing inputs).")
        except Exception as exc:  # never abort the batch on one figure
            print(f"Failed {name} figure: {exc}")
    print(f"Generated {written}/{len(makers)} diagnostic figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
