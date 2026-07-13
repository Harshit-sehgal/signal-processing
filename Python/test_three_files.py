import os
import sys
import glob
import random
import numpy as np
import scipy.io
import scipy.signal
import scipy.fftpack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import EMD
from PyEMD import CEEMDAN

# 1. Helper function to load config
def load_pipeline_config():
    import json
    config_path = "/home/harshit/Documents/Research/Python/config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

# 2. Preprocess function
def preprocess_raw_signal(raw_vibration, low_cutoff=100.0, fs=10000.0):
    nyquist = 0.5 * fs
    high_cutoff = 4000.0
    b, a = scipy.signal.butter(3, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
    denoised = scipy.signal.filtfilt(b, a, raw_vibration)
    detrended = scipy.signal.detrend(denoised)
    max_val = np.max(np.abs(detrended))
    return detrended / (max_val if max_val > 0 else 1.0)

# 3. Select max energy segment
def select_max_energy_segment(signal, time, segment_points=10000):
    if len(signal) > segment_points:
        squared_signal = np.square(signal)
        window = np.ones(segment_points)
        rolling_energy = np.convolve(squared_signal, window, mode='valid')
        start_idx = np.argmax(rolling_energy)
        end_idx = start_idx + segment_points
        return signal[start_idx:end_idx], time[start_idx:end_idx], start_idx
    else:
        return signal, time, 0

# 4. Fast MMI evaluation
def evaluate_mode_mixing(s_seg, ceemdan_cfg):
    trials = ceemdan_cfg["search_trials"]
    epsilon = ceemdan_cfg["epsilon"]
    fxe = ceemdan_cfg.get("sifting_iterations", 0)
    
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, FIXE=fxe, parallel=True)
    ceemdan.noise_seed(ceemdan_cfg["noise_seed"])
    imfs = ceemdan(s_seg)
    num_imfs = imfs.shape[0]
    
    corrs = []
    for i in range(num_imfs - 2):
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        if not np.isnan(corr):
            corrs.append(corr)
    avg_corr = np.mean(corrs) if corrs else 1.0
    return avg_corr

# 5. EMD Diagnostics
def compute_diagnostics(original, imfs):
    num_imfs = imfs.shape[0]
    N = original.size
    
    # 1. Reconstruction Error
    reconstructed = np.sum(imfs, axis=0)
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    nrmse = rmse / np.sqrt(np.mean(original ** 2)) if np.any(original) else 0.0
    
    # 2. Adjacent correlation
    adj_corrs = []
    for i in range(num_imfs - 2):
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        adj_corrs.append(corr if not np.isnan(corr) else 0.0)
    mean_adj_corr = np.mean(adj_corrs) if adj_corrs else 0.0
    
    # 3. Frequency Spectrum
    fs = 10000.0
    frequencies = scipy.fftpack.fftfreq(N, 1/fs)
    idx = np.where(frequencies >= 0)
    frequencies = frequencies[idx]
    
    imf_mean_freqs = []
    for i in range(num_imfs - 1):
        fft_vals = np.abs(scipy.fftpack.fft(imfs[i]))[idx]
        psd = fft_vals ** 2
        psd_sum = np.sum(psd)
        mean_f = np.sum(frequencies * psd) / psd_sum if psd_sum > 0 else 0.0
        imf_mean_freqs.append(mean_f)
        
    # 4. Energy
    energies = np.sum(imfs ** 2, axis=1)
    total_energy = np.sum(energies)
    energy_percentages = (energies / total_energy) * 100.0 if total_energy > 0 else np.zeros(num_imfs)
    
    # 5. Orthogonality Index (OI)
    cross_terms = 0.0
    for i in range(num_imfs):
        for j in range(i + 1, num_imfs):
            cross_terms += np.sum(imfs[i] * imfs[j])
    orig_energy = np.sum(original ** 2)
    oi = 2.0 * cross_terms / orig_energy if orig_energy > 0 else 0.0
    
    return {
        'num_imfs': num_imfs,
        'nrmse': nrmse,
        'oi': oi,
        'mean_adj_corr': mean_adj_corr,
        'imf_mean_freqs': imf_mean_freqs,
        'energy_percentages': energy_percentages
    }

# 6. MAIW Reconstitution
def run_maiw(imfs, time, config):
    maiw_cfg = config["maiw"]
    num_imfs = imfs.shape[0]
    N = time.size
    
    # Exclude residue (last IMF) from weighting
    num_weighted = num_imfs - 1
    
    # Metrics
    C = np.zeros(num_weighted)
    E = np.zeros(num_weighted)
    K = np.zeros(num_weighted)
    F = np.zeros(num_weighted)
    
    fs = 10000.0
    chatter_center = maiw_cfg["chatter_band_center"]
    chatter_spread = maiw_cfg["chatter_band_spread"]
    
    # Overall signal energy
    sum_imfs = np.sum(imfs[:num_weighted], axis=0)
    sig_std = np.std(sum_imfs)
    
    for i in range(num_weighted):
        imf = imfs[i]
        
        # A. Correlation coefficient
        corr = np.abs(np.corrcoef(imf, sum_imfs)[0, 1])
        C[i] = corr if not np.isnan(corr) else 0.0
        
        # B. Energy percentage
        E[i] = np.sum(imf ** 2) / np.sum(sum_imfs ** 2)
        
        # C. Kurtosis
        # kurtosis = E[(x - mu)^4] / sigma^4 - 3 (or fisher=False)
        mu = np.mean(imf)
        std = np.std(imf)
        if std > 0:
            K[i] = np.mean((imf - mu) ** 4) / (std ** 4)
        else:
            K[i] = 3.0 # normal kurtosis
            
        # D. Frequency Proximity
        # Peak frequency of IMF
        fft_vals = np.abs(scipy.fftpack.fft(imf))
        freqs = scipy.fftpack.fftfreq(N, 1/fs)
        pos_freqs = freqs[freqs >= 0]
        pos_fft = fft_vals[freqs >= 0]
        peak_f = pos_freqs[np.argmax(pos_fft)]
        
        # Gaussian proximity score
        F[i] = np.exp(-((peak_f - chatter_center) ** 2) / (2 * (chatter_spread ** 2)))
        
    # Standardize indicators
    def norm_ind(I):
        s = np.sum(I)
        return I / s if s > 0 else np.ones_like(I) / len(I)
        
    nC = norm_ind(C)
    nE = norm_ind(E)
    nK = norm_ind(K)
    nF = norm_ind(F)
    
    # Weights
    W = maiw_cfg["alpha"]*nC + maiw_cfg["beta"]*nE + maiw_cfg["gamma"]*nK + maiw_cfg["delta"]*nF
    
    # Reconstructed signal
    reconstructed = np.zeros(N)
    for i in range(num_weighted):
        reconstructed += W[i] * imfs[i]
        
    # Add residue
    reconstructed += imfs[-1]
    
    return reconstructed, W, C, E, K, F

# 7. Wavelet Denoising
def run_wavelet_denoise(signal, time, config):
    import pywt
    wav_cfg = config["wavelet"]
    wavelet_name = wav_cfg["wavelet_name"]
    requested_level = wav_cfg["level"]
    
    # Calculate max possible level
    max_level = pywt.dwt_max_level(data_len=len(signal), filter_len=pywt.Wavelet(wavelet_name).dec_len)
    level = min(requested_level, max_level)
    
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # BayesShrink thresholding on detail coefficients
    for i in range(1, len(coeffs)):
        details = coeffs[i]
        # Estimating noise sigma from Median Absolute Deviation of highest level details
        if i == 1:
            sigma_w = np.median(np.abs(details)) / 0.6745
        
        # Signal variance estimate
        var_y = np.mean(details ** 2)
        var_x = max(0, var_y - (sigma_w ** 2))
        
        # BayesShrink threshold
        if var_x > 0:
            threshold = (sigma_w ** 2) / np.sqrt(var_x)
        else:
            threshold = np.max(np.abs(details)) # threshold everything
            
        coeffs[i] = pywt.threshold(details, threshold, mode='soft')
        
    # Reconstruct
    clean_signal = pywt.waverec(coeffs, wavelet_name)
    # Clip/truncate to exact original length
    clean_signal = clean_signal[:len(signal)]
    
    # Normalize clean signal
    max_val = np.max(np.abs(clean_signal))
    if max_val > 0:
        clean_signal = clean_signal / max_val
        
    return clean_signal

# 8. Main execution
def main():
    config = load_pipeline_config()
    ceemdan_cfg = config["ceemdan"]
    
    # Directory paths
    raw_dir = "/home/harshit/Documents/Research/Vibration - ML"
    testing_dir = "/home/harshit/Documents/Research/testing/t1"
    os.makedirs(testing_dir, exist_ok=True)
    
    # Find all mat files
    all_raw_files = glob.glob(os.path.join(raw_dir, "**/*.mat"), recursive=True)
    # Exclude doc file
    all_raw_files = [f for f in all_raw_files if not f.endswith("combinations.xlsx") and "~lock" not in f]
    
    # Seed random for reproducibility
    random.seed(999)
    selected_files = random.sample(all_raw_files, 3)
    
    print("=" * 65)
    print("🧪 HIGH-FIDELITY FINE-TUNING VALIDATION RUN (3 RANDOM FILES) 🧪")
    print("=" * 65)
    print(f"Selected files:")
    for f in selected_files:
        print(f"  - {os.path.relpath(f, raw_dir)}")
    print("-" * 65)
    
    diagnostics_summary = []
    
    for raw_path in selected_files:
        folder_name = os.path.basename(os.path.dirname(raw_path))
        base_name = os.path.basename(raw_path).replace(".mat", "")
        print(f"\nProcessing: {folder_name}/{base_name}.mat")
        
        # Load raw data
        raw_data = scipy.io.loadmat(raw_path)['tsDS']
        time_arr = raw_data[:, 0]
        raw_vibration = raw_data[:, 1]
        
        # 1. Parameter loop over cutoffs to find best preprocessing
        cutoffs = ceemdan_cfg["search_cutoffs"]
        best_score = float('inf')
        best_cutoff = 100
        
        for cut in cutoffs:
            normalized = preprocess_raw_signal(raw_vibration, cut)
            s_seg, _, _ = select_max_energy_segment(normalized, time_arr)
            score = evaluate_mode_mixing(s_seg, ceemdan_cfg)
            if score < best_score:
                best_score = score
                best_cutoff = cut
                
        print(f"  Optimal cutoff chosen: {best_cutoff} Hz (score: {best_score:.4f})")
        
        # 2. Final preprocessing
        opt_normalized = preprocess_raw_signal(raw_vibration, best_cutoff)
        opt_s_seg, opt_t_seg, start_idx = select_max_energy_segment(opt_normalized, time_arr)
        
        # 3. Final high-fidelity decomposition
        trials = ceemdan_cfg["trials"]
        epsilon = ceemdan_cfg["epsilon"]
        seed = ceemdan_cfg["noise_seed"]
        fxe = ceemdan_cfg.get("sifting_iterations", 0)
        
        print(f"  Running final CEEMDAN (trials={trials}, epsilon={epsilon}, FIXE={fxe})...")
        final_ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, FIXE=fxe, parallel=True)
        final_ceemdan.noise_seed(seed)
        imfs = final_ceemdan(opt_s_seg)
        
        # Save IMFs
        npz_path = os.path.join(testing_dir, f"{base_name}_IMFs.npz")
        np.savez_compressed(npz_path, time=opt_t_seg, original_signal=opt_s_seg, imfs=imfs)
        
        # Save EMD plot
        num_imfs = imfs.shape[0]
        plt.figure(figsize=(14, 2 * (num_imfs + 1)))
        plt.subplot(num_imfs + 1, 1, 1)
        plt.plot(opt_t_seg, opt_s_seg, color='black')
        plt.title(f"Fine-Tuned CEEMDAN Decomposition (epsilon={epsilon}, FIXE={fxe}) - {base_name}.mat")
        plt.ylabel("Input")
        for i in range(num_imfs):
            plt.subplot(num_imfs + 1, 1, i + 2)
            if i == num_imfs - 1:
                plt.plot(opt_t_seg, imfs[i], color='red')
                plt.ylabel("Residual")
            else:
                plt.plot(opt_t_seg, imfs[i], color='blue')
                plt.ylabel(f"IMF {i+1}")
        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plot_path = os.path.join(testing_dir, f"{base_name}_IMFs_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        # 4. MAIW Reconstruction
        reconstructed, W, C, E, K, F = run_maiw(imfs, opt_t_seg, config)
        maiw_path = os.path.join(testing_dir, f"{base_name}_MAIW.mat")
        scipy.io.savemat(maiw_path, {
            'tsDS': np.column_stack((opt_t_seg, reconstructed)),
            'weights': W,
            'correlation': C,
            'energy': E,
            'kurtosis': K,
            'frequency_proximity': F
        })
        
        # 5. Wavelet Denoising
        clean_signal = run_wavelet_denoise(reconstructed, opt_t_seg, config)
        clean_path = os.path.join(testing_dir, f"{base_name}_Clean.mat")
        scipy.io.savemat(clean_path, {
            'tsDS': np.column_stack((opt_t_seg, clean_signal))
        })
        
        # 6. Pipeline comparison plot
        plt.figure(figsize=(14, 10))
        # Plot 0: Raw
        plt.subplot(4, 1, 1)
        # crop raw to match segment
        crop_raw = raw_vibration[start_idx:start_idx + 10000]
        plt.plot(opt_t_seg, crop_raw, color='grey', alpha=0.8)
        plt.title(f"Stage 0: Raw Vibration Signal - {base_name}.mat", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        # Plot 1: Preprocessed
        plt.subplot(4, 1, 2)
        plt.plot(opt_t_seg, opt_s_seg, color='orange')
        plt.title(f"Stage 1: Bandpass Preprocessed Signal (Cutoff = {best_cutoff} Hz)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        # Plot 2: MAIW
        plt.subplot(4, 1, 3)
        plt.plot(opt_t_seg, reconstructed, color='blue')
        plt.title("Stage 2: Multi-Criteria Adaptive IMF Weighting (Reconstructed)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        # Plot 3: Clean
        plt.subplot(4, 1, 4)
        plt.plot(opt_t_seg, clean_signal, color='green')
        plt.title("Stage 3: Bayesian Adaptive Wavelet Denoised (Final Clean Output)", fontsize=12, fontweight='bold')
        plt.xlabel("Time (seconds)")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(testing_dir, f"{base_name}_pipeline_comparison.png")
        plt.savefig(comparison_plot_path, dpi=150)
        plt.close()
        
        # 7. Diagnostics calculation
        diag = compute_diagnostics(opt_s_seg, imfs)
        diagnostics_summary.append({
            'folder': folder_name,
            'file': base_name,
            'cutoff': best_cutoff,
            'diag': diag
        })
        print(f"  Diagnostics calculated: NRMSE={diag['nrmse']:.2e}, MMI={diag['mean_adj_corr']:.4f}, OI={diag['oi']:.4f}")

    # Generate Markdown Summary Report
    md_content = """# High-Fidelity Validation Run: Testing Summary (T1)

This report presents EMD diagnostics for three randomly selected data files processed with the fine-tuned parameters (`epsilon=0.02`, `sifting_iterations=16`).

---

## 📊 Summary Comparison Metrics

| File Path | Cutoff Frequency | Number of IMFs | Reconstruction Error (NRMSE) | Overall Orthogonality (OI) | Mean Adjacent IMF Corr (MMI) |
|---|---|---|---|---|---|
"""
    for item in diagnostics_summary:
        f_path = f"{item['folder']}/{item['file']}.mat"
        diag = item['diag']
        md_content += f"| **{f_path}** | {item['cutoff']} Hz | {diag['num_imfs']} | {diag['nrmse']:.2e} | {diag['oi']:.4f} | {diag['mean_adj_corr']:.4f} |\n"
        
    md_content += """
---

## ⚡ Spectral Analysis (Mean Frequencies of IMF Layers)
Below are the spectral centroid frequencies (in Hz) for the physical IMFs of each test file:

| Layer | """ + " | ".join([f"{item['file']}.mat" for item in diagnostics_summary]) + " |\n"
    md_content += "|---| " + " | ".join(["---" for _ in diagnostics_summary]) + " |\n"
    
    max_imfs_all = max([item['diag']['num_imfs'] for item in diagnostics_summary])
    for i in range(max_imfs_all - 1):
        row = f"| **IMF {i+1}** | "
        cols = []
        for item in diagnostics_summary:
            diag = item['diag']
            if i < diag['num_imfs'] - 1:
                cols.append(f"{diag['imf_mean_freqs'][i]:.1f} Hz")
            else:
                cols.append("N/A")
        row += " | ".join(cols) + " |\n"
        md_content += row
    # Add Residual row
    row = "| **Residual** | "
    row += " | ".join(["Residual" for _ in diagnostics_summary]) + " |\n"
    md_content += row

    md_report_path = os.path.join(testing_dir, "diagnostics_summary.md")
    with open(md_report_path, "w") as fd:
        fd.write(md_content)
    print(f"\nSaved diagnostics summary to: {md_report_path}")

if __name__ == "__main__":
    main()
