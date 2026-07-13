import os
import sys
import numpy as np
import scipy.io
import scipy.fftpack

def compute_diagnostics(npz_path):
    if not os.path.exists(npz_path):
        return None
        
    data = np.load(npz_path)
    original = data['original_signal']
    imfs = data['imfs'] # shape (num_imfs, N)
    
    num_imfs = imfs.shape[0]
    N = original.size
    
    # 1. Reconstruction Error
    # Reconstructed signal is sum of all IMFs (the last one is residual in PyEMD)
    reconstructed = np.sum(imfs, axis=0)
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    nrmse = rmse / np.sqrt(np.mean(original ** 2)) if np.any(original) else 0.0
    
    # 2. Correlation of neighbouring IMFs
    adj_corrs = []
    for i in range(num_imfs - 2): # exclude residual layer in adjacent correlations
        corr = np.abs(np.corrcoef(imfs[i], imfs[i+1])[0, 1])
        adj_corrs.append(corr if not np.isnan(corr) else 0.0)
    mean_adj_corr = np.mean(adj_corrs) if adj_corrs else 0.0
    
    # 3. Frequency Centroid (Mean Frequency) and Peak Frequency of each IMF
    fs = 10000.0 # 10 kHz sampling rate
    frequencies = scipy.fftpack.fftfreq(N, 1/fs)
    idx = np.where(frequencies >= 0)
    frequencies = frequencies[idx]
    
    imf_mean_freqs = []
    imf_peak_freqs = []
    for i in range(num_imfs - 1): # exclude residual
        fft_vals = np.abs(scipy.fftpack.fft(imfs[i]))[idx]
        psd = fft_vals ** 2
        psd_sum = np.sum(psd)
        
        # Mean frequency (spectral centroid)
        mean_f = np.sum(frequencies * psd) / psd_sum if psd_sum > 0 else 0.0
        imf_mean_freqs.append(mean_f)
        
        # Peak frequency
        peak_f = frequencies[np.argmax(psd)]
        imf_peak_freqs.append(peak_f)
        
    # 4. Energy Contribution
    energies = np.sum(imfs ** 2, axis=1)
    total_energy = np.sum(energies)
    energy_percentages = (energies / total_energy) * 100.0 if total_energy > 0 else np.zeros(num_imfs)
    
    # 5. Orthogonality Index (OI)
    # Standard overall index of orthogonality (IO)
    # IO = sum_{t} (sum_{i!=j} IMF_i(t)*IMF_j(t)) / sum_{t} x(t)^2
    cross_terms = 0.0
    for i in range(num_imfs):
        for j in range(i + 1, num_imfs):
            cross_terms += np.sum(imfs[i] * imfs[j])
    orig_energy = np.sum(original ** 2)
    oi = 2.0 * cross_terms / orig_energy if orig_energy > 0 else 0.0
    
    return {
        'num_imfs': num_imfs,
        'nrmse': nrmse,
        'rmse': rmse,
        'mean_adj_corr': mean_adj_corr,
        'adj_corrs': adj_corrs,
        'imf_mean_freqs': imf_mean_freqs,
        'imf_peak_freqs': imf_peak_freqs,
        'energy_percentages': energy_percentages,
        'oi': oi
    }

def print_comparison(stickout, file_name):
    base_name = file_name.replace(".mat", "")
    baseline_npz = f"/home/harshit/Documents/Research/Vibration_IMFs/{stickout}/{base_name}_IMFs.npz"
    finetuned_npz = f"/home/harshit/Documents/Research/Vibration_IMFs_FineTuned/{stickout}/{base_name}_IMFs.npz"
    
    base_diag = compute_diagnostics(baseline_npz)
    fine_diag = compute_diagnostics(finetuned_npz)
    
    print("=" * 65)
    print(f"📊 DECOMPOSITION DIAGNOSTICS: {stickout}/{file_name} 📊")
    print("=" * 65)
    
    if base_diag is None:
        print("Baseline npz file not found.")
    if fine_diag is None:
        print("Fine-tuned npz file not found.")
        
    if base_diag and fine_diag:
        print(f"{'Metric':<30} | {'Baseline v1':<14} | {'Fine-Tuned':<14}")
        print("-" * 65)
        print(f"{'Number of IMFs':<30} | {base_diag['num_imfs']:<14} | {fine_diag['num_imfs']:<14}")
        print(f"{'Reconstruction Error (NRMSE)':<30} | {base_diag['nrmse']:<14.2e} | {fine_diag['nrmse']:<14.2e}")
        print(f"{'Overall Orthogonality (OI)':<30} | {base_diag['oi']:<14.4f} | {fine_diag['oi']:<14.4f}")
        print(f"{'Mean Adjacent IMF Corr (MMI)':<30} | {base_diag['mean_adj_corr']:<14.4f} | {fine_diag['mean_adj_corr']:<14.4f}")
        print("-" * 65)
        print("\n⚡ ENERGY & FREQUENCY ANALYSIS BY IMF LAYER:")
        print("-" * 75)
        print(f"{'Layer':<8} | {'Baseline Energy %':<18} | {'Fine-Tuned Energy %':<19} | {'Base Mean Freq':<14} | {'Fine Mean Freq':<14}")
        print("-" * 75)
        
        max_layers = max(base_diag['num_imfs'], fine_diag['num_imfs'])
        for i in range(max_layers):
            is_residual_base = (i == base_diag['num_imfs'] - 1)
            is_residual_fine = (i == fine_diag['num_imfs'] - 1)
            
            # Energy
            eb = f"{base_diag['energy_percentages'][i]:.2f}%" if i < base_diag['num_imfs'] else "N/A"
            ef = f"{fine_diag['energy_percentages'][i]:.2f}%" if i < fine_diag['num_imfs'] else "N/A"
            
            # Freq
            if i < base_diag['num_imfs'] - 1:
                fb = f"{base_diag['imf_mean_freqs'][i]:.1f} Hz"
            else:
                fb = "Residual" if is_residual_base else "N/A"
                
            if i < fine_diag['num_imfs'] - 1:
                ff = f"{fine_diag['imf_mean_freqs'][i]:.1f} Hz"
            else:
                ff = "Residual" if is_residual_fine else "N/A"
                
            layer_name = f"IMF {i+1}" if (not is_residual_base and not is_residual_fine) else "Residual"
            print(f"{layer_name:<8} | {eb:<18} | {ef:<19} | {fb:<14} | {ff:<14}")
        print("-" * 75)
        
        # Write to Markdown file
        md_content = f"""# EMD Decomposition Diagnostics: {stickout}/{file_name}

## 📊 Summary Comparison Metrics
| Metric | Baseline v1 (Original) | Fine-Tuned (Optimized) |
|---|---|---|
| **Number of IMFs** | {base_diag['num_imfs']} | {fine_diag['num_imfs']} |
| **Reconstruction Error (NRMSE)** | {base_diag['nrmse']:.2e} | {fine_diag['nrmse']:.2e} |
| **Overall Orthogonality Index (OI)** | {base_diag['oi']:.4f} | {fine_diag['oi']:.4f} |
| **Mean Adjacent IMF Correlation (MMI)** | {base_diag['mean_adj_corr']:.4f} | {fine_diag['mean_adj_corr']:.4f} |

---

## ⚡ Energy & Spectral Analysis
| Layer | Baseline Energy % | Fine-Tuned Energy % | Baseline Mean Freq | Fine-Tuned Mean Freq |
|---|---|---|---|---|
"""
        for i in range(max_layers):
            is_residual_base = (i == base_diag['num_imfs'] - 1)
            is_residual_fine = (i == fine_diag['num_imfs'] - 1)
            eb = f"{base_diag['energy_percentages'][i]:.2f}%" if i < base_diag['num_imfs'] else "N/A"
            ef = f"{fine_diag['energy_percentages'][i]:.2f}%" if i < fine_diag['num_imfs'] else "N/A"
            fb = f"{base_diag['imf_mean_freqs'][i]:.1f} Hz" if i < base_diag['num_imfs'] - 1 else ("Residual" if is_residual_base else "N/A")
            ff = f"{fine_diag['imf_mean_freqs'][i]:.1f} Hz" if i < fine_diag['num_imfs'] - 1 else ("Residual" if is_residual_fine else "N/A")
            layer_name = f"IMF {i+1}" if (i < max(base_diag['num_imfs']-1, fine_diag['num_imfs']-1)) else "Residual"
            md_content += f"| **{layer_name}** | {eb} | {ef} | {fb} | {ff} |\n"
            
        md_path = f"/home/harshit/.gemini/antigravity-cli/brain/653b7dd5-5975-4225-ae8d-35c787e7fcf9/decomposition_diagnostics.md"
        with open(md_path, "w") as fd:
            fd.write(md_content)
        print(f"\nSaved diagnostic report to: {md_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        print_comparison(sys.argv[1], sys.argv[2])
    else:
        print_comparison("2p5inch_stickout", "u_570_005.mat")
