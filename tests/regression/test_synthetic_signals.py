import numpy as np

from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.weighting import calculate_physics_gated_weights, reconstruct_weighted_signal
from pg_amcd.decomposition import run_ceemdan
from pg_amcd.denoising import wavelet_denoise

def test_synthetic_signal_pipeline():
    """Generates a synthetic signal with forced tooth harmonics, self-excited chatter,
    baseline drift, and sensor white noise, and verifies pipeline performance.
    """
    fs = 10000.0
    N = 10000 # 1.0 second duration
    t = np.arange(N) / fs
    
    # 1. Spindle parameters: 600 RPM (10 Hz spindle, 10 Hz tooth passing frequency)
    rpm = 600.0
    tooth_count = 1
    f_tp = (rpm / 60.0) * tooth_count
    
    # 2. Generate forced tooth passing harmonics (10 Hz, 20 Hz, 30 Hz)
    x_forced = 0.6 * np.sin(2 * np.pi * f_tp * t) + 0.3 * np.sin(2 * np.pi * (2 * f_tp) * t)
    
    # 3. Generate self-excited chatter component at 1250 Hz (starts at 0.5 seconds)
    # Using a sigmoidal amplitude envelope for onset
    chatter_envelope = 1.0 / (1.0 + np.exp(-15 * (t - 0.5)))
    x_chatter = chatter_envelope * np.sin(2 * np.pi * 1250 * t)
    
    # 4. Generate low-frequency sensor baseline drift (quadratic)
    x_drift = 0.5 * (t ** 2)
    
    # 5. Generate broadband white noise
    np.random.seed(1234)
    x_noise = np.random.normal(0, 0.25, N)
    
    # Total input signal
    raw_signal = x_forced + x_chatter + x_drift + x_noise
    
    # 6. Apply preprocessing (Butterworth bandpass using SOS)
    # Target chatter center is 1250 Hz, so BP filter from 100 Hz to 4000 Hz
    phys_prep, scaled_prep, scale_factor = preprocess_signal(raw_signal, 100.0, 4000.0, fs)
    
    # Preprocessing should remove baseline drift
    # Detrended/filtered signal should have zero mean
    assert abs(np.mean(phys_prep)) < 0.05
    
    # 7. Run CEEMDAN on preprocessed scaled signal
    # Use fast settings for test runtime
    imfs = run_ceemdan(scaled_prep, trials=50, epsilon=0.02, noise_seed=42, sifting_iterations=8)
    
    # 8. Run Physics-Guided Sigmoidal Gating
    config = {
        "sampling_rate": fs,
        "maiw": {
            "alpha": 0.25,
            "beta": 0.25,
            "gamma": 0.25,
            "delta": 0.25,
            "chatter_band_center": 1250.0,
            "chatter_band_spread": 500.0
        }
    }
    gates, C, E_chatter, K, E_harmonics = calculate_physics_gated_weights(
        imfs, scaled_prep, fs, rpm, tooth_count, config
    )
    
    # Physics gating should suppress forced harmonics (e.g. low frequency IMFs dominated by 10 Hz)
    # Identify which IMF has the maximum energy in the forced harmonics band
    # Verify that its gate value is small compared to the chatter IMF gate
    chatter_imf_idx = np.argmax(E_chatter)
    harmonics_imf_idx = np.argmax(E_harmonics)
    
    # The IMF with high chatter energy should have a higher gate value than the forced harmonic IMF
    assert gates[chatter_imf_idx] > gates[harmonics_imf_idx]
    
    # 9. Reconstruct weighted signal
    reconstructed_scaled = reconstruct_weighted_signal(imfs, gates)
    
    # 10. Run Band-Aware Denoising
    denoised_scaled = wavelet_denoise(
        reconstructed_scaled,
        wavelet_name="db8",
        level=4,
        fs=fs,
        chatter_center=1250.0,
        chatter_spread=500.0,
        band_aware=True
    )
    
    denoised_physical = denoised_scaled * scale_factor
    
    # 11. Calculate SNR Improvement
    # Noise power in original signal
    noise_power_orig = np.var(x_noise)
    
    # Noise power in denoised signal (difference from true chatter component)
    # Preprocessing and weighting should have removed forced components and drift
    # So the clean signal should resemble the true chatter signal
    noise_power_clean = np.var(denoised_physical - x_chatter)
    
    snr_improvement = 10 * np.log10(noise_power_orig / noise_power_clean)
    print(f"Synthetic Validation Result - SNR Improvement: {snr_improvement:.2f} dB")
    
    # Expect positive SNR improvement (at least 2 dB)
    assert snr_improvement > 2.0
