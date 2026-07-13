import pytest
import numpy as np
from pg_amcd.preprocessing import butter_bandpass_filter_sos, preprocess_signal

def test_invalid_filter_cutoffs():
    signal = np.random.randn(1000)
    # low cutoff > high cutoff
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(signal, 200.0, 100.0, 10000.0)
    # low cutoff <= 0
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(signal, 0.0, 100.0, 10000.0)
    # high cutoff >= Nyquist (5000)
    with pytest.raises(ValueError):
        butter_bandpass_filter_sos(signal, 100.0, 5000.0, 10000.0)

def test_sos_filtering():
    fs = 1000.0
    t = np.arange(1000) / fs
    # Signal with 10 Hz and 150 Hz components
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 150 * t)
    
    # Bandpass filter from 80 Hz to 200 Hz (should keep the 150 Hz and suppress the 10 Hz)
    filtered = butter_bandpass_filter_sos(signal, 80.0, 200.0, fs, order=3)
    
    # Calculate energy in both components after filtering
    # Generate reference components
    comp_10hz = np.sin(2 * np.pi * 10 * t)
    comp_150hz = np.sin(2 * np.pi * 150 * t)
    
    # The filtered signal should correlate highly with 150 Hz and poorly with 10 Hz
    corr_10 = np.abs(np.corrcoef(filtered, comp_10hz)[0, 1])
    corr_150 = np.abs(np.corrcoef(filtered, comp_150hz)[0, 1])
    
    assert corr_150 > 0.90
    assert corr_10 < 0.10

def test_scale_factor_preservation():
    # Signal with maximum amplitude 5.2
    signal = np.zeros(1000)
    t = np.arange(1000) / 10000.0
    signal = 5.2 * np.sin(2 * np.pi * 500 * t)
    
    phys_prep, scaled_prep, scale_factor = preprocess_signal(signal, 100.0, 4000.0, 10000.0)
    
    # The scale factor should be close to 5.2 (allowing for filter start-up transient overshoots)
    assert 5.0 <= scale_factor <= 6.0
    # The scaled preprocessed signal should have a max amplitude close to 1.0 (allowing 99.5th vs 100th percentile gap)
    assert 0.9 <= np.max(np.abs(scaled_prep)) <= 1.3
    # The physical preprocessed signal should equal scaled * scale_factor
    assert np.allclose(phys_prep, scaled_prep * scale_factor, atol=1e-12)
