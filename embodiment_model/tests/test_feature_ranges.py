"""
Test that extracted features are in valid ranges
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "shared"))

from src.features.physiological import (
    extract_hrv_features,
    extract_emg_features,
    extract_eda_features
)
from src.features.control_accuracy import extract_control_features


def test_hrv_features_range():
    """Test HRV features are in valid physiological ranges"""
    # Simulate ECG signal (1000 Hz, 10 seconds)
    t = np.linspace(0, 10, 10000)
    heart_rate_hz = 75 / 60  # 75 bpm
    ecg = np.sin(2 * np.pi * heart_rate_hz * t) + 0.1 * np.random.randn(len(t))
    
    features = extract_hrv_features(ecg, fs=1000)
    
    # Check HR is in valid range (40-200 bpm)
    assert 40 <= features['hr_mean'] <= 200, f"HR out of range: {features['hr_mean']}"
    
    # Check HRV metrics are positive
    assert features['hrv_rmssd'] >= 0, "RMSSD should be positive"
    assert features['hrv_sdnn'] >= 0, "SDNN should be positive"


def test_emg_features_range():
    """Test EMG features are in valid ranges"""
    # Simulate EMG signal
    emg = 0.1 * np.random.randn(1000) + 0.05 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    
    features = extract_emg_features(emg, fs=1000)
    
    # Check all features are finite
    assert np.isfinite(features['emg_rms']), "EMG RMS should be finite"
    assert np.isfinite(features['emg_mean_abs']), "EMG mean abs should be finite"
    assert np.isfinite(features['emg_peak_frequency']), "EMG peak freq should be finite"
    
    # Check RMS is positive
    assert features['emg_rms'] >= 0, "EMG RMS should be positive"


def test_eda_features_range():
    """Test EDA features are in valid ranges"""
    # Simulate EDA signal
    t = np.linspace(0, 60, 60000)  # 1 minute at 1000 Hz
    eda = 2.0 + 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.random.randn(len(t))
    
    features = extract_eda_features(eda, fs=1000)
    
    # Check mean is in typical range (0.5-5 µS)
    assert 0.5 <= features['eda_mean'] <= 5.0, f"EDA mean out of range: {features['eda_mean']}"
    
    # Check peak count is reasonable (0-10 peaks per minute)
    assert 0 <= features['eda_phasic_peaks'] <= 10, f"EDA peaks out of range: {features['eda_phasic_peaks']}"


def test_control_features_range():
    """Test control accuracy features are in valid ranges"""
    # Simulate hand trajectory
    t = np.linspace(0, 5, 575)  # 5 seconds at 115 Hz
    hand_position = np.column_stack([
        10 * np.sin(2 * np.pi * 0.5 * t),
        200 + 5 * np.cos(2 * np.pi * 0.3 * t),
        -100 + 3 * np.sin(2 * np.pi * 0.4 * t)
    ])
    
    leap_data = {
        'hand_position': hand_position,
        'timestamps': t
    }
    
    features = extract_control_features(leap_data)
    
    # Check smoothness is finite
    assert np.isfinite(features['movement_smoothness']), "Smoothness should be finite"
    
    # Check variance is positive
    assert features['position_variance'] >= 0, "Position variance should be positive"


def test_no_nan_features():
    """Test that features don't contain NaN values"""
    # Generate random but valid sensor data
    ecg = np.random.randn(1000)
    emg = np.random.randn(1000)
    eda = 2.0 + 0.1 * np.random.randn(1000)
    
    hrv_features = extract_hrv_features(ecg, fs=1000)
    emg_features = extract_emg_features(emg, fs=1000)
    eda_features = extract_eda_features(eda, fs=1000)
    
    all_features = {**hrv_features, **emg_features, **eda_features}
    
    for name, value in all_features.items():
        assert not np.isnan(value), f"Feature {name} is NaN"


def test_feature_consistency():
    """Test that same input produces same features"""
    # Fixed random seed
    np.random.seed(42)
    emg1 = np.random.randn(1000)
    
    features1 = extract_emg_features(emg1, fs=1000)
    
    # Reset seed and generate same data
    np.random.seed(42)
    emg2 = np.random.randn(1000)
    
    features2 = extract_emg_features(emg2, fs=1000)
    
    # Check features are identical
    for key in features1.keys():
        np.testing.assert_almost_equal(
            features1[key], 
            features2[key],
            decimal=10,
            err_msg=f"Feature {key} inconsistent"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])