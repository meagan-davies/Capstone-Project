"""
Updated EMG Feature Extraction

Handles:
- Single-channel (Avanti): 1D array → 8 features
- Multi-channel (Galileo): 2D array (n_samples, 4) → 8 features (combined across channels)

This matches your training approach where Galileo channels are processed together.
"""

import numpy as np

def extract_emg_features(window, fs):
    """
    Extract comprehensive EMG features from a window.
    
    Args:
        window: Either 1D array (n_samples,) for single-channel
                or 2D array (n_samples, n_channels) for multi-channel
        fs: Sampling frequency
    
    Returns:
        Array of 8 features (always 8, regardless of number of channels)
    """
    # Handle both 1D and 2D input
    if window.ndim == 1:
        # Single channel - use as is
        signal = window
    else:
        # Multi-channel - combine all channels (mean or RMS across channels)
        # This matches training where 4 Galileo channels are treated as one unit
        signal = np.mean(window, axis=1)  # Average across channels
    
    # Time-domain features
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)
    wl = np.sum(np.abs(np.diff(signal)))
    
    # Zero crossings
    zc = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    
    # Slope sign changes
    diff_signal = np.diff(signal)
    ssc = np.sum(np.diff(np.sign(diff_signal)) != 0) / (len(signal) - 1)
    
    # Frequency-domain features
    fft_vals = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_vals) ** 2

    # IMPORTANT: derive freqs from FFT length, not window length
    freqs = np.linspace(0, fs / 2, len(power_spectrum))

    total_power = np.sum(power_spectrum)

    if total_power > 0:
        mnf = np.sum(freqs * power_spectrum) / total_power
        cumsum = np.cumsum(power_spectrum)
        mdf_idx = np.argmax(cumsum >= total_power / 2)

        # Defensive clamp (prevents rare edge crashes)
        mdf_idx = min(mdf_idx, len(freqs) - 1)
        mdf = freqs[mdf_idx]
    else:
        mnf = 0.0
        mdf = 0.0

    return np.array([mav, rms, var, wl, zc, ssc, mnf, mdf])


# Test the function
if __name__ == "__main__":
    print("Testing EMG feature extraction\n")
    
    # Test 1: Single channel (Avanti)
    print("Test 1: Avanti (1D array)")
    print("-" * 50)
    avanti_data = np.random.randn(192) * 0.001
    features_avanti = extract_emg_features(avanti_data, fs=1259)
    print(f"Input shape: {avanti_data.shape}")
    print(f"Features: {len(features_avanti)}")
    print(f"✓ Expected 8 features: {len(features_avanti) == 8}\n")
    
    # Test 2: Multi-channel (Galileo)
    print("Test 2: Galileo (2D array, 4 channels)")
    print("-" * 50)
    galileo_data = np.random.randn(192, 4) * 0.001
    features_galileo = extract_emg_features(galileo_data, fs=963)
    print(f"Input shape: {galileo_data.shape}")
    print(f"Features: {len(features_galileo)}")
    print(f"✓ Expected 8 features: {len(features_galileo) == 8}\n")
    
    print("="*50)
    print("Feature count calculation:")
    print(f"  5 Avanti × 8 features = 40")
    print(f"  2 Galileo × 8 features = 16")
    print(f"  7 IMU × 25 features = 175")
    print(f"  Total = 231 features ✓")