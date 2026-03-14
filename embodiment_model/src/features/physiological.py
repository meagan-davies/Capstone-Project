"""
Physiological features from BioRadio and Apple Watch
"""
import numpy as np
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from signal_processing import bandpass_filter


def extract_hrv_features(ecg_signal, fs=1000):
    """
    Extract heart rate variability features
    
    Args:
        ecg_signal: ECG signal
        fs: Sampling frequency
    
    Returns:
        Dict of HRV features
    """
    # Filter ECG
    ecg_filtered = bandpass_filter(ecg_signal, 0.5, 50, fs)
    
    # Detect R-peaks (simplified)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ecg_filtered, distance=fs*0.6, height=np.std(ecg_filtered))
    
    # RR intervals (ms)
    rr_intervals = np.diff(peaks) / fs * 1000
    
    if len(rr_intervals) < 2:
        return {
            'hr_mean': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_sdnn': np.nan,
        }
    
    # Features
    features = {
        'hr_mean': 60000 / np.mean(rr_intervals),  # bpm
        'hrv_rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),  # RMSSD
        'hrv_sdnn': np.std(rr_intervals),  # SDNN
    }
    
    return features

# TODO: this may not be incorporated into the model data, revisit this implementation
def extract_emg_features(emg_signal, fs=1000):
    """
    Extract EMG features
    
    Args:
        emg_signal: EMG signal
        fs: Sampling frequency
    
    Returns:
        Dict of EMG features
    """
    # Filter EMG
    emg_filtered = bandpass_filter(emg_signal, 20, 450, fs)
    
    # Features
    features = {
        'emg_rms': np.sqrt(np.mean(emg_filtered**2)),
        'emg_mean_abs': np.mean(np.abs(emg_filtered)),
        'emg_peak_frequency': extract_peak_frequency(emg_filtered, fs),
    }
    
    return features

# TODO: confirm frequency rate in data and/or datasheet
def extract_eda_features(eda_signal, fs=1000):
    """
    Extract electrodermal activity features
    
    Args:
        eda_signal: EDA/GSR signal
        fs: Sampling frequency
    
    Returns:
        Dict of EDA features
    """
    # Decompose into tonic and phasic components (simplified)
    from scipy.signal import butter, filtfilt
    
    # Tonic (slow component)
    b, a = butter(3, 0.05 / (fs/2), btype='low')
    tonic = filtfilt(b, a, eda_signal)
    phasic = eda_signal - tonic
    
    # Count peaks in phasic component
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(phasic, height=0.01)
    
    features = {
        'eda_mean': np.mean(eda_signal),
        'eda_tonic_mean': np.mean(tonic),
        'eda_phasic_peaks': len(peaks),
        'eda_slope': np.polyfit(np.arange(len(eda_signal)), eda_signal, 1)[0],
    }
    
    return features


def extract_peak_frequency(signal, fs):
    """Extract peak frequency from FFT"""
    from scipy.fft import fft, fftfreq
    
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal), 1/fs)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]
    
    peak_idx = np.argmax(positive_fft)
    return positive_freqs[peak_idx]

# TODO: if removing emg this will need to be modified
def extract_all_physiological_features(bioradio_data, apple_watch_data):
    """
    Extract all physiological features
    
    Args:
        bioradio_data: Dict with 'emg', 'ecg', 'eda' arrays
        apple_watch_data: Dict with 'heart_rate', 'accelerometer' arrays
    
    Returns:
        Combined feature dict
    """
    features = {}
    
    # BioRadio features
    if 'ecg' in bioradio_data:
        features.update(extract_hrv_features(bioradio_data['ecg']))
    
    if 'emg' in bioradio_data:
        features.update(extract_emg_features(bioradio_data['emg']))
    
    if 'eda' in bioradio_data:
        features.update(extract_eda_features(bioradio_data['eda']))
    
    # Apple Watch features
    if 'heart_rate' in apple_watch_data:
        hr = apple_watch_data['heart_rate']
        features['watch_hr_mean'] = np.mean(hr)
        features['watch_hr_std'] = np.std(hr)
    
    if 'accelerometer' in apple_watch_data:
        accel = apple_watch_data['accelerometer']
        features['watch_accel_magnitude'] = np.mean(np.linalg.norm(accel, axis=1))
    
    return features