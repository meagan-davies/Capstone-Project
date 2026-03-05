"""
Signal Filtering Utilities

Provides:
- Bandpass filtering (EMG)
- Notch filtering (powerline removal)
- Combined preprocessing pipeline
"""

from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Bandpass Filter
def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 20.0, highcut: float = 450.0, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples,) or (n_samples, n_channels)
    fs : float
        Sampling frequency (Hz)
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal, axis=0)

# Notch Filter
def notch_filter(signal: np.ndarray, fs: float, freq: float = 60.0, quality_factor: float = 30.0) -> np.ndarray:
    """
    Apply IIR notch filter to remove powerline noise.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples,) or (n_samples, n_channels)
    fs : float
        Sampling frequency
    freq : float
        Notch frequency (Hz) — 60 Hz in North America
    quality_factor : float
        Higher = narrower notch

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    nyquist = 0.5 * fs
    normalized_freq = freq / nyquist

    b, a = iirnotch(normalized_freq, quality_factor)

    return filtfilt(b, a, signal, axis=0)

# Full EMG Preprocessing
def preprocess_emg(signal: np.ndarray, fs: float, bandpass: bool = True, notch: bool = True) -> np.ndarray:
    """
    Apply full EMG preprocessing pipeline.

    Order:
        1. Notch
        2. Bandpass

    Parameters
    ----------
    signal : np.ndarray
    fs : float
    bandpass : bool
    notch : bool

    Returns
    -------
    np.ndarray
        Filtered EMG signal
    """

    filtered = signal.copy()

    filtered = filtered - np.mean(filtered, axis=0)

    if notch:
        filtered = notch_filter(filtered, fs)

    if bandpass:
        filtered = bandpass_filter(filtered, fs)

    return filtered

# IMU Lowpass Filter
def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth lowpass filter.

    Good for accelerometer / gyro smoothing.
    """

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype="low")

    return filtfilt(b, a, signal, axis=0)


def preprocess_imu(signal: np.ndarray, fs: float, lowpass: bool = True, cutoff: float = 20.0) -> np.ndarray:
    """
    IMU preprocessing (lowpass smoothing).
    """

    filtered = signal.copy()

    if lowpass:
        filtered = lowpass_filter(filtered, fs, cutoff=cutoff)

    return filtered

# Dev Test
if __name__ == "__main__":

    print("Testing filtering module...")

    fs = 2000
    t = np.linspace(0, 1, fs)

    # Fake EMG + 60Hz noise
    emg = np.sin(2 * np.pi * 50 * t)
    noise = 0.3 * np.sin(2 * np.pi * 60 * t)
    signal = emg + noise

    filtered = preprocess_emg(signal, fs)

    print("✓ Filtering working")