"""
Shared signal processing utilities
Used by both prosthetic control and embodiment model
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Bandpass filter for any signal
    
    Args:
        signal: Input signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, freq, fs, quality=30):
    """
    Notch filter (typically for 60Hz powerline noise)
    
    Args:
        signal: Input signal
        freq: Frequency to remove (Hz)
        fs: Sampling frequency (Hz)
        quality: Quality factor
    
    Returns:
        Filtered signal
    """
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)

# TODO: Check and update compatibility with prosthetic control structure 
def sliding_window(signal, window_size, step_size):
    """
    Create sliding windows from signal
    
    Args:
        signal: Input signal (1D or 2D array)
        window_size: Window size in samples
        step_size: Step size in samples
    
    Returns:
        Array of shape (n_windows, window_size, n_channels)
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n_samples, n_channels = signal.shape
    n_windows = (n_samples - window_size) // step_size + 1
    
    windows = np.zeros((n_windows, window_size, n_channels))
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = signal[start:end]
    
    return windows

# TODO: revise this normalization function not in line with the robust scaling used in ML 
#       check that this method is reasonable
def normalize_signal(signal, method='zscore'):
    """
    Normalize signal
    
    Args:
        signal: Input signal
        method: 'zscore' or 'minmax'
    
    Returns:
        Normalized signal
    """
    if method == 'zscore':
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    elif method == 'minmax':
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")