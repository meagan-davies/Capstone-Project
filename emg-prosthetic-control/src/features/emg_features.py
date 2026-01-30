import numpy as np

def extract_emg_features(window, fs):
    """Extract comprehensive EMG features from a window"""
    # Time-domain features
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    var = np.var(window)
    wl = np.sum(np.abs(np.diff(window)))
    
    # Zero crossings
    zc = np.sum(np.diff(np.sign(window)) != 0) / len(window)
    
    # Slope sign changes
    diff_signal = np.diff(window)
    ssc = np.sum(np.diff(np.sign(diff_signal)) != 0) / (len(window) - 1)
    
    # Frequency-domain features
    fft_vals = np.fft.rfft(window)
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
