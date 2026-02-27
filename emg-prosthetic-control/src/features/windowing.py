import numpy as np
from .emg_features import extract_emg_features
from .imu_features import extract_imu_features

def window_and_extract_features(emg_dict, imu_dict, fs_emg, fs_imu, window_sec, overlap_sec, include_partial=True):
    """
    Window EMG and IMU data separately (accounting for different sampling rates)
    and extract features from aligned windows.

    Parameters
    ----------
    emg_dict : dict
        Keys = sensor names, Values = 1D np.array of EMG signals
    imu_dict : dict
        Keys = sensor names, Values = np.array of IMU signals (NxM)
    fs_emg : float
        EMG sampling frequency
    fs_imu : float
        IMU sampling frequency
    window_sec : float
        Window duration in seconds
    overlap_sec : float
        Window overlap in seconds
    include_partial : bool
        If True, include final partial window for sensors

    Returns
    -------
    np.ndarray
        Array of shape (n_windows, n_features)
    """

    # Calculate window sizes and steps
    emg_win_size = int(window_sec * fs_emg)
    emg_step = max(1, int((window_sec - overlap_sec) * fs_emg))

    imu_win_size = int(window_sec * fs_imu)
    imu_step = max(1, int((window_sec - overlap_sec) * fs_imu))

    all_features = []

    # Sort sensors for consistent ordering
    emg_sensors = sorted(emg_dict.keys())
    imu_sensors = sorted(imu_dict.keys())

    # Determine number of windows per sensor
    def calc_n_windows(data_len, win_size, step):
        if data_len < win_size:
            return 1 if include_partial else 0
        return ((data_len - win_size) // step) + 1

    emg_n_windows = [calc_n_windows(len(emg_dict[s]), emg_win_size, emg_step) 
                     for s in emg_sensors]
    imu_n_windows = [calc_n_windows(len(imu_dict[s]), imu_win_size, imu_step) 
                     for s in imu_sensors]

    n_windows = min(emg_n_windows + imu_n_windows)
    if n_windows == 0:
        raise ValueError("No windows could be created. Check window/overlap parameters or data length.")

    # Extract features window by window
    for win_idx in range(n_windows):
        window_features = []

        # EMG features
        for sensor in emg_sensors:
            start = win_idx * emg_step
            end = start + emg_win_size
            if end > len(emg_dict[sensor]):
                if include_partial:
                    window = emg_dict[sensor][start:]
                else:
                    continue
            else:
                window = emg_dict[sensor][start:end]

            feats = extract_emg_features(window, fs_emg)
            window_features.extend(feats)

        # IMU features
        for sensor in imu_sensors:
            start = win_idx * imu_step
            end = start + imu_win_size
            if end > len(imu_dict[sensor]):
                if include_partial:
                    window = imu_dict[sensor][start:]
                else:
                    continue
            else:
                window = imu_dict[sensor][start:end]

            feats = extract_imu_features(window)
            window_features.extend(feats)

        if len(window_features) > 0:
            all_features.append(window_features)

    return np.array(all_features)