import numpy as np
from .emg_features import extract_emg_features
from .imu_features import extract_imu_features


def window_and_extract_features(
    emg_dict,
    imu_dict,
    emg_fs_map,
    fs_imu,
    window_sec,
    overlap_sec,
    include_partial=True
):
    """
    Window EMG and IMU data and extract features.
    Each EMG sensor uses its own sampling rate from emg_fs_map.

    Parameters
    ----------
    emg_dict : dict
        {sensor_name: np.ndarray}  EMG data per sensor
    imu_dict : dict
        {sensor_name: np.ndarray}  IMU data per sensor
    emg_fs_map : dict
        {sensor_name: float}  Per-sensor EMG sampling rate (from CSV header)
    fs_imu : float
        IMU sampling frequency (shared across all IMU sensors)
    window_sec : float
        Window duration in seconds
    overlap_sec : float
        Window overlap in seconds
    include_partial : bool
        Include final partial window if data doesn't divide evenly

    Returns
    -------
    np.ndarray
        Shape (n_windows, n_features)
    """

    emg_sensors = sorted(emg_dict.keys())
    imu_sensors = sorted(imu_dict.keys())

    # Per-sensor EMG window sizes and steps
    emg_win_sizes = {
        s: int(window_sec * emg_fs_map[s]) for s in emg_sensors
    }
    emg_steps = {
        s: max(1, int((window_sec - overlap_sec) * emg_fs_map[s])) for s in emg_sensors
    }

    # Shared IMU window size and step
    imu_win_size = int(window_sec * fs_imu)
    imu_step = max(1, int((window_sec - overlap_sec) * fs_imu))

    def calc_n_windows(data_len, win_size, step):
        if data_len < win_size:
            return 1 if include_partial else 0
        return ((data_len - win_size) // step) + 1

    emg_n_windows = [
        calc_n_windows(len(emg_dict[s]), emg_win_sizes[s], emg_steps[s])
        for s in emg_sensors
    ]
    imu_n_windows = [
        calc_n_windows(len(imu_dict[s]), imu_win_size, imu_step)
        for s in imu_sensors
    ]

    n_windows = min(emg_n_windows + imu_n_windows)
    if n_windows == 0:
        raise ValueError(
            "No windows could be created. Check window/overlap parameters or data length."
        )

    all_features = []

    for win_idx in range(n_windows):
        window_features = []

        # --- EMG features (per-sensor fs) ---
        for sensor in emg_sensors:
            fs      = emg_fs_map[sensor]
            win_size = emg_win_sizes[sensor]
            step    = emg_steps[sensor]
            data    = emg_dict[sensor]

            start = win_idx * step
            end   = start + win_size

            if end > len(data):
                window = data[start:] if include_partial else None
            else:
                window = data[start:end]

            if window is None or len(window) == 0:
                continue

            feats = extract_emg_features(window, fs)   # correct fs per sensor
            window_features.extend(feats)

        # --- IMU features (shared fs) ---
        for sensor in imu_sensors:
            data  = imu_dict[sensor]
            start = win_idx * imu_step
            end   = start + imu_win_size

            if end > len(data):
                window = data[start:] if include_partial else None
            else:
                window = data[start:end]

            if window is None or len(window) == 0:
                continue

            feats = extract_imu_features(window)
            window_features.extend(feats)

        if len(window_features) > 0:
            all_features.append(window_features)

    return np.array(all_features)