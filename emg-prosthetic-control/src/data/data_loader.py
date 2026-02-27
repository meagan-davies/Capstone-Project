import pandas as pd
import numpy as np

from .filtering import preprocess_emg
from .filtering import preprocess_imu


def load_emg_imu(csv_path, fs_emg=962.963, fs_imu=148.1481, galileo_only=False, apply_filtering=True):
    """
    Load EMG + IMU data from Delsys Trigno CSV with proper sensor identification.

    Parameters
    ----------
    csv_path : str
        Path to CSV export from Trigno.
    fs_emg : float
        Fallback EMG sampling frequency (Hz) if not found in CSV header.
    fs_imu : float
        IMU sampling frequency (Hz).
    galileo_only : bool
        Keep only sensors with 'Galileo' in name.
    apply_filtering : bool
        Apply EMG and IMU preprocessing filters.

    Returns
    -------
    emg_data : dict
        {sensor_name: np.ndarray (n_samples, n_emg_channels)}
    imu_data : dict
        {sensor_name: np.ndarray (n_samples, 6)}
    time_data : dict
        {sensor_name: time_emg, sensor_name_imu: time_imu}
    emg_fs_map : dict
        {sensor_name: fs} — per-sensor EMG sampling rate parsed from CSV header
    fs_imu : float
        IMU sampling rate
    """

    # --- STEP 1: Read header rows ---
    # Row index 1 (skiprows=1): contains sampling rate strings like "962.963 Hz"
    fs_row_raw  = pd.read_csv(csv_path, header=None, skiprows=1, nrows=1).iloc[0].tolist()
    sensor_row  = pd.read_csv(csv_path, header=None, skiprows=3, nrows=1).iloc[0].tolist()
    meas_row    = pd.read_csv(csv_path, header=None, skiprows=5, nrows=1).iloc[0].tolist()

    # Clean names
    sensor_row = [str(s).strip() if pd.notna(s) and str(s).strip() != '' else None for s in sensor_row]
    meas_row   = [str(m).strip() if pd.notna(m) and str(m).strip() != '' else None for m in meas_row]
    fs_row_raw = [str(f).strip() if pd.notna(f) else None for f in fs_row_raw]

    # Extend rows to equal length
    max_len = max(len(sensor_row), len(meas_row), len(fs_row_raw))
    sensor_row  += [None] * (max_len - len(sensor_row))
    meas_row    += [None] * (max_len - len(meas_row))
    fs_row_raw  += [None] * (max_len - len(fs_row_raw))

    # Forward-fill sensor names
    for i in range(len(sensor_row)):
        if sensor_row[i] is None:
            j = i - 1
            while j >= 0 and sensor_row[j] is None:
                j -= 1
            sensor_row[i] = sensor_row[j] if j >= 0 else f"Unknown_{i}"

    # Keep only columns with valid measurement label
    valid_indices = [i for i, m in enumerate(meas_row) if m is not None]
    sensor_row  = [sensor_row[i] for i in valid_indices]
    meas_row    = [meas_row[i]   for i in valid_indices]
    fs_row_raw  = [fs_row_raw[i] for i in valid_indices]

    # Parse fs values from header (e.g. "962.963 Hz" -> 962.963)
    def _parse_fs(val):
        if val is None:
            return None
        try:
            return float(str(val).replace('Hz', '').strip())
        except ValueError:
            return None

    fs_values = [_parse_fs(v) for v in fs_row_raw]

    # Combined column names
    combined_cols = [f"{sensor}||{meas}" for sensor, meas in zip(sensor_row, meas_row)]

    # Load numeric data
    df = pd.read_csv(csv_path, header=None, skiprows=8, usecols=valid_indices, low_memory=False)
    df.columns = combined_cols
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.strip(), errors='coerce'))

    # --- STEP 2: Initialize sensor dictionary ---
    unique_sensors = set(sensor_row)
    sensor_data = {}
    for sensor in unique_sensors:
        if galileo_only and "Galileo" not in sensor:
            continue
        sensor_data[sensor] = {
            'emg': [], 'time_emg': None,
            'acc_x': None, 'acc_y': None, 'acc_z': None,
            'gyro_x': None, 'gyro_y': None, 'gyro_z': None,
            'time_imu': None,
            'emg_fs': None,   # will be filled from header
        }

    # --- STEP 3: Parse columns ---
    for col_idx, col in enumerate(combined_cols):
        sensor_name, meas_name = col.split('||')
        meas_name_lower = meas_name.lower()
        if galileo_only and "Galileo" not in sensor_name:
            continue
        if sensor_name not in sensor_data:
            sensor_data[sensor_name] = {
                'emg': [], 'time_emg': None,
                'acc_x': None, 'acc_y': None, 'acc_z': None,
                'gyro_x': None, 'gyro_y': None, 'gyro_z': None,
                'time_imu': None,
                'emg_fs': None,
            }

        data_col = df[col].values

        if 'emg' in meas_name_lower or '(mv)' in meas_name_lower:
            sensor_data[sensor_name]['emg'].append(data_col)
            # Capture fs from header for this sensor (first EMG channel seen)
            if sensor_data[sensor_name]['emg_fs'] is None:
                parsed = fs_values[col_idx]
                if parsed is not None:
                    sensor_data[sensor_name]['emg_fs'] = parsed

        elif 'time' in meas_name_lower and 'emg' in meas_name_lower:
            sensor_data[sensor_name]['time_emg'] = data_col
        elif 'acc x' in meas_name_lower:
            sensor_data[sensor_name]['acc_x'] = data_col
        elif 'acc y' in meas_name_lower:
            sensor_data[sensor_name]['acc_y'] = data_col
        elif 'acc z' in meas_name_lower:
            sensor_data[sensor_name]['acc_z'] = data_col
        elif 'gyro x' in meas_name_lower:
            sensor_data[sensor_name]['gyro_x'] = data_col
        elif 'gyro y' in meas_name_lower:
            sensor_data[sensor_name]['gyro_y'] = data_col
        elif 'gyro z' in meas_name_lower:
            sensor_data[sensor_name]['gyro_z'] = data_col
        elif 'acc' in meas_name_lower and 'time' in meas_name_lower:
            sensor_data[sensor_name]['time_imu'] = data_col

    # --- STEP 4: Build final arrays ---
    emg_data, imu_data, time_data, emg_fs_map = {}, {}, {}, {}

    for sensor, data in sensor_data.items():
        # EMG
        if data['emg']:
            min_len_emg = min(len(ch) for ch in data['emg'])
            emg_stack = np.column_stack([ch[:min_len_emg] for ch in data['emg']])
            emg_stack = emg_stack[~np.isnan(emg_stack).any(axis=1)]

            # Determine this sensor's actual fs
            sensor_fs = data['emg_fs'] if data['emg_fs'] is not None else fs_emg

            if apply_filtering:
                # preprocess_emg handles DC removal internally — don't subtract mean here
                emg_stack = preprocess_emg(emg_stack, fs=sensor_fs)

            emg_data[sensor] = emg_stack
            emg_fs_map[sensor] = sensor_fs

            if data['time_emg'] is not None:
                time_emg_trimmed = data['time_emg'][:min_len_emg]
                time_data[sensor] = time_emg_trimmed[:emg_stack.shape[0]].reshape(-1, 1)

        # IMU
        imu_channels = [data[k] for k in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                        if data[k] is not None]
        imu_channels = [ch for ch in imu_channels if not np.all(np.isnan(ch))]
        if imu_channels:
            min_len_imu = min(len(ch) for ch in imu_channels)
            imu_stack = np.column_stack([ch[:min_len_imu] for ch in imu_channels])
            imu_stack = imu_stack[~np.isnan(imu_stack).any(axis=1)]

            if apply_filtering:
                imu_stack = preprocess_imu(imu_stack, fs=fs_imu)

            imu_data[sensor] = imu_stack

            if data['time_imu'] is not None:
                time_imu_trimmed = data['time_imu'][:min_len_imu]
                time_data[f"{sensor}_imu"] = time_imu_trimmed[:imu_stack.shape[0]].reshape(-1, 1)

    print(f"  EMG sensors loaded: {sorted(emg_fs_map.keys())}")
    print(f"  Per-sensor EMG fs:  {emg_fs_map}")

    return emg_data, imu_data, time_data, emg_fs_map, fs_imu