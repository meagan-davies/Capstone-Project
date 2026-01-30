import pandas as pd
import numpy as np

def load_emg_imu(csv_path, fs_emg=962.963, fs_imu=148.1481, galileo_only=False):
    """
    Load EMG + IMU data from Delsys Trigno CSV with proper sensor identification.

    Returns:
        emg_data: dict {sensor_name: (n_samples_emg, n_emg_channels)}
        imu_data: dict {sensor_name: (n_samples_imu, 6)}
        time_data: dict {sensor_name: time_emg, sensor_name_imu: time_imu}
    """
    # --- STEP 1: Read sensor + measurement rows ---
    sensor_row = pd.read_csv(csv_path, header=None, skiprows=3, nrows=1).iloc[0].tolist()
    meas_row   = pd.read_csv(csv_path, header=None, skiprows=5, nrows=1).iloc[0].tolist()

    # Clean names
    sensor_row = [str(s).strip() if pd.notna(s) and str(s).strip() != '' else None for s in sensor_row]
    meas_row   = [str(m).strip() if pd.notna(m) and str(m).strip() != '' else None for m in meas_row]

    # Extend sensor_row to match meas_row length
    if len(sensor_row) < len(meas_row):
        sensor_row += [None]*(len(meas_row)-len(sensor_row))

    # Forward-fill sensor names
    for i in range(len(sensor_row)):
        if sensor_row[i] is None:
            j = i-1
            while j >= 0 and sensor_row[j] is None:
                j -= 1
            sensor_row[i] = sensor_row[j] if j >= 0 else f"Unknown_{i}"

    # Keep only columns with valid measurement
    valid_indices = [i for i, m in enumerate(meas_row) if m is not None]
    sensor_row = [sensor_row[i] for i in valid_indices]
    meas_row   = [meas_row[i] for i in valid_indices]

    # Combined column names
    combined_cols = [f"{sensor}||{meas}" for sensor, meas in zip(sensor_row, meas_row)]

    # Load data
    df = pd.read_csv(csv_path, header=None, skiprows=8, usecols=valid_indices, low_memory=False)
    df.columns = combined_cols
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.strip(), errors='coerce'))

    # --- STEP 2: Initialize sensor dictionary ---
    unique_sensors = set(sensor_row)
    sensor_data = {}
    for sensor in unique_sensors:
        if galileo_only and "Galileo" not in sensor:
            continue
        sensor_data[sensor] = {'emg': [], 'time_emg': None,
                               'acc_x': None,'acc_y': None,'acc_z': None,
                               'gyro_x': None,'gyro_y': None,'gyro_z': None,
                               'time_imu': None}

    # --- STEP 3: Parse columns ---
    for col in df.columns:
        sensor_name, meas_name = col.split('||')
        meas_name_lower = meas_name.lower()
        if galileo_only and "Galileo" not in sensor_name:
            continue
        if sensor_name not in sensor_data:
            sensor_data[sensor_name] = {'emg': [], 'time_emg': None,
                                        'acc_x': None,'acc_y': None,'acc_z': None,
                                        'gyro_x': None,'gyro_y': None,'gyro_z': None,
                                        'time_imu': None}

        data_col = df[col].values

        # EMG channels
        if 'emg' in meas_name_lower or '(mv)' in meas_name_lower:
            sensor_data[sensor_name]['emg'].append(data_col)
        elif 'time' in meas_name_lower and 'emg' in meas_name_lower:
            sensor_data[sensor_name]['time_emg'] = data_col
        # IMU channels
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

    # --- STEP 4: Build final EMG and IMU arrays ---
    emg_data, imu_data, time_data = {}, {}, {}
    for sensor, data in sensor_data.items():
        # --- EMG: stack channels, trim to min length of EMG channels, remove rows with NaN ---
        if data['emg']:
            min_len_emg = min(len(ch) for ch in data['emg'])
            emg_stack = np.column_stack([ch[:min_len_emg] for ch in data['emg']])
            # Remove any rows containing NaN
            emg_stack = emg_stack[~np.isnan(emg_stack).any(axis=1)]
            emg_data[sensor] = emg_stack
            if data['time_emg'] is not None:
                # Trim time array to match EMG rows after NaN removal
                time_emg_trimmed = data['time_emg'][:min_len_emg]
                time_data[sensor] = time_emg_trimmed[:emg_stack.shape[0]].reshape(-1,1)

        # --- IMU: stack channels, trim to min length of available channels, remove rows with NaN ---
        imu_channels = [data[k] for k in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'] if data[k] is not None]
        # Skip empty IMU columns entirely
        imu_channels = [ch for ch in imu_channels if not np.all(np.isnan(ch))]
        if imu_channels:
            min_len_imu = min(len(ch) for ch in imu_channels)
            imu_stack = np.column_stack([ch[:min_len_imu] for ch in imu_channels])
            # Remove rows containing NaN
            imu_stack = imu_stack[~np.isnan(imu_stack).any(axis=1)]
            imu_data[sensor] = imu_stack
            if data['time_imu'] is not None:
                time_imu_trimmed = data['time_imu'][:min_len_imu]
                time_data[f"{sensor}_imu"] = time_imu_trimmed[:imu_stack.shape[0]].reshape(-1,1)

    # # --- STEP 5: Debug prints ---
    # print("Loaded EMG sensors:", list(emg_data.keys()))
    # print("Loaded IMU sensors:", list(imu_data.keys()))
    # for sensor in emg_data.keys():
    #     print(f"{sensor}: EMG {emg_data[sensor].shape}, IMU {imu_data.get(sensor,'NA') if sensor in imu_data else 'NA'}")
    return emg_data, imu_data, time_data, fs_emg, fs_imu
