import pandas as pd
import numpy as np

def load_emg_imu(csv_path, fs_emg, fs_imu):
    """
    Load EMG + IMU data from Delsys Trigno CSV with PROPER sensor grouping.
    
    FIXED: Groups all channels from the same physical sensor together.
    
    Args:
        csv_path (str): Path to CSV file.
        fs_emg (float): EMG sampling frequency from config (Hz).
        fs_imu (float): IMU sampling frequency from config (Hz).

    Returns:
        emg_data: dict {sensor_name: (n_samples, n_emg_channels)}
        imu_data: dict {sensor_name: (n_samples, 6)} - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        time_data: dict {sensor_name: (n_samples, 1)}
        fs_emg: float
        fs_imu: float
    """
    # Read sensor row (row 4)
    sensor_row = pd.read_csv(csv_path, header=None, skiprows=3, nrows=1, skipinitialspace=True)
    sensor_row = sensor_row.ffill(axis=1).iloc[0].tolist()

    # Read measurement row (row 6)
    meas_row = pd.read_csv(csv_path, header=None, skiprows=5, nrows=1, skipinitialspace=True)
    meas_row = meas_row.iloc[0].tolist()

    # Create unique column names
    combined_cols = []
    for sensor, meas in zip(sensor_row, meas_row):
        sensor = str(sensor).strip() if pd.notna(sensor) else "Unknown"
        meas = str(meas).strip() if pd.notna(meas) else "Unknown"
        combined_cols.append(f"{sensor}||{meas}")

    # Load data starting from row 9
    df = pd.read_csv(csv_path, header=None, skiprows=8, skipinitialspace=True, 
                     low_memory=False, on_bad_lines='skip')

    # Handle column count mismatch
    n_cols = df.shape[1]
    if len(combined_cols) < n_cols:
        for i in range(n_cols - len(combined_cols)):
            combined_cols.append(f"Extra_{i}||Unknown")
    elif len(combined_cols) > n_cols:
        combined_cols = combined_cols[:n_cols]

    df.columns = combined_cols
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.strip(), errors='coerce'))
    df = df.dropna(how='all')  # Remove completely empty rows

    # Organize data by PHYSICAL sensor (not individual channels!)
    sensor_data = {}
    
    for col in df.columns:
        if '||' not in col:
            continue
        
        sensor_name, meas_name = col.split('||')
        
        # Extract base sensor name (remove the _2, _3, etc. suffix)
        # e.g., "Avanti Sensor 1 (87913)_2" -> "Avanti Sensor 1 (87913)"
        base_sensor_name = sensor_name.rsplit('_', 1)[0] if '_' in sensor_name else sensor_name
        
        if base_sensor_name not in sensor_data:
            sensor_data[base_sensor_name] = {
                'emg': [],  # List to collect multiple EMG channels
                'time_emg': None,
                'acc_x': None, 'acc_y': None, 'acc_z': None,
                'gyro_x': None, 'gyro_y': None, 'gyro_z': None,
                'time_imu': None
            }
        
        data_col = df[col].dropna().values
        
        # Classify measurement type
        if '(mV)' in meas_name and 'EMG' in meas_name:
            sensor_data[base_sensor_name]['emg'].append(data_col)
        elif 'Time Series' in meas_name and 'EMG' in meas_name:
            if sensor_data[base_sensor_name]['time_emg'] is None:
                sensor_data[base_sensor_name]['time_emg'] = data_col
        elif 'ACC X' in meas_name:
            sensor_data[base_sensor_name]['acc_x'] = data_col
        elif 'ACC Y' in meas_name:
            sensor_data[base_sensor_name]['acc_y'] = data_col
        elif 'ACC Z' in meas_name:
            sensor_data[base_sensor_name]['acc_z'] = data_col
        elif 'GYRO X' in meas_name:
            sensor_data[base_sensor_name]['gyro_x'] = data_col
        elif 'GYRO Y' in meas_name:
            sensor_data[base_sensor_name]['gyro_y'] = data_col
        elif 'GYRO Z' in meas_name:
            sensor_data[base_sensor_name]['gyro_z'] = data_col
        elif 'Time Series' in meas_name and 'ACC' in meas_name:
            if sensor_data[base_sensor_name]['time_imu'] is None:
                sensor_data[base_sensor_name]['time_imu'] = data_col

    # Extract organized data
    emg_data = {}
    imu_data = {}
    time_data = {}

    for sensor_name, data in sensor_data.items():
        # EMG data - combine multiple EMG channels into one array
        if data['emg'] and len(data['emg']) > 0:
            # Find minimum length across all EMG channels
            min_len = min(len(ch) for ch in data['emg'])
            # Truncate all to same length and stack
            emg_channels = [ch[:min_len] for ch in data['emg']]
            emg_data[sensor_name] = np.column_stack(emg_channels)
            
            if data['time_emg'] is not None:
                time_data[sensor_name] = data['time_emg'][:min_len].reshape(-1, 1)
        
        # IMU data (stack acc + gyro)
        imu_channels = []
        for key in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            if data[key] is not None and len(data[key]) > 0:
                imu_channels.append(data[key])
        
        if len(imu_channels) > 0:
            min_len = min(len(ch) for ch in imu_channels)
            imu_channels = [ch[:min_len] for ch in imu_channels]
            imu_data[sensor_name] = np.column_stack(imu_channels)

    return emg_data, imu_data, time_data, fs_emg, fs_imu
