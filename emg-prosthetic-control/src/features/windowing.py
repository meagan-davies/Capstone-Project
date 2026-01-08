def window_and_extract_features(emg_dict, imu_dict, fs_emg=963, fs_imu=148.148,
                                 window_sec=0.20, overlap_sec=0.10):
    """
    Window EMG and IMU data separately (accounting for different sampling rates)
    and extract features from aligned windows.
    """
    # Calculate window parameters
    emg_win_size = int(window_sec * fs_emg)
    emg_step = int((window_sec - overlap_sec) * fs_emg)
    
    imu_win_size = int(window_sec * fs_imu)
    imu_step = int((window_sec - overlap_sec) * fs_imu)
    
    all_features = []
    
    # Sort sensors for consistent ordering
    emg_sensors = sorted(emg_dict.keys())
    imu_sensors = sorted(imu_dict.keys())
    
    # Find minimum number of windows across all sensors
    min_windows = float('inf')
    
    for sensor in emg_sensors:
        n_windows = (len(emg_dict[sensor]) - emg_win_size) // emg_step
        min_windows = min(min_windows, n_windows)
    
    for sensor in imu_sensors:
        n_windows = (len(imu_dict[sensor]) - imu_win_size) // imu_step
        min_windows = min(min_windows, n_windows)
    
    # Extract features window by window
    for win_idx in range(max(1, min_windows)):
        window_features = []
        
        # EMG features
        for sensor in emg_sensors:
            start = win_idx * emg_step
            end = start + emg_win_size
            
            if end > len(emg_dict[sensor]):
                break
                
            window = emg_dict[sensor][start:end].flatten()
            
            feats = extract_emg_features(window)
            window_features.extend(feats)
        
        # IMU features
        for sensor in imu_sensors:
            start = win_idx * imu_step
            end = start + imu_win_size
            
            if end > len(imu_dict[sensor]):
                break
                
            window = imu_dict[sensor][start:end]
            
            feats = extract_imu_features(window)
            window_features.extend(feats)
        
        if len(window_features) > 0:
            all_features.append(window_features)
    
    return np.array(all_features)