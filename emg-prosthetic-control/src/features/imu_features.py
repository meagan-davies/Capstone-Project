def extract_imu_features(window):
    """Extract IMU features from a window (works for any number of axes)"""
    # Time-domain features
    mean_val = np.mean(window, axis=0)
    std_val = np.std(window, axis=0)
    rms_val = np.sqrt(np.mean(window**2, axis=0))
    range_val = np.max(window, axis=0) - np.min(window, axis=0)
    
    # Signal magnitude area (for 3-axis signals)
    if window.shape[1] >= 3:
        sma = np.mean(np.sum(np.abs(window[:, :3]), axis=1))
    else:
        sma = np.mean(np.sum(np.abs(window), axis=1))
    
    # Combine features
    features = np.concatenate([mean_val, std_val, rms_val, range_val, [sma]])
    return features