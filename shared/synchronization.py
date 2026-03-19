"""
Multi-sensor synchronization utilities
"""
import numpy as np
from scipy.interpolate import interp1d


def synchronize_streams(streams, target_fs=100):
    """
    Synchronize multiple sensor streams to common sampling rate
    
    Args:
        streams: Dict of {sensor_name: {'data': array, 'fs': float, 'timestamps': array}}
        target_fs: Target sampling frequency
    
    Returns:
        Dict of synchronized streams with uniform timestamps
    """
    # Find common time range
    min_time = max(stream['timestamps'][0] for stream in streams.values())
    max_time = min(stream['timestamps'][-1] for stream in streams.values())
    
    # Create uniform time grid
    n_samples = int((max_time - min_time) * target_fs)
    uniform_time = np.linspace(min_time, max_time, n_samples)
    
    # Resample each stream
    synchronized = {'timestamps': uniform_time}
    
    for name, stream in streams.items():
        data = stream['data']
        timestamps = stream['timestamps']
        
        if data.ndim == 1:
            # Single channel
            interpolator = interp1d(timestamps, data, kind='linear', fill_value='extrapolate')
            synchronized[name] = interpolator(uniform_time)
        else:
            # Multiple channels
            resampled = np.zeros((n_samples, data.shape[1]))
            for ch in range(data.shape[1]):
                interpolator = interp1d(timestamps, data[:, ch], kind='linear', fill_value='extrapolate')
                resampled[:, ch] = interpolator(uniform_time)
            synchronized[name] = resampled
    
    return synchronized


def align_by_trigger(streams, trigger_channel, threshold=0.5):
    """
    Align streams using a trigger signal
    
    Args:
        streams: Dict of sensor streams
        trigger_channel: Name of stream containing trigger
        threshold: Trigger detection threshold
    
    Returns:
        Aligned streams starting from trigger
    """
    trigger = streams[trigger_channel]
    trigger_idx = np.where(trigger > threshold)[0][0]
    
    aligned = {}
    for name, data in streams.items():
        if name == 'timestamps':
            aligned[name] = data[trigger_idx:] - data[trigger_idx]
        else:
            aligned[name] = data[trigger_idx:]
    
    return aligned