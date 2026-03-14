"""
Control accuracy metrics from Leap Motion
Used by both prosthetic control (evaluation) and embodiment model (features)
"""
import numpy as np


def calculate_tracking_error(hand_positions, target_trajectory):
    """
    Calculate RMSE between hand position and target
    
    Args:
        hand_positions: Array of shape (n_samples, 3) [x, y, z]
        target_trajectory: Array of shape (n_samples, 3)
    
    Returns:
        RMSE in cm
    """
    errors = np.linalg.norm(hand_positions - target_trajectory, axis=1)
    return np.sqrt(np.mean(errors**2))


def calculate_movement_smoothness(hand_positions, fs=115):
    """
    Calculate movement smoothness (normalized jerk)
    
    Lower values = smoother movement
    
    Args:
        hand_positions: Array of shape (n_samples, 3)
        fs: Sampling frequency
    
    Returns:
        Smoothness metric (lower is better)
    """
    dt = 1.0 / fs
    
    # Calculate derivatives
    velocity = np.diff(hand_positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    
    # Normalized jerk
    duration = len(hand_positions) * dt
    path_length = np.sum(np.linalg.norm(velocity, axis=1)) * dt
    
    if path_length < 1e-6:
        return 0.0
    
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    smoothness = -np.sqrt(np.mean(jerk_magnitude**2)) * (duration**5) / (path_length**2)
    
    return smoothness


def calculate_path_efficiency(hand_positions, target_start, target_end):
    """
    Calculate path efficiency (actual vs optimal distance)
    
    Args:
        hand_positions: Array of shape (n_samples, 3)
        target_start: Start position [x, y, z]
        target_end: End position [x, y, z]
    
    Returns:
        Efficiency ratio (0-1, higher is better)
    """
    # Actual path length
    displacements = np.diff(hand_positions, axis=0)
    actual_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    # Optimal path length (straight line)
    optimal_length = np.linalg.norm(target_end - target_start)
    
    if actual_length < 1e-6:
        return 0.0
    
    return optimal_length / actual_length


def calculate_response_latency(hand_positions, target_onset_time, movement_threshold=0.5):
    """
    Calculate reaction time from target appearance to movement onset
    
    Args:
        hand_positions: Array of shape (n_samples, 3)
        target_onset_time: Time when target appears (in samples)
        movement_threshold: Velocity threshold for movement detection (cm/s)
    
    Returns:
        Latency in milliseconds
    """
    # Calculate velocity magnitude
    velocity = np.diff(hand_positions, axis=0)
    velocity_magnitude = np.linalg.norm(velocity, axis=1) * 115  # Convert to cm/s
    
    # Find first sample after target onset where velocity exceeds threshold
    post_onset = velocity_magnitude[target_onset_time:]
    movement_onset = np.where(post_onset > movement_threshold)[0]
    
    if len(movement_onset) == 0:
        return np.nan
    
    latency_samples = movement_onset[0]
    latency_ms = (latency_samples / 115) * 1000
    
    return latency_ms


def extract_all_control_metrics(leap_data, target_trajectory=None):
    """
    Extract all control accuracy features in one call
    
    Args:
        leap_data: Dict with keys 'hand_position', 'timestamps', etc.
        target_trajectory: Optional target trajectory for tracking tasks
    
    Returns:
        Dict of control metrics
    """
    hand_pos = leap_data['hand_position']
    
    metrics = {
        'movement_smoothness': calculate_movement_smoothness(hand_pos),
        'path_length': np.sum(np.linalg.norm(np.diff(hand_pos, axis=0), axis=1)),
    }
    
    if target_trajectory is not None:
        metrics['tracking_error'] = calculate_tracking_error(hand_pos, target_trajectory)
        metrics['path_efficiency'] = calculate_path_efficiency(
            hand_pos, 
            target_trajectory[0], 
            target_trajectory[-1]
        )
    
    if 'target_onset' in leap_data:
        metrics['response_latency'] = calculate_response_latency(
            hand_pos, 
            leap_data['target_onset']
        )
    
    return metrics