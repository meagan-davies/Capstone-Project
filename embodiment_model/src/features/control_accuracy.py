"""
Control accuracy features for embodiment model
Uses shared.control_metrics
"""
import sys
from pathlib import Path

# TODO: Revisit this file I think the implementation doesn't align with goal of
#       incorporating the control data accuracy (Note this conceptually might be flawed with our setup)

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from control_metrics import extract_all_control_metrics

def extract_control_features(leap_data, target_trajectory=None):
    """
    Extract control accuracy features for embodiment prediction
    
    Args:
        leap_data: Leap Motion data dict
        target_trajectory: Optional target trajectory
    
    Returns:
        Dict of control features
    """
    # Use shared utility
    metrics = extract_all_control_metrics(leap_data, target_trajectory)
    
    # Add embodiment-specific features
    hand_pos = leap_data['hand_position']
    
    # Additional features for embodiment
    metrics['position_variance'] = hand_pos.var(axis=0).mean()
    metrics['tracking_reliability'] = leap_data.get('tracking_confidence', 1.0)
    
    return metrics