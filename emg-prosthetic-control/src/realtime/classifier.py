"""
Real-Time Classifier

Loads trained model and makes predictions on streaming features.
"""

import pickle
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class RealtimeClassifier:
    """
    Real-time gesture classifier with prediction smoothing.
    """
    
    def __init__(self, model_path: str, scaler_path: str, 
                 class_names: Optional[Dict[int, str]] = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained_model.pkl
            scaler_path: Path to scaler.pkl
            class_names: Dictionary mapping class IDs to names
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Class names
        self.class_names = class_names or {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping"
        }
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=5)
        
        print(f"✓ Classifier loaded")
        print(f"  Model: {type(self.model).__name__}")
        print(f"  Classes: {list(self.class_names.values())}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict class from features.
        
        Args:
            features: Feature vector
            
        Returns:
            (predicted_label, probabilities)
        """
        if features is None:
            return None, None
        
        # Scale features (same as training!)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        pred_label = self.model.predict(features_scaled)[0]
        pred_proba = self.model.predict_proba(features_scaled)[0]
        
        return pred_label, pred_proba
    
    def predict_smoothed(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict with temporal smoothing (majority vote).
        
        Args:
            features: Feature vector
            
        Returns:
            (smoothed_label, probabilities)
        """
        pred_label, pred_proba = self.predict(features)
        
        if pred_label is not None:
            # Add to buffer
            self.prediction_buffer.append(pred_label)
            
            # Majority vote
            if len(self.prediction_buffer) > 0:
                smoothed_pred = max(
                    set(self.prediction_buffer),
                    key=self.prediction_buffer.count
                )
                return smoothed_pred, pred_proba
        
        return pred_label, pred_proba
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label"""
        return self.class_names.get(label, f"Unknown ({label})")