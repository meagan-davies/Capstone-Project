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

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        class_names: Optional[Dict[int, str]] = None,
        smoothing_window: int = 5,
        strict_feature_check: bool = True,
    ):
        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.expected_n_features = self.scaler.n_features_in_
        self.strict_feature_check = strict_feature_check

        # Class names
        self.class_names = class_names or {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping",
        }

        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=smoothing_window)

        print("✓ Classifier loaded")
        print(f"  Model: {type(self.model).__name__}")
        print(f"  Expected features: {self.expected_n_features}")
        print(f"  Classes: {list(self.class_names.values())}")

    # def _validate_features(self, features: np.ndarray) -> bool:
    #     """Validate feature vector before prediction."""
    #     if features is None:
    #         return False

    #     if np.any(np.isnan(features)):
    #         return False

    #     if features.ndim != 1:
    #         raise ValueError(f"Expected 1D feature vector, got {features.shape}")

    #     if features.shape[0] != self.expected_n_features:
    #         msg = (
    #             f"Feature mismatch: got {features.shape[0]}, "
    #             f"expected {self.expected_n_features}"
    #         )
    #         if self.strict_feature_check:
    #             raise ValueError(msg)
    #         else:
    #             print(f"⚠ {msg}")
    #             return False

    #     return True

    # In classifier.py, replace the predict method:
    def predict(self, features: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Predict class from features."""
        
        # DEBUG: Print feature info
        print(f"Received {len(features)} features, expected {self.expected_n_features}")
        
        # Temporary: Pad with zeros if too short (THIS IS JUST FOR TESTING!)
        if len(features) < self.expected_n_features:
            print(f"⚠ WARNING: Padding {self.expected_n_features - len(features)} features with zeros")
            features = np.pad(features, (0, self.expected_n_features - len(features)), 'constant')
        elif len(features) > self.expected_n_features:
            print(f"⚠ WARNING: Truncating to {self.expected_n_features} features")
            features = features[:self.expected_n_features]
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        pred_label = self.model.predict(features_scaled)[0]
        pred_proba = self.model.predict_proba(features_scaled)[0]
        
        return pred_label, pred_proba


    # def predict_smoothed(
    #     self, features: np.ndarray
    # ) -> Tuple[Optional[int], Optional[np.ndarray]]:
    #     """Predict with temporal smoothing."""
    #     pred_label, pred_proba = self.predict(features)

    #     if pred_label is None:
    #         return None, None

    #     self.prediction_buffer.append(pred_label)

    #     smoothed_label = max(
    #         set(self.prediction_buffer),
    #         key=self.prediction_buffer.count,
    #     )

    #     return smoothed_label, pred_proba

    def get_class_name(self, label: int) -> str:
        return self.class_names.get(label, f"Unknown ({label})")
