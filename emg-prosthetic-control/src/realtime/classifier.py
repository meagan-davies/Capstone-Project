# ============================================================
# REAL-TIME CLASSIFIER
# ============================================================

class RealtimeClassifier:
    """
    Real-time gesture classifier.
    Loads trained model and scaler, performs predictions on streaming data.
    """
    
    def __init__(self, model_path, scaler_path, class_names=None):
        # Load trained model and scaler
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.class_names = class_names or {
            0: "Neutral",
            1: "Pinching", 
            2: "Grasping",
            3: "Zipping"
        }
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
    def predict(self, features):
        """Predict class from features"""
        if features is None:
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        pred_label = self.model.predict(features_scaled)[0]
        pred_proba = self.model.predict_proba(features_scaled)[0]
        
        return pred_label, pred_proba
    
    def predict_smoothed(self, features):
        """Predict with temporal smoothing"""
        pred_label, pred_proba = self.predict(features)
        
        if pred_label is not None:
            self.prediction_buffer.append(pred_label)
            
            # Majority vote
            if len(self.prediction_buffer) > 0:
                smoothed_pred = max(set(self.prediction_buffer), 
                                   key=self.prediction_buffer.count)
                return smoothed_pred, pred_proba
        
        return pred_label, pred_proba
    
    def get_class_name(self, label):
        """Get class name from label"""
        return self.class_names.get(label, f"Unknown ({label})")