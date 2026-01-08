"""
Real-Time EMG Classification System

Brings together:
- DelsysClient (hardware)
- RealtimeProcessor (data processing)
- RealtimeClassifier (predictions)
"""

import time
from threading import Thread
from .delsys_client import DelsysClient, load_credentials
from .processor import RealtimeProcessor
from .classifier import RealtimeClassifier


class RealtimeEMGSystem:
    """
    Complete real-time EMG+IMU classification system.
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize system.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to scaler
        """
        # Load credentials
        key, license = load_credentials()
        
        # Create client
        self.client = DelsysClient(key, license)
        
        # Classifier
        self.classifier = RealtimeClassifier(model_path, scaler_path)
        
        # Processor (will be created after channel discovery)
        self.processor = None
        
        # State
        self.is_running = False
        self.poll_thread = None
    
    def setup(self) -> bool:
        """Connect and configure hardware"""
        print("\n" + "="*60)
        print("SYSTEM SETUP")
        print("="*60)
        
        # Connect
        if not self.client.connect():
            return False
        
        # Scan
        if not self.client.scan_sensors():
            return False
        
        # Configure
        if not self.client.configure():
            return False
        
        # Create processor with discovered channels
        emg_guids = self.client.get_emg_channel_guids()
        imu_guids = self.client.get_imu_channel_guids()
        
        self.processor = RealtimeProcessor(emg_guids, imu_guids)
        
        print(f"\n✓ Setup complete!")
        return True
    
    def _poll_loop(self):
        """Background thread to poll data"""
        while self.is_running:
            data = self.client.poll_data()
            if data:
                self.processor.add_data(data)
            time.sleep(0.001)
    
    def start(self):
        """Start real-time classification"""
        if not self.client.start_streaming():
            return
        
        self.is_running = True
        
        # Start polling thread
        self.poll_thread = Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        
        print("\n" + "="*60)
        print("REAL-TIME CLASSIFICATION")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                if self.processor.is_window_ready():
                    features = self.processor.extract_window_features()
                    
                    if features is not None:
                        pred_label, pred_proba = self.classifier.predict_smoothed(features)
                        class_name = self.classifier.get_class_name(pred_label)
                        confidence = pred_proba[pred_label] * 100
                        
                        print(f"\r{class_name:12s} | Confidence: {confidence:5.1f}%", 
                              end='', flush=True)
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
    
    def stop(self):
        """Stop system"""
        self.is_running = False
        
        if self.poll_thread:
            self.poll_thread.join(timeout=1)
        
        self.client.disconnect()
        print("✓ System stopped")
