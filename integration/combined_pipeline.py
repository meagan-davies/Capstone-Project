"""
Combined prosthetic control and embodiment monitoring system
"""

import sys
from pathlib import Path
import time
import numpy as np
from collections import deque

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "prosthetic_control"))
sys.path.insert(0, str(Path(__file__).parent.parent / "embodiment_model"))

from sensors.leap_motion import LeapMotionCapture
from sensors.bioradio import BioRadioCapture
from sensors.apple_watch import AppleWatchCapture

# Import models (assuming these exist)
# from prosthetic_control.src.models.classifier import load_gesture_classifier
# from embodiment_model.src.models.trainer import load_model as load_embodiment_model


class CombinedPipeline:
    """
    Combined real-time system running both prosthetic control and embodiment monitoring
    """
    
    def __init__(self, 
                 control_model_path: str,
                 embodiment_model_path: str):
        """
        Initialize combined pipeline
        
        Args:
            control_model_path: Path to trained prosthetic control model
            embodiment_model_path: Path to trained embodiment model
        """
        print("\n" + "="*70)
        print(" "*15 + "COMBINED PROSTHETIC & EMBODIMENT SYSTEM")
        print("="*70 + "\n")
        
        # Load models
        print("Loading models...")
        # self.control_model = load_gesture_classifier(control_model_path)
        # self.embodiment_model, _, _ = load_embodiment_model(embodiment_model_path)
        print("⚠ Model loading not implemented - using simulation mode")
        
        # Initialize sensors
        print("\nInitializing sensors...")
        self.leap = LeapMotionCapture()
        self.bioradio = BioRadioCapture()
        self.watch = AppleWatchCapture()
        
        # Data buffers
        self.embodiment_window = 5.0  # seconds
        self.buffer_size = int(self.embodiment_window * 100)
        self.sensor_buffer = {
            'leap': deque(maxlen=self.buffer_size),
            'bioradio': deque(maxlen=self.buffer_size),
            'watch': deque(maxlen=self.buffer_size)
        }
        
        # State
        self.is_running = False
        self.current_gesture = "Neutral"
        self.current_embodiment = 50.0
        self.embodiment_history = deque(maxlen=100)
        
        print("\n✓ System initialized")
    
    def start(self):
        """Start combined real-time system"""
        print("\n" + "="*70)
        print("Starting combined system...")
        print("="*70 + "\n")
        
        print("Instructions:")
        print("  - System will classify gestures in real-time")
        print("  - Embodiment score updated every 0.5s")
        print("  - Press Ctrl+C to stop")
        print()
        
        input("Press Enter to begin...")
        
        self.is_running = True
        
        # Start sensors
        self.leap.start_recording()
        self.bioradio.start_recording()
        self.watch.start_recording()
        
        last_embodiment_update = time.time()
        embodiment_update_interval = 0.5  # Update embodiment every 0.5s
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Collect sensor samples
                leap_sample = self.leap.get_frame()
                bioradio_sample = self.bioradio.get_sample()
                watch_sample = self.watch.get_sample()
                
                # Buffer for embodiment
                if leap_sample: self.sensor_buffer['leap'].append(leap_sample)
                if bioradio_sample: self.sensor_buffer['bioradio'].append(bioradio_sample)
                if watch_sample: self.sensor_buffer['watch'].append(watch_sample)
                
                # Update gesture classification (high frequency)
                # gesture = self._classify_gesture(leap_sample, bioradio_sample)
                # self.current_gesture = gesture
                
                # Update embodiment (lower frequency)
                if current_time - last_embodiment_update >= embodiment_update_interval:
                    embodiment = self._compute_embodiment()
                    if embodiment is not None:
                        self.current_embodiment = embodiment
                        self.embodiment_history.append({
                            'timestamp': current_time,
                            'score': embodiment
                        })
                    
                    last_embodiment_update = current_time
                
                # Display status
                self._display_status()
                
                time.sleep(0.01)  # 100 Hz loop
        
        except KeyboardInterrupt:
            print("\n\nStopping system...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        
        self.leap.stop_recording()
        self.bioradio.stop_recording()
        self.watch.stop_recording()
        
        print("\n\n" + "="*70)
        print("System stopped")
        print("="*70)
        
        # Print summary statistics
        if self.embodiment_history:
            scores = [e['score'] for e in self.embodiment_history]
            print(f"\nEmbodiment Summary:")
            print(f"  Mean: {np.mean(scores):.1f}")
            print(f"  Std:  {np.std(scores):.1f}")
            print(f"  Min:  {np.min(scores):.1f}")
            print(f"  Max:  {np.max(scores):.1f}")
    
    def _classify_gesture(self, leap_sample, bioradio_sample):
        """Classify current gesture (placeholder)"""
        # In real implementation:
        # features = extract_control_features(leap_sample, bioradio_sample)
        # gesture = self.control_model.predict(features)
        
        # Simulated gesture cycling
        gestures = ["Neutral", "Pinching", "Grasping", "Zipping"]
        idx = int(time.time() / 5) % len(gestures)
        return gestures[idx]
    
    def _compute_embodiment(self):
        """Compute embodiment score from buffered data (placeholder)"""
        if len(self.sensor_buffer['leap']) < 10:
            return None
        
        # In real implementation:
        # features = extract_embodiment_features(self.sensor_buffer)
        # score = self.embodiment_model.predict(features)
        
        # Simulated embodiment with trend
        base_score = 60 + 20 * np.sin(time.time() / 10)
        noise = 5 * np.random.randn()
        return np.clip(base_score + noise, 0, 100)
    
    def _display_status(self):
        """Display current system status"""
        # Create status bar
        embodiment_bar = self._create_bar(self.current_embodiment, 100)
        
        status = f"\r[Gesture: {self.current_gesture:12}] "
        status += f"[Embodiment: {self.current_embodiment:5.1f}/100 {embodiment_bar}]  "
        
        print(status, end='', flush=True)
    
    def _create_bar(self, value, max_value, width=20):
        """Create a text-based progress bar"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined prosthetic control and embodiment system")
    parser.add_argument("--control-model", type=str, 
                       default="../artifacts/prosthetic_control/model.pkl",
                       help="Path to prosthetic control model")
    parser.add_argument("--embodiment-model", type=str,
                       default="../artifacts/embodiment_model/embodiment_model.pkl",
                       help="Path to embodiment model")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CombinedPipeline(
        control_model_path=args.control_model,
        embodiment_model_path=args.embodiment_model
    )
    
    pipeline.start()


if __name__ == "__main__":
    main()