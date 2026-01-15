"""
Real-Time EMG Classification System

Orchestrates the complete real-time classification pipeline:
- Hardware connection (DelsysClient)
- Data processing (RealtimeProcessor)
- Prediction (RealtimeClassifier)

Updated to work with official Delsys AeroPy API.
"""

import time
import os
from threading import Thread
from pathlib import Path
from typing import Optional

from .delsys_client import DelsysClient, load_credentials
from .processor import RealtimeProcessor
from .classifier import RealtimeClassifier


class RealtimeEMGSystem:
    """
    Complete real-time EMG+IMU classification system.
    
    Usage:
        >>> system = RealtimeEMGSystem(
        ...     model_path="models/saved_models/model_latest/trained_model.pkl",
        ...     scaler_path="models/saved_models/model_latest/scaler.pkl"
        ... )
        >>> system.setup()
        >>> system.start()
    """
    
    def __init__(self, model_path: str, scaler_path: str, 
                 key: Optional[str] = None, license: Optional[str] = None):
        """
        Initialize system.
        
        Args:
            model_path: Path to trained_model.pkl
            scaler_path: Path to scaler.pkl
            key: Delsys API key (if None, loads from file)
            license: Delsys license (if None, loads from file)
        """
        # Verify model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load credentials if not provided
        if key is None or license is None:
            print("Loading credentials from files...")
            key, license = load_credentials()
        
        # Create Delsys client (official API)
        self.client = DelsysClient(key, license)
        
        # Create classifier
        self.classifier = RealtimeClassifier(model_path, scaler_path)
        
        # Processor (created after channel discovery)
        self.processor = None
        
        # State
        self.is_running = False
        self.poll_thread = None
    
    def setup(self) -> bool:
        """
        Connect to hardware and configure for streaming.
        
        This does:
        1. Connect to base station (ValidateBase)
        2. Scan for sensors (ScanSensors)
        3. Configure pipeline (Configure)
        4. Create processor with discovered channels
        
        Returns:
            True if setup successful
        """
        print("\n" + "="*70)
        print("SYSTEM SETUP")
        print("="*70)
        
        # Step 1: Connect to base station
        print("\n[1/4] Connecting to Trigno base station...")
        if not self.client.connect():
            print("✗ Connection failed")
            return False
        
        # Step 2: Scan for sensors
        print("\n[2/4] Scanning for paired sensors...")
        if not self.client.scan_sensors():
            print("✗ No sensors found")
            print("\nTroubleshooting:")
            print("  - Ensure sensors are powered on")
            print("  - Check sensors are paired to this base")
            print("  - Verify sensors are in range")
            return False
        
        # Step 3: Configure data collection
        print("\n[3/4] Configuring data collection pipeline...")
        if not self.client.configure():
            print("✗ Configuration failed")
            return False
        
        # Step 4: Create processor with discovered channels
        print("\n[4/4] Setting up data processor...")
        emg_guids = self.client.get_emg_channel_guids()
        imu_guids = self.client.get_imu_channel_guids()
        
        if len(emg_guids) == 0:
            print("✗ No EMG channels found!")
            print("   Check sensor modes are set correctly")
            return False
        
        print(f"  EMG channels: {len(emg_guids)}")
        print(f"  IMU channels: {len(imu_guids)}")
        
        self.processor = RealtimeProcessor(emg_guids, imu_guids)
        
        print("\n" + "="*70)
        print("✓ SETUP COMPLETE")
        print("="*70)
        print(f"Pipeline state: {self.client.get_pipeline_state()}")
        print(f"Ready to start streaming...")
        
        return True
    
    def _poll_loop(self):
        """
        Background thread to continuously poll data from sensors.
        
        This runs in a separate thread to avoid blocking the main
        classification loop.
        """
        while self.is_running:
            # Poll data from Delsys API
            data = self.client.poll_data()
            
            if data:
                # Add to processor buffers
                self.processor.add_data(data)
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
    
    def start(self, verbose: bool = True):
        """
        Start real-time classification.
        
        This:
        1. Starts data streaming from Delsys
        2. Launches background polling thread
        3. Runs main classification loop
        4. Displays predictions in real-time
        
        Args:
            verbose: If True, print predictions continuously
        
        Press Ctrl+C to stop.
        """
        if self.processor is None:
            print("✗ System not set up - call setup() first")
            return
        
        # Start streaming from Delsys
        print("\nStarting data stream...")
        if not self.client.start_streaming():
            print("✗ Failed to start streaming")
            return
        
        self.is_running = True
        
        # Start background polling thread
        self.poll_thread = Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        
        print("\n" + "="*70)
        print("REAL-TIME CLASSIFICATION ACTIVE")
        print("="*70)
        print("Press Ctrl+C to stop\n")
        
        if verbose:
            print("Waiting for first window...")
        
        # Main classification loop
        prediction_count = 0
        
        try:
            while self.is_running:
                # Check if we have enough data for a window
                if self.processor.is_window_ready():
                    # Extract features
                    features = self.processor.extract_window_features()
                    
                    if features is not None:
                        # Get prediction (with smoothing)
                        pred_label, pred_proba = self.classifier.predict_smoothed(features)
                        class_name = self.classifier.get_class_name(pred_label)
                        confidence = pred_proba[pred_label] * 100
                        
                        prediction_count += 1
                        
                        if verbose:
                            # Display prediction
                            print(
                                f"\r[{prediction_count:4d}] {class_name:12s} | "
                                f"Confidence: {confidence:5.1f}%",
                                end='',
                                flush=True
                            )
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
        
        print(f"\n✓ Classification complete: {prediction_count} predictions made")
    
    def stop(self):
        """Stop the real-time system"""
        self.is_running = False
        
        # Wait for polling thread to finish
        if self.poll_thread:
            self.poll_thread.join(timeout=1)
        
        # Stop Delsys streaming
        self.client.stop_streaming()
        
        print("✓ System stopped")
    
    def disconnect(self):
        """
        Disconnect and cleanup.
        
        Stops streaming if running, resets pipeline, and disconnects.
        """
        if self.is_running:
            self.stop()
        
        # Reset pipeline (Armed → Connected)
        self.client.reset_pipeline()
        
        # Disconnect
        self.client.disconnect()
        
        print("✓ Disconnected from hardware")
    
    def get_status(self) -> dict:
        """
        Get current system status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'is_running': self.is_running,
            'pipeline_state': self.client.get_pipeline_state(),
            'total_packets': self.client.get_total_packets(),
        }
        
        if self.processor:
            status['processor_status'] = self.processor.get_buffer_status()
        
        return status
    
    def print_status(self):
        """Print current system status"""
        status = self.get_status()
        
        print("\n" + "="*70)
        print("SYSTEM STATUS")
        print("="*70)
        print(f"Running: {status['is_running']}")
        print(f"Pipeline State: {status['pipeline_state']}")
        print(f"Total Packets: {status['total_packets']}")
        
        if 'processor_status' in status:
            proc = status['processor_status']
            print(f"\nProcessor:")
            print(f"  Window Ready: {proc['window_ready']}")
            print(f"  EMG Samples: {proc['emg_sample_count']}")
            print(f"  IMU Samples: {proc['imu_sample_count']}")
        
        print("="*70)


# Convenience function for quick usage
def run_realtime_classification(
    model_name: str = "model_latest",
    model_dir: str = "models/saved_models"
):
    """
    Convenience function to quickly start real-time classification.
    
    Args:
        model_name: Name of saved model folder
        model_dir: Directory containing saved models
        
    Example:
        >>> from src.realtime.system import run_realtime_classification
        >>> run_realtime_classification("model_latest")
    """
    # Build paths
    model_path = os.path.join(model_dir, model_name, "trained_model.pkl")
    scaler_path = os.path.join(model_dir, model_name, "scaler.pkl")
    
    # Create and run system
    system = RealtimeEMGSystem(model_path, scaler_path)
    
    if system.setup():
        system.start()
        system.disconnect()
    else:
        print("\n✗ Setup failed - cannot start classification")


# Test/demo code
if __name__ == "__main__":
    """
    Test the system with a saved model.
    
    Run: python -m src.realtime.system
    """
    import sys
    
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "model_latest"
    
    print("="*70)
    print("REAL-TIME EMG CLASSIFICATION SYSTEM TEST")
    print("="*70)
    print(f"Model: {model_name}\n")
    
    try:
        run_realtime_classification(model_name)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nAvailable models:")
        model_dir = "models/saved_models"
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                print(f"  - {item}")
        else:
            print(f"  No models directory found: {model_dir}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()