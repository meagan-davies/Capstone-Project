#!/usr/bin/env python
"""
Real-Time EMG Classification Script

Usage:
    python scripts/realtime_classify.py --model-name model_latest
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor
from src.realtime.classifier import RealtimeClassifier
import time
from threading import Thread


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time EMG classification")
    parser.add_argument(
        '--model-name',
        type=str,
        default='model_latest',
        help='Name of saved model to use'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/',
        help='Directory containing saved models'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Construct model paths
    model_path = os.path.join(args.model_dir, args.model_name, "trained_model.pkl")
    scaler_path = os.path.join(args.model_dir, args.model_name, "scaler.pkl")
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print(f"\nAvailable models:")
        if os.path.exists(args.model_dir):
            for item in os.listdir(args.model_dir):
                print(f"  - {item}")
        exit(1)
    
    print("="*70)
    print("REAL-TIME EMG CLASSIFICATION")
    print("="*70)
    
    # ========== STEP 1: Load Credentials ==========
    print("\nStep 1: Loading credentials...")
    try:
        KEY, LICENSE = load_credentials()
        print(f"✓ Credentials loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
    
    # ========== STEP 2: Connect to Hardware ==========
    print("\nStep 2: Connecting to Delsys system...")
    client = DelsysClient(KEY, LICENSE)
    
    if not client.connect():
        print("✗ Connection failed")
        exit(1)
    
    # ========== STEP 3: Scan Sensors ==========
    print("\nStep 3: Scanning for sensors...")
    if not client.scan_sensors():
        print("✗ No sensors found")
        client.disconnect()
        exit(1)
    
    # ========== STEP 4: Configure ==========
    print("\nStep 4: Configuring data collection...")
    if not client.configure():
        print("✗ Configuration failed")
        client.disconnect()
        exit(1)
    
    # ========== STEP 5: Create Processor ==========
    print("\nStep 5: Setting up data processor...")
    emg_guids = client.get_emg_channel_guids()
    imu_guids = client.get_imu_channel_guids()
    
    print(f"  EMG channels: {len(emg_guids)}")
    print(f"  IMU channels: {len(imu_guids)}")
    
    processor = RealtimeProcessor(emg_guids, imu_guids)
    
    # ========== STEP 6: Load Classifier ==========
    print("\nStep 6: Loading trained model...")
    classifier = RealtimeClassifier(model_path, scaler_path)
    
    # ========== STEP 7: Start Streaming ==========
    print("\nStep 7: Starting data stream...")
    if not client.start_streaming():
        print("✗ Failed to start streaming")
        client.disconnect()
        exit(1)
    
    # ========== STEP 8: Real-Time Loop ==========
    print("\n" + "="*70)
    print("REAL-TIME CLASSIFICATION ACTIVE")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    is_running = True
    
    def poll_loop():
        """Background thread to continuously poll data"""
        while is_running:
            data = client.poll_data()
            if data:
                processor.add_data(data)
            time.sleep(0.001)
    
    # Start polling thread
    poll_thread = Thread(target=poll_loop, daemon=True)
    poll_thread.start()
    
    # Main classification loop
    try:
        prediction_count = 0
        
        while True:
            if processor.is_window_ready():
                features = processor.extract_window_features()
                
                if features is not None:
                    pred_label, pred_proba = classifier.predict(features)
                    class_name = classifier.get_class_name(pred_label)
                    confidence = pred_proba[pred_label] * 100
                    
                    prediction_count += 1
                    
                    # Display prediction
                    print(
                        f"\r[{prediction_count:4d}] {class_name:12s} | "
                        f"Confidence: {confidence:5.1f}%", 
                        end='', 
                        flush=True
                    )
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
        is_running = False
    
    # ========== CLEANUP ==========
    print("\nCleaning up...")
    poll_thread.join(timeout=1)
    client.stop_streaming()
    client.disconnect()
    
    print(f"\n✓ Session complete: {prediction_count} predictions made")
    print("="*70)


if __name__ == "__main__":
    main()