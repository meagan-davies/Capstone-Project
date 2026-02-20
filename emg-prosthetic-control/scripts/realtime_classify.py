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
from threading import Thread, Lock
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor
from src.realtime.classifier import RealtimeClassifier


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

    # ✓ Configuration succeeded - NOW run diagnostic
    print("\n" + "="*70)
    print("COMPLETE CHANNEL DIAGNOSTIC")
    print("="*70)
    print(f"Total channels in hardware: {len(client.channel_info)}")

    # Group by type
    type_counts = {}
    for guid, info in client.channel_info.items():
        chan_type = info['type']
        type_counts[chan_type] = type_counts.get(chan_type, 0) + 1

    print("\nChannel types:")
    for chan_type, count in sorted(type_counts.items()):
        print(f"  {chan_type:20s}: {count}")

    print(f"\nTotal: {sum(type_counts.values())}")
    print("="*70)

    # PAUSE HERE - Share this output with me!
    input("\nPress Enter to continue...")

    
    # ========== STEP 5: Create Processor ==========
    print("\nStep 5: Setting up data processor...")

    processor = RealtimeProcessor(
        delsys_client=client,          # Delsys client object
        model_path=model_path,         # trained model .pkl
        scaler_path=scaler_path,       # scaler .pkl
        fs_emg=963,
        fs_imu=148.148,
        window_sec=0.2,
        overlap_sec=0.1
    )

    print(f"  EMG channels: {len(processor.emg_channel_order)}")
    print(f"  IMU channels: {len(processor.imu_channel_order)}")

    # ========== STEP 6: Start Streaming ==========
    print("\nStep 6: Starting data stream...")
    if not client.start_streaming():
        print("✗ Failed to start streaming")
        client.disconnect()
        exit(1)

    # ========== STEP 7: Real-Time Polling Thread ==========
    is_running = True

    latest_packet = None
    packet_lock = Lock()

    def poll_loop():
        """Continuously poll data from Delsys client and push to processor"""
        nonlocal latest_packet
        while is_running:
            packet = client.poll_data()
            if packet:
                with packet_lock:
                    latest_packet = packet
                processor.add_raw_data(packet)  # ✅ correct method
            time.sleep(0.001)  # small sleep to prevent CPU spin

    # Start polling thread
    poll_thread = Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    # ========== STEP 8: Main Real-Time Loop ==========
    print("\n" + "="*70)
    print("REAL-TIME CLASSIFICATION ACTIVE")
    print("="*70)
    print("Press Ctrl+C to stop\n")

    prediction_count = 0

    try:
        while True:
            if processor.is_window_ready():
                result = processor.predict()
                if result is not None:
                    pred_class, pred_probs, class_name = result
                    confidence = pred_probs[pred_class] * 100
                    prediction_count += 1

                    print(f"\n[{prediction_count:4d}] {class_name:12s} | Confidence: {confidence:5.1f}%")

                    # Optional: print raw packet that produced this window
                    # with packet_lock:
                    #     packet = latest_packet
                    # if packet is not None:
                    #     client.describe_packet(packet)

            time.sleep(0.5)  # adjust for responsiveness

    except KeyboardInterrupt:
        print("\n\nStopping real-time classification...")
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