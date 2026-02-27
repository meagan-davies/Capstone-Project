#!/usr/bin/env python
"""
Real-Time EMG Classification Script - Refactored

Usage:
    python scripts/realtime_classify.py --model-name model_latest
"""

import argparse
import sys
import time
from pathlib import Path
from threading import Thread, Lock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time EMG classification")
    parser.add_argument('--model-name', type=str, default='model_latest', help='Saved model folder name')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory containing saved models')
    parser.add_argument('--window-sec', type=float, default=0.2, help='Sliding window size in seconds')
    parser.add_argument('--overlap-sec', type=float, default=0.1, help='Sliding window overlap in seconds')
    parser.add_argument('--fs-emg', type=float, default=963.0, help='EMG sampling frequency (Hz)')
    parser.add_argument('--fs-imu', type=float, default=148.148, help='IMU sampling frequency (Hz)')
    parser.add_argument('--skip-diagnostics', action='store_true', help='Skip the channel diagnostic pause')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output from processor')
    return parser.parse_args()


def main():
    args = parse_args()

    # Construct model paths
    model_dir = Path(args.model_dir) / args.model_name
    model_path = model_dir / "trained_model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print(f"✗ Model or scaler not found in {model_dir}")
        if model_dir.exists():
            print("Available models:")
            for item in sorted(model_dir.parent.iterdir()):
                if item.is_dir():
                    print(f"  - {item.name}")
        sys.exit(1)

    print("="*70)
    print("REAL-TIME EMG CLASSIFICATION")
    print("="*70)

    # Step 1: Load credentials
    print("\nStep 1: Loading credentials...")
    try:
        KEY, LICENSE = load_credentials()
        print("✓ Credentials loaded")
    except Exception as e:
        print(f"✗ Error loading credentials: {e}")
        sys.exit(1)

    # Step 2: Connect to Delsys hardware
    print("\nStep 2: Connecting to Delsys system...")
    client = DelsysClient(KEY, LICENSE)
    if not client.connect():
        print("✗ Connection failed")
        sys.exit(1)

    # Step 3: Scan sensors
    print("\nStep 3: Scanning for sensors...")
    if not client.scan_sensors():
        print("✗ No sensors found")
        client.disconnect()
        sys.exit(1)

    # Step 4: Configure data collection
    print("\nStep 4: Configuring data collection...")
    if not client.configure():
        print("✗ Configuration failed")
        client.disconnect()
        sys.exit(1)

    # Channel diagnostic (optional pause)
    print("\n" + "="*70)
    print("CHANNEL DIAGNOSTIC")
    print("="*70)
    print(f"Total channels in hardware: {len(client.channel_info)}")
    type_counts = {}
    for info in client.channel_info.values():
        chan_type = info['type']
        type_counts[chan_type] = type_counts.get(chan_type, 0) + 1
    print("\nChannel types:")
    for chan_type, count in sorted(type_counts.items()):
        print(f"  {chan_type:20s}: {count}")
    print(f"\nTotal channels: {sum(type_counts.values())}")
    print("="*70)

    if not args.skip_diagnostics:
        input("\nPress Enter to continue...")

    # Step 5: Create processor
    print("\nStep 5: Setting up data processor...")
    processor = RealtimeProcessor(
        delsys_client=client,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        fs_emg=args.fs_emg,
        fs_imu=args.fs_imu,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        validate_features=True,
    )

    print(f"✓ Processor ready")
    print(f"  EMG channels: {len(processor.emg_channel_order)}")
    print(f"  IMU channels: {len(processor.imu_channel_order)}")

    # Step 6: Start streaming
    print("\nStep 6: Starting data stream...")
    if not client.start_streaming():
        print("✗ Failed to start streaming")
        client.disconnect()
        sys.exit(1)

    # Step 7: Polling thread
    is_running = True
    latest_packet = None
    packet_lock = Lock()

    def poll_loop():
        nonlocal latest_packet
        while is_running:
            packet = client.poll_data()
            if packet:
                with packet_lock:
                    latest_packet = packet
                processor.add_raw_data(packet)
            time.sleep(0.001)

    poll_thread = Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    # Step 8: Main real-time loop
    print("\n" + "="*70)
    print("REAL-TIME CLASSIFICATION ACTIVE")
    print("="*70)
    print("Press Ctrl+C to stop\n")

    prediction_count = 0
    try:
        while True:
            result = processor.predict()
            if result:
                pred_class, pred_probs, class_name = result
                confidence = pred_probs[pred_class] * 100
                prediction_count += 1

                print(f"[{prediction_count:4d}] {class_name:12s} | Confidence: {confidence:5.1f}%")
                if args.verbose:
                    status = processor.get_buffer_status()
                    print(f"  Buffer status: {status}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping real-time classification...")

    # Cleanup
    is_running = False
    poll_thread.join()
    client.stop_streaming()
    client.disconnect()
    print(f"\n✓ Session complete: {prediction_count} predictions made")
    print("="*70)


if __name__ == "__main__":
    main()