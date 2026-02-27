#!/usr/bin/env python
"""
Real-Time EMG Classification Script - Sensor-Based (Updated)

Usage:
    python scripts/realtime_classify.py --model-name model_latest_v1
"""

import argparse
import sys
import time
from pathlib import Path
from threading import Thread, Lock
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time EMG classification")
    parser.add_argument('--model-name', type=str, default='model_latest_v1', help='Model bundle folder name')
    parser.add_argument('--window-sec', type=float, default=0.206, help='Sliding window size in seconds')
    parser.add_argument('--overlap-sec', type=float, default=0.1, help='Sliding window overlap in seconds')
    parser.add_argument('--fs-emg', type=float, default=963.0, help='EMG sampling frequency (Hz)')
    parser.add_argument('--fs-imu', type=float, default=148.148, help='IMU sampling frequency (Hz)')
    parser.add_argument('--skip-diagnostics', action='store_true', help='Skip channel diagnostic pause')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output from processor')
    return parser.parse_args()


def main():
    args = parse_args()

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

    # Channel diagnostic
    print("\n" + "="*70)
    print("CHANNEL DIAGNOSTIC")
    print("="*70)
    type_counts = defaultdict(int)
    for info in client.channel_info.values():
        chan_type = info['type']
        type_counts[chan_type] += 1
    for chan_type, count in sorted(type_counts.items()):
        print(f"  {chan_type:20s}: {count}")
    print(f"  Total channels: {sum(type_counts.values())}")
    print("="*70)
    if not args.skip_diagnostics:
        input("\nPress Enter to continue...")

    # Step 5: Initialize processor
    print("\nStep 5: Setting up data processor...")
    model_path = Path("models") / args.model_name
    try:
        processor = RealtimeProcessor(
            delsys_client=client,
            model_path=model_path,
            fs_emg=args.fs_emg,
            fs_imu=args.fs_imu,
            window_sec=args.window_sec,
            overlap_sec=args.overlap_sec
        )
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        client.disconnect()
        sys.exit(1)

    print(f"✓ Processor ready")
    print(f"  EMG sensors: {len(processor.emg_sensor_map)} ({len(processor.emg_channel_order)} channels)")
    print(f"  IMU sensors: {len(processor.imu_sensor_map)} ({len(processor.imu_channel_order)} channels)")

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
                    buf_status = processor.get_buffer_status()
                    print(f"  Buffers -> EMG: {buf_status['emg_buffer_fill']}, IMU: {buf_status['imu_buffer_fill']}, Window ready: {buf_status['window_ready']}")
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