import argparse
import sys
import os
from pathlib import Path
import time
from threading import Thread, Event, Lock
import numpy as np

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time EMG classification")
    parser.add_argument('--model-name', type=str, default='model_latest', help='Name of saved model to use')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory containing saved models')
    return parser.parse_args()


def main():
    args = parse_args()

    # Paths
    model_path = os.path.join(args.model_dir, args.model_name, "trained_model.pkl")
    scaler_path = os.path.join(args.model_dir, args.model_name, "scaler.pkl")

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        exit(1)

    print("\n=== REAL-TIME EMG CLASSIFICATION ===\n")

    # Load credentials
    KEY, LICENSE = load_credentials()
    print("✓ Credentials loaded")

    # Connect to hardware
    client = DelsysClient(KEY, LICENSE)
    if not client.connect():
        print("✗ Connection failed")
        exit(1)

    # Scan sensors
    if not client.scan_sensors():
        print("✗ No sensors found")
        client.disconnect()
        exit(1)

    # Configure
    if not client.configure():
        print("✗ Configuration failed")
        client.disconnect()
        exit(1)

    # Create processor
    processor = RealtimeProcessor(client, model_path=model_path, scaler_path=scaler_path)

    # Start streaming
    if not client.start_streaming():
        print("✗ Failed to start streaming")
        client.disconnect()
        exit(1)

    # Setup polling thread
    is_running = Event()
    is_running.set()
    packet_lock = Lock()
    latest_packet = {}

    def poll_loop():
        """Poll Delsys data and add to processor"""
        nonlocal latest_packet
        while is_running.is_set():
            data = client.poll_data()
            if data:
                # Debug: show what is being polled
                if prediction_count % 50 == 0:
                    print("Polling active...")
         
                # Check if UUIDs match processor mapping
                for uuid in data:
                    if uuid not in processor.uuid_to_emg_sensor and uuid not in sum(processor.imu_sensor_map.values(), []):
                        print(f"⚠ Unmapped UUID: {uuid}")

                with packet_lock:
                    latest_packet = data
                processor.add_raw_data(data)
            time.sleep(0.005)  # ~200 Hz polling

    poll_thread = Thread(target=poll_loop, daemon=True)
    poll_thread.start()


    # Main loop
    try:
        prediction_count = 0
        print("Press Ctrl+C to stop...\n")
        while True:
            status = processor.get_buffer_status()
            if status['emg_sample_count'] >= processor.emg_win_size and status['imu_sample_count'] >= processor.imu_win_size:
                label, proba = processor.predict_current_window()
                if label is not None:
                    confidence = np.max(proba) * 100
                    prediction_count += 1
                    print(f"[{prediction_count:4d}] {label:12s} ({confidence:5.1f}%)")
            else:
                # Optional: show buffer fill
                print(f"Buffer filling: EMG {status['emg_sample_count']}/{processor.emg_win_size}, "
                      f"IMU {status['imu_sample_count']}/{processor.imu_win_size}", end='\r')
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopping...")

    # Cleanup
    is_running.clear()
    poll_thread.join(timeout=1)
    client.stop_streaming()
    client.disconnect()
    print(f"\n✓ Session complete: {prediction_count} predictions made")


if __name__ == "__main__":
    main()
