#!/usr/bin/env python
"""
Real-Time EMG Classification with Motor Control

Usage:
    # With motor control
    python scripts/realtime_classify.py --model-name model_latest_v1
    
    # Specify Arduino port
    python scripts/realtime_classify.py --motor-port COM3
    
    # Without motor (testing only)
    python scripts/realtime_classify.py --no-motor
"""

import argparse
import sys
import time
from pathlib import Path
from threading import Thread, Lock
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
from src.realtime.processor import RealtimeProcessor
from src.realtime.motor_controller import MotorController


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time EMG classification with motor control")
    parser.add_argument('--model-name',       type=str,   default='model_latest_v1')
    parser.add_argument('--window-sec',       type=float, default=0.2)
    parser.add_argument('--overlap-sec',      type=float, default=0.1)
    parser.add_argument('--fs-imu',           type=float, default=148.1481)
    parser.add_argument('--skip-diagnostics', action='store_true')
    parser.add_argument('--verbose',          action='store_true')
    parser.add_argument('--motor-port',       type=str,   default=None, 
                       help='Arduino serial port (e.g., COM3 or /dev/ttyACM0)')
    parser.add_argument('--no-motor',         action='store_true', 
                       help='Disable motor control (testing only)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("REAL-TIME EMG CLASSIFICATION")
    print("=" * 70)

    # ========== MOTOR SETUP ==========
    motor = None
    if not args.no_motor:
        print("\n[MOTOR] Initializing motor controller...")
        motor = MotorController(port=args.motor_port)
        
        if not motor.connect():
            print("\n⚠️  Motor connection failed")
            response = input("Continue without motor? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
            motor = None
        else:
            print("✓ Motor controller ready")
    else:
        print("\n⚠️  Motor control disabled (--no-motor)")

    # ========== DELSYS SETUP ==========
    # Step 1: Load credentials
    print("\nStep 1: Loading credentials...")
    try:
        KEY, LICENSE = load_credentials()
        print("✓ Credentials loaded")
    except Exception as e:
        print(f"✗ Error loading credentials: {e}")
        if motor:
            motor.disconnect()
        sys.exit(1)

    # Step 2: Connect to Delsys hardware
    print("\nStep 2: Connecting to Delsys system...")
    client = DelsysClient(KEY, LICENSE)
    if not client.connect():
        print("✗ Connection failed")
        if motor:
            motor.disconnect()
        sys.exit(1)

    # Step 3: Scan sensors
    print("\nStep 3: Scanning for sensors...")
    if not client.scan_sensors():
        print("✗ No sensors found")
        client.disconnect()
        if motor:
            motor.disconnect()
        sys.exit(1)

    # Step 4: Configure data collection
    print("\nStep 4: Configuring data collection...")
    if not client.configure():
        print("✗ Configuration failed")
        client.disconnect()
        if motor:
            motor.disconnect()
        sys.exit(1)

    # Channel diagnostic
    print("\n" + "=" * 70)
    print("CHANNEL DIAGNOSTIC")
    print("=" * 70)
    type_counts = defaultdict(int)
    for info in client.channel_info.values():
        type_counts[info['type']] += 1
    for chan_type, count in sorted(type_counts.items()):
        print(f"  {chan_type:20s}: {count}")
    print(f"  Total channels: {sum(type_counts.values())}")
    print("=" * 70)
    
    if not args.skip_diagnostics:
        input("\nPress Enter to continue...")

    # Step 5: Initialize processor
    print("\nStep 5: Setting up data processor...")
    model_path = Path("prosthetic_control/models") / args.model_name
    try:
        processor = RealtimeProcessor(
            delsys_client=client,
            model_path=model_path,
            fs_imu=args.fs_imu,
            window_sec=args.window_sec,
            overlap_sec=args.overlap_sec,
            debug=args.verbose
        )
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        client.disconnect()
        if motor:
            motor.disconnect()
        sys.exit(1)

    print(f"✓ Processor ready")

    # Step 6: Start streaming
    print("\nStep 6: Starting data stream...")
    if not client.start_streaming():
        print("✗ Failed to start streaming")
        client.disconnect()
        if motor:
            motor.disconnect()
        sys.exit(1)

    # Polling thread
    is_running = True

    def poll_loop():
        while is_running:
            packet = client.poll_data()
            if packet:
                processor.add_raw_data(packet)
            time.sleep(0.001)

    poll_thread = Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    # Step 7: Main classification loop
    print("\n" + "=" * 70)
    print("REAL-TIME CLASSIFICATION ACTIVE")
    if motor:
        print("🤖 MOTOR CONTROL ENABLED")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")

    prediction_count = 0
    try:
        while True:
            result = processor.predict()
            if result:
                pred_class, pred_probs, class_name = result
                confidence = pred_probs[pred_class] * 100
                prediction_count += 1
                
                # Display prediction
                motor_status = ""
                if motor:
                    # Send to motor
                    if motor.send_gesture(class_name):
                        motor_status = " → 🤖 Motor"
                    else:
                        motor_status = " → ⚠️ Motor error"
                
                print(f"[{prediction_count:4d}] {class_name:12s} | {confidence:5.1f}%{motor_status}")
                
                if args.verbose:
                    buf_status = processor.get_buffer_status()
                    print(f"         Buffer: EMG={buf_status['emg_sample_count']}, "
                          f"IMU={buf_status['imu_sample_count']}")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopping real-time classification...")

    # Cleanup
    is_running = False
    poll_thread.join(timeout=1)
    client.stop_streaming()
    client.disconnect()
    
    if motor:
        print("\nReturning motor to neutral...")
        motor.send_gesture("Neutral")
        time.sleep(0.5)
        motor.disconnect()
    
    print(f"\n✓ Session complete: {prediction_count} predictions made")
    print("=" * 70)


if __name__ == "__main__":
    main()