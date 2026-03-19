#!/usr/bin/env python
"""
Motor Test Mode - Cycles through gestures without sensors

Usage:
    python scripts/motor_test.py --motor-port COM3
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.motor_controller import MotorController


def parse_args():
    parser = argparse.ArgumentParser(description="Test motor by cycling through gestures")
    parser.add_argument('--motor-port', type=str, default=None, 
                       help='Arduino serial port (e.g., COM3 or /dev/ttyACM0)')
    parser.add_argument('--delay',      type=float, default=5.0,
                       help='Seconds between gestures (default: 3)')
    parser.add_argument('--cycles',     type=int, default=0,
                       help='Number of cycles (0 = infinite)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("MOTOR TEST MODE - Gesture Cycling")
    print("=" * 70)

    # Connect to motor
    print("\nConnecting to Arduino...")
    motor = MotorController(port=args.motor_port)
    
    if not motor.connect():
        print("✗ Motor connection failed")
        sys.exit(1)
    
    print("✓ Motor connected")
    
    # Gesture sequence
    gestures = ["Neutral", "Pinching", "Grasping", "Zipping"]
    
    print(f"\nStarting gesture cycle (delay: {args.delay}s)")
    print("Press Ctrl+C to stop\n")
    
    cycle_count = 0
    gesture_count = 0
    
    try:
        while True:
            for gesture in gestures:
                # Send gesture
                print(f"[{gesture_count:4d}] {gesture:12s} → ", end='', flush=True)
                
                if motor.send_gesture(gesture):
                    print("✓ Sent")
                else:
                    print("✗ Failed")
                
                gesture_count += 1
                time.sleep(args.delay)
            
            cycle_count += 1
            print(f"\nCycle {cycle_count} complete")
            
            # Check if we should stop
            if args.cycles > 0 and cycle_count >= args.cycles:
                break
            
            print()  # Blank line between cycles
    
    except KeyboardInterrupt:
        print("\n\nStopping test...")
    
    # Return to neutral and disconnect
    print("\nReturning to neutral...")
    motor.send_gesture("Neutral")
    time.sleep(1)
    motor.disconnect()
    
    print(f"\n✓ Test complete: {gesture_count} gestures sent, {cycle_count} cycles")
    print("=" * 70)


if __name__ == "__main__":
    main()