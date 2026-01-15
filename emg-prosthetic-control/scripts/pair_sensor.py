#!/usr/bin/env python
"""
Pair sensors to the base station

Usage:
    python scripts/pair_sensors.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
import time

def main():
    print("="*70)
    print("DELSYS SENSOR PAIRING")
    print("="*70)
    
    # Load credentials
    print("\nLoading credentials...")
    KEY, LICENSE = load_credentials()
    print("✓ Credentials loaded")
    
    # Create client
    client = DelsysClient(KEY, LICENSE)
    
    # Connect
    print("\nConnecting to base station...")
    if not client.connect():
        print("✗ Connection failed")
        return
    
    # Pair sensors
    print("\n" + "="*70)
    print("PAIRING MODE")
    print("="*70)
    print("\nInstructions:")
    print("1. Remove sensor from charger (it will show green LED)")
    print("2. Touch sensor to magnet again (it will turn RED)")
    print("3. Wait for pairing to complete (LED will turn GREEN)")
    print("4. Repeat for each sensor")
    print("\nPress Ctrl+C when done pairing all sensors\n")
    
    sensor_count = 0
    
    try:
        while True:
            # Start pairing for next sensor
            pair_number = sensor_count + 1
            print(f"\nPairing sensor #{pair_number}...")
            print("  Touch sensor to magnet now...")
            
            # Call pair method
            try:
                # Note: You may need to adjust this based on actual API
                client.base.PairSensor(pair_number)
                
                # Wait for pairing to complete
                while client.base.CheckPairStatus():
                    time.sleep(0.5)
                    print(".", end="", flush=True)
                
                print(f"\n✓ Sensor #{pair_number} paired successfully!")
                sensor_count += 1
                
                # Ask if more sensors
                response = input("\nPair another sensor? (y/n): ")
                if response.lower() != 'y':
                    break
                    
            except Exception as e:
                print(f"\n✗ Pairing error: {e}")
                break
    
    except KeyboardInterrupt:
        print("\n\nPairing stopped by user")
    
    # Disconnect
    client.disconnect()
    
    print(f"\n{'='*70}")
    print(f"✓ Paired {sensor_count} sensor(s)")
    print("="*70)
    print("\nNow run: python scripts/realtime_classify.py")

if __name__ == "__main__":
    main()