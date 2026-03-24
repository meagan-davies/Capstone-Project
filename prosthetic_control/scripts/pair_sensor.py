#!/usr/bin/env python
"""
Pair sensors to the base station and show connected sensors

Usage:
    python scripts/pair_sensor.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
import time


def show_paired_sensors(client):
    """Display all currently paired sensors"""
    print("\n" + "="*70)
    print("CURRENTLY PAIRED SENSORS")
    print("="*70)
    
    try:
        # Try to get paired sensors
        paired_sensors = list(client.base.GetPairedSensors())
        
        if len(paired_sensors) == 0:
            print("\n⚠️  No sensors currently paired")
            return 0
        
        print(f"\n✓ Found {len(paired_sensors)} paired sensor(s):\n")
        
        for i, sensor in enumerate(paired_sensors, 1):
            try:
                pair_num = sensor.PairNumber
                sid = sensor.Properties.Sid
                mode = sensor.Configuration.ModeString
                
                # Try to get firmware version if available
                try:
                    fw_version = sensor.Properties.FirmwareVersion
                    print(f"  {i}. Pair #{pair_num} | SID: {sid} | Mode: {mode} | FW: {fw_version}")
                except:
                    print(f"  {i}. Pair #{pair_num} | SID: {sid} | Mode: {mode}")
                
            except Exception as e:
                print(f"  {i}. Sensor info unavailable: {e}")
        
        return len(paired_sensors)
        
    except Exception as e:
        print(f"\n⚠️  Could not retrieve paired sensors: {e}")
        print("  This is normal if no sensors are paired yet")
        return 0


def scan_for_sensors(client):
    """Scan for active sensors in range"""
    print("\n" + "="*70)
    print("SCANNING FOR ACTIVE SENSORS")
    print("="*70)
    print("\nScanning... (this may take a few seconds)")
    
    try:
        # Scan for sensors
        client.scan_sensors()
        
        if len(client.sensors) == 0:
            print("\n⚠️  No active sensors found")
            print("\nTips:")
            print("  - Make sure sensors are powered on (remove from charger)")
            print("  - Press the button on each sensor to wake them up")
            print("  - Ensure sensors are in range of the base station")
            return False
        
        print(f"\n✓ Found {len(client.sensors)} active sensor(s):\n")
        
        for i, sensor in enumerate(client.sensors, 1):
            try:
                pair_num = sensor.PairNumber
                sid = sensor.Properties.Sid
                mode = sensor.Configuration.ModeString
                print(f"  {i}. Pair #{pair_num} | SID: {sid} | Mode: {mode}")
            except Exception as e:
                print(f"  {i}. Sensor info unavailable: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Scan error: {e}")
        return False


def pair_new_sensor(client, pair_number):
    """Pair a new sensor"""
    print(f"\nPairing sensor #{pair_number}...")
    print("  Touch sensor to magnet now...")
    
    try:
        # Start pairing
        client.base.PairSensor(pair_number)
        
        # Wait for pairing to complete
        timeout = 30  # 30 second timeout
        elapsed = 0
        
        while client.base.CheckPairStatus() and elapsed < timeout:
            time.sleep(0.5)
            elapsed += 0.5
            print(".", end="", flush=True)
        
        if elapsed >= timeout:
            print(f"\n✗ Pairing timed out after {timeout}s")
            return False
        
        print(f"\n✓ Sensor #{pair_number} paired successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Pairing error: {e}")
        return False


def main():
    print("="*70)
    print("DELSYS SENSOR PAIRING & STATUS")
    print("="*70)
    
    # Load credentials
    print("\nStep 1: Loading credentials...")
    try:
        KEY, LICENSE = load_credentials()
        print("✓ Credentials loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Create client
    client = DelsysClient(KEY, LICENSE)
    
    # Connect
    print("\nStep 2: Connecting to base station...")
    if not client.connect():
        print("✗ Connection failed")
        return
    
    # Show currently paired sensors
    num_paired = show_paired_sensors(client)
    
    # Scan for active sensors
    scan_success = scan_for_sensors(client)
    
    # Ask user what to do
    print("\n" + "="*70)
    print("OPTIONS")
    print("="*70)
    print("1. Pair new sensor(s)")
    print("2. Exit")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice != "1":
        print("\nExiting without pairing")
        client.disconnect()
        return
    
    # Pairing mode
    print("\n" + "="*70)
    print("PAIRING MODE")
    print("="*70)
    print("\nInstructions:")
    print("1. Remove sensor from charger (LED will blink GREEN)")
    print("2. Touch sensor to pairing magnet (LED will turn RED)")
    print("3. Wait for pairing to complete (LED will turn solid GREEN)")
    print("4. Repeat for each sensor")
    print("\nPress Ctrl+C to stop pairing\n")
    
    sensor_count = num_paired
    new_pairs = 0
    
    try:
        while True:
            # Start pairing for next sensor
            pair_number = sensor_count + 1
            
            if pair_new_sensor(client, pair_number):
                sensor_count += 1
                new_pairs += 1
            
            # Ask if more sensors
            response = input("\nPair another sensor? (y/n): ").strip().lower()
            if response != 'y':
                break
    
    except KeyboardInterrupt:
        print("\n\nPairing stopped by user")
    
    # Show final status
    client.disconnect()
    
    print("\n" + "="*70)
    print("PAIRING COMPLETE")
    print("="*70)
    print(f"  Previously paired: {num_paired}")
    print(f"  Newly paired:      {new_pairs}")
    print(f"  Total paired:      {sensor_count}")
    print("="*70)
    
    if sensor_count > 0:
        print("\n✓ Ready to collect data!")
        print("  Run: python scripts/realtime_classify.py")
    else:
        print("\n⚠️  No sensors paired")


if __name__ == "__main__":
    main()