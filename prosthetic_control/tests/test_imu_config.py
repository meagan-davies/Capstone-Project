#!/usr/bin/env python
"""
Test and Fix IMU Channel Enablement

This script will:
1. Connect to Delsys
2. Scan sensors
3. Show which channels are enabled by default
4. Enable ALL channels (including IMU)
5. Re-configure and show the difference

Usage:
    python scripts/test_imu_channels.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime.delsys_client import DelsysClient, load_credentials
import time

def show_channel_status(client):
    """Show current channel status"""
    print("\nCurrent channel status:")
    
    type_counts = {'EMG': 0, 'ACC': 0, 'GYRO': 0, 'Other': 0}
    enabled_counts = {'EMG': 0, 'ACC': 0, 'GYRO': 0, 'Other': 0}
    
    for sensor_idx, sensor in enumerate(client.sensors):
        channels = list(sensor.TrignoChannels)
        print(f"\nSensor {sensor_idx + 1}:")
        
        for channel in channels:
            chan_type = str(channel.Type)
            name = channel.Name
            is_enabled = channel.IsEnabled
            
            # Count by type
            if chan_type == 'EMG':
                type_counts['EMG'] += 1
                if is_enabled:
                    enabled_counts['EMG'] += 1
            elif chan_type == 'ACC':
                type_counts['ACC'] += 1
                if is_enabled:
                    enabled_counts['ACC'] += 1
            elif chan_type == 'GYRO':
                type_counts['GYRO'] += 1
                if is_enabled:
                    enabled_counts['GYRO'] += 1
            else:
                type_counts['Other'] += 1
                if is_enabled:
                    enabled_counts['Other'] += 1
            
            status = "✓ ENABLED" if is_enabled else "✗ DISABLED"
            print(f"  {name:12s} ({chan_type:10s}): {status}")
    
    print(f"\nSummary:")
    print(f"  EMG:   {enabled_counts['EMG']}/{type_counts['EMG']} enabled")
    print(f"  ACC:   {enabled_counts['ACC']}/{type_counts['ACC']} enabled")
    print(f"  GYRO:  {enabled_counts['GYRO']}/{type_counts['GYRO']} enabled")
    print(f"  Other: {enabled_counts['Other']}/{type_counts['Other']} enabled")
    
    return type_counts, enabled_counts


def enable_all_channels(client):
    """Enable ALL channels including IMU"""
    print("\nEnabling ALL channels (EMG + IMU + other)...")
    
    for i in range(len(client.sensors)):
        sensor = client.sensors[i]
        channels = list(sensor.TrignoChannels)
        
        for channel in channels:
            channel.IsEnabled = True
        
        print(f"  Sensor {i+1}: Enabled {len(channels)} channels")
    
    print("✓ All channels enabled")


def main():
    print("="*70)
    print("IMU CHANNEL ENABLEMENT TEST")
    print("="*70)
    
    # Load credentials
    print("\nStep 1: Loading credentials...")
    KEY, LICENSE = load_credentials()
    print("✓ Credentials loaded")
    
    # Connect
    print("\nStep 2: Connecting to base station...")
    client = DelsysClient(KEY, LICENSE)
    if not client.connect():
        print("✗ Connection failed")
        return
    
    # Scan sensors (default behavior)
    print("\nStep 3: Scanning sensors...")
    if not client.scan_sensors():
        print("✗ No sensors found")
        return
    
    # Show default channel status
    print("\n" + "="*70)
    print("DEFAULT CHANNEL STATUS (before fix)")
    print("="*70)
    type_counts, enabled_counts = show_channel_status(client)
    
    # Check if IMU is disabled
    imu_disabled = (enabled_counts['ACC'] == 0 and type_counts['ACC'] > 0) or \
                   (enabled_counts['GYRO'] == 0 and type_counts['GYRO'] > 0)
    
    if imu_disabled:
        print("\n❌ PROBLEM DETECTED: IMU channels are DISABLED!")
        print("   This is why you only see 18 channels instead of 60.")
        
        # Enable all channels
        print("\n" + "="*70)
        print("APPLYING FIX")
        print("="*70)
        enable_all_channels(client)
        
        # Show new status
        print("\n" + "="*70)
        print("CHANNEL STATUS (after fix)")
        print("="*70)
        type_counts2, enabled_counts2 = show_channel_status(client)
        
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"  EMG:   {enabled_counts['EMG']} → {enabled_counts2['EMG']}")
        print(f"  ACC:   {enabled_counts['ACC']} → {enabled_counts2['ACC']}")
        print(f"  GYRO:  {enabled_counts['GYRO']} → {enabled_counts2['GYRO']}")
        
        if enabled_counts2['ACC'] > 0 and enabled_counts2['GYRO'] > 0:
            print("\n✅ SUCCESS! IMU channels are now enabled.")
            print("\nTo make this permanent, add this code to your delsys_client.py")
            print("in the scan_sensors() method after selecting sensors:")
            print("""
    # Enable all channels (including IMU)
    for i in range(len(self.sensors)):
        sensor = self.sensors[i]
        for channel in list(sensor.TrignoChannels):
            channel.IsEnabled = True
            """)
        
    else:
        print("\n✓ All expected channels are already enabled!")
        print("  If you're still having issues, the problem is elsewhere.")
    
    # Now test configuration
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70)
    
    print("\nConfiguring with all channels enabled...")
    if client.configure():
        print("\nChannel breakdown after configuration:")
        client._parse_channels()
        
        emg_count = len([g for g, i in client.channel_info.items() if i['type'] == 'EMG'])
        acc_count = len([g for g, i in client.channel_info.items() if i['type'] == 'ACC'])
        gyro_count = len([g for g, i in client.channel_info.items() if i['type'] == 'GYRO'])
        
        print(f"\nFinal channel counts:")
        print(f"  EMG:  {emg_count}")
        print(f"  ACC:  {acc_count}")
        print(f"  GYRO: {gyro_count}")
        print(f"  Total: {len(client.channel_info)}")
        
        if acc_count > 0 and gyro_count > 0:
            print("\n🎉 SUCCESS! IMU channels are working!")
        else:
            print("\n❌ Still no IMU channels after configuration")
    
    # Cleanup
    client.disconnect()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()