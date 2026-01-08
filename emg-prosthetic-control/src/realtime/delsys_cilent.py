"""
Delsys Client - Wrapper around official Delsys TrignoBase API

This file wraps the official Delsys Example-Applications Python code
to make it easier to use in our real-time classification system.

Prerequisites:
    1. Copy Example-Applications/Python/ files to resources/DelsysAPI/
    2. Install pythonnet: pip install pythonnet
    3. Create resources/delsys_key.txt with your API key
    4. Create resources/delsys_license.lic with your license
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np


# Add DelsysAPI to Python path
def _setup_delsys_path():
    """Add Delsys API directories to Python path"""
    project_root = Path(__file__).parent.parent.parent
    delsys_path = project_root / "resources" / "DelsysAPI"
    
    if not delsys_path.exists():
        raise FileNotFoundError(
            f"DelsysAPI not found at: {delsys_path}\n"
            f"Please copy files from Example-Applications/Python/ to resources/DelsysAPI/"
        )
    
    if str(delsys_path) not in sys.path:
        sys.path.insert(0, str(delsys_path))

_setup_delsys_path()

# Import official Delsys API
try:
    import clr
    clr.AddReference("DelsysAPI")
    from Aero import AeroPy
    DELSYS_AVAILABLE = True
except Exception as e:
    print(f"⚠ Warning: Could not load DelsysAPI: {e}")
    DELSYS_AVAILABLE = False


class DelsysClient:
    """
    Wrapper around official Delsys TrignoBase API.
    
    Simplifies the official API for our real-time classification needs.
    """
    
    def __init__(self, key: str, license: str):
        """
        Initialize client with credentials.
        
        Args:
            key: Your Delsys API key
            license: Your Delsys license string
        """
        if not DELSYS_AVAILABLE:
            raise RuntimeError("DelsysAPI not available - check installation")
        
        self.key = key
        self.license = license
        self.trigno_base = None
        self.is_connected = False
        self.is_streaming = False
        self.channel_guids = []
        self.channel_info = {}
    
    def connect(self) -> bool:
        """
        Connect to Trigno base station.
        
        Returns:
            True if successful
        """
        try:
            # Create TrignoBase object (official Delsys API)
            self.trigno_base = AeroPy.TrignoBase(self.key, self.license)
            
            # Validate connection
            if not self.trigno_base.ValidateBase("RF"):
                print("✗ Failed to validate Trigno base")
                return False
            
            self.is_connected = True
            print("✓ Connected to Trigno base station")
            return True
            
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    def scan_sensors(self) -> bool:
        """
        Scan for paired sensors.
        
        Returns:
            True if sensors found
        """
        if not self.trigno_base:
            print("✗ Not connected")
            return False
        
        try:
            sensors = self.trigno_base.ScanSensors()
            
            if len(sensors) == 0:
                print("✗ No sensors found")
                return False
            
            print(f"✓ Found {len(sensors)} sensor(s)")
            for i, sensor in enumerate(sensors):
                print(f"  Sensor {i+1}: {sensor.FriendlyName}")
            
            return True
            
        except Exception as e:
            print(f"✗ Scan error: {e}")
            return False
    
    def configure(self) -> bool:
        """
        Configure data collection.
        
        Returns:
            True if successful
        """
        if not self.trigno_base:
            print("✗ Not connected")
            return False
        
        try:
            # Set sample mode (EMG + IMU)
            self.trigno_base.SetSampleMode(self.trigno_base.SampleMode.Default)
            
            # Configure
            if not self.trigno_base.Configure():
                print("✗ Configuration failed")
                return False
            
            # Get channel information
            self.channel_guids = list(self.trigno_base.GetChannelGuids())
            self._parse_channels()
            
            print(f"✓ Configured {len(self.channel_guids)} channels")
            return True
            
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            return False
    
    def _parse_channels(self):
        """Parse channel information for later use"""
        for guid in self.channel_guids:
            try:
                name = self.trigno_base.GetChannelName(guid)
                chan_type = str(self.trigno_base.GetChannelType(guid))
                enabled = self.trigno_base.GetChannelIsEnabled(guid)
                
                if enabled:
                    self.channel_info[guid] = {
                        'name': name,
                        'type': chan_type
                    }
            except:
                pass
        
        # Count channels by type
        emg_count = sum(1 for c in self.channel_info.values() if 'EMG' in c['type'])
        acc_count = sum(1 for c in self.channel_info.values() if 'ACC' in c['type'])
        gyro_count = sum(1 for c in self.channel_info.values() if 'GYRO' in c['type'])
        
        print(f"  EMG: {emg_count}, ACC: {acc_count}, GYRO: {gyro_count}")
    
    def start_streaming(self) -> bool:
        """
        Start data streaming.
        
        Returns:
            True if successful
        """
        if not self.trigno_base:
            print("✗ Not connected")
            return False
        
        try:
            self.trigno_base.Start()
            self.is_streaming = True
            print("✓ Started streaming")
            return True
            
        except Exception as e:
            print(f"✗ Start error: {e}")
            return False
    
    def poll_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Poll data from sensors.
        
        Returns:
            Dictionary mapping channel GUIDs to data arrays, or None if no data
        """
        if not self.is_streaming:
            return None
        
        try:
            # Check if data available
            if not self.trigno_base.CheckDataQueue():
                return None
            
            # Get data
            data_dict = self.trigno_base.PollData()
            
            # Convert to numpy arrays
            result = {}
            for guid, values in data_dict.items():
                result[guid] = np.array(list(values))
            
            return result
            
        except Exception as e:
            print(f"⚠ Poll error: {e}")
            return None
    
    def stop_streaming(self):
        """Stop data streaming"""
        if self.is_streaming:
            try:
                self.trigno_base.Stop()
                self.is_streaming = False
                print("✓ Stopped streaming")
            except Exception as e:
                print(f"⚠ Stop error: {e}")
    
    def disconnect(self):
        """Disconnect from base station"""
        self.stop_streaming()
        if self.is_connected and self.trigno_base:
            try:
                self.trigno_base.Dispose()
                self.is_connected = False
                print("✓ Disconnected")
            except Exception as e:
                print(f"⚠ Disconnect error: {e}")
    
    def get_emg_channel_guids(self):
        """Get list of EMG channel GUIDs"""
        return [guid for guid, info in self.channel_info.items() 
                if 'EMG' in info['type']]
    
    def get_imu_channel_guids(self):
        """Get list of IMU channel GUIDs (ACC + GYRO)"""
        return [guid for guid, info in self.channel_info.items() 
                if 'ACC' in info['type'] or 'GYRO' in info['type']]


def load_credentials(
    key_file: str = "resources/delsys_key.txt",
    license_file: str = "resources/delsys_license.lic"
) -> tuple:
    """
    Load Delsys credentials from files.
    
    Returns:
        (key, license) tuple
    """
    key_path = Path(key_file)
    license_path = Path(license_file)
    
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found: {key_path}")
    
    if not license_path.exists():
        raise FileNotFoundError(f"License file not found: {license_path}")
    
    key = key_path.read_text().strip()
    license = license_path.read_text().strip()
    
    return key, license


# Test code
if __name__ == "__main__":
    print("Testing Delsys client...")
    
    try:
        # Load credentials
        KEY, LICENSE = load_credentials()
        print(f"✓ Credentials loaded")
        
        # Create client
        client = DelsysClient(KEY, LICENSE)
        
        # Test connection
        if client.connect():
            print("\n✓ Connection successful")
            
            if client.scan_sensors():
                print("\n✓ Sensors found")
                
                if client.configure():
                    print("\n✓ Configuration successful")
                    print("\nAll tests passed!")
            
            client.disconnect()
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")