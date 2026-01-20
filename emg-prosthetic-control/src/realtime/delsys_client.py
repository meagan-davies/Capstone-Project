"""
Delsys Client - Official AeroPy API Wrapper

Based on official Delsys Example-Applications Python documentation.
This wraps the official AeroPy API to work with our classification system.

Prerequisites:
    1. Copy DelsysAPI.dll to resources/DelsysAPI/
    2. Install pythonnet: pip install pythonnet
    3. Get key/license from Delsys
"""

# CRITICAL: Set .NET runtime BEFORE any CLR imports
import os
os.environ['PYTHONNET_RUNTIME'] = 'coreclr'  # or 'netfx'

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def _setup_delsys_path():
    """Add DelsysAPI to Python path"""
    project_root = Path(__file__).parent.parent.parent
    delsys_path = project_root / "resources" / "DelsysAPI" / "resources"
    
    if not delsys_path.exists():
        raise FileNotFoundError(
            f"DelsysAPI not found at: {delsys_path}\n"
            f"Please copy DelsysAPI.dll to resources/DelsysAPI/resources"
        )
    
    if str(delsys_path) not in sys.path:
        sys.path.insert(0, str(delsys_path))

_setup_delsys_path()


# Import official Delsys API (following official documentation)
try:
    import clr
    clr.AddReference("DelsysAPI")
    clr.AddReference("System.Collections")
    from Aero import AeroPy
    DELSYS_AVAILABLE = True
except Exception as e:
    print(f"⚠ Warning: Could not load DelsysAPI: {e}")
    print("Make sure DelsysAPI.dll is in resources/DelsysAPI/")
    DELSYS_AVAILABLE = False


class DelsysClient:
    """
    Wrapper around official Delsys AeroPy API.
    
    Follows the official Delsys Python example structure but
    simplified for our real-time classification needs.
    
    Official docs pattern:
        BaseInstance = AeroPy()
        BaseInstance.ValidateBase(key, license)
        BaseInstance.ScanSensors()
        ...
    """
    
    def __init__(self, key: str, license: str):
        """
        Initialize client with credentials.
        
        Args:
            key: Your Delsys API key string
            license: Your Delsys license string
        """
        if not DELSYS_AVAILABLE:
            raise RuntimeError("DelsysAPI not available - check installation")
        
        self.key = key
        self.license = license
        
        # Create AeroPy instance (official pattern)
        self.base = AeroPy()
        
        # State tracking
        self.is_connected = False
        self.is_configured = False
        self.is_streaming = False
        
        # Sensor and channel info
        self.sensors = []
        self.channel_guids = []
        self.channel_info = {}
    
    def connect(self) -> bool:
        """
        Connect and validate base station.
        
        Official API method: ValidateBase(key, license)
        
        Returns:
            True if successful
        """
        try:
            print("Connecting to Trigno base station...")
            
            # Official API call
            self.base.ValidateBase(self.key, self.license)
            
            # Check pipeline state
            state = self.base.GetPipelineState()
            print(f"  Pipeline state: {state}")
            
            if state not in ["Off", "Connected"]:
                print(f"✗ Unexpected pipeline state: {state}")
                return False
            
            self.is_connected = True
            print("✓ Connected to Trigno base station")
            return True
            
        except Exception as e:
            print(f"✗ Connection error: {e}")
            print("\nTroubleshooting:")
            print("  - Check key and license are correct")
            print("  - Ensure base station is plugged in via USB")
            print("  - Make sure no other software is using the base")
            return False
    
    def scan_sensors(self) -> bool:
        if not self.is_connected:
            print("✗ Not connected - call connect() first")
            return False
        
        try:
            print("Scanning for sensors...")
            print("  (Starting system to activate sensors...)")
            
            # START THE SYSTEM FIRST to wake up sensors
            # This makes the blinking green LEDs go solid
            self.base.Start(False)
            
            import time
            time.sleep(2)  # Give sensors time to activate
            
            # Now scan for active sensors
            self.base.ScanSensors()
            
            # Stop the system (we'll restart later during actual streaming)
            self.base.Stop()
            self.base.ResetPipeline()
            
            # Get scanned sensors
            self.sensors = list(self.base.GetScannedSensorsFound())
            
            if len(self.sensors) == 0:
                print("✗ No sensors found")
                print("\nTroubleshooting:")
                print("  - Ensure sensors are powered on (remove from charger)")
                print("  - Check sensors are paired (use PairSensor if needed)")
                print("  - Verify sensors are in range")
                return False
            
            print(f"✓ Found {len(self.sensors)} sensor(s)")
            
            # Display sensor info
            for i, sensor in enumerate(self.sensors):
                pair_num = sensor.PairNumber
                sid = sensor.Properties.Sid
                mode = sensor.Configuration.ModeString
                print(f"  Sensor {i+1}: Pair#{pair_num}, SID:{sid}, Mode:{mode}")
            
            return True
            
        except Exception as e:
            print(f"✗ Scan error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def scan_sensors(self) -> bool:
        """
        Scan for paired sensors.
        
        Official API method: ScanSensors()
        Pipeline must be in Off or Connected state.
        
        Returns:
            True if sensors found
        """
        if not self.is_connected:
            print("✗ Not connected - call connect() first")
            return False
        
        try:
            print("Scanning for sensors...")
            
            # Ensure we're in the right state
            state = self.base.GetPipelineState()
            print(f"  Current state: {state}")
            
            if state not in ["Off", "Connected"]:
                print(f"✗ Cannot scan from state: {state}")
                print("  Try resetting the pipeline first")
                return False
            
            # Official API call (async)
            print("  Running scan...")
            self.base.ScanSensors()
            
            # Give it a moment to complete
            import time
            time.sleep(1)
            
            # Try multiple methods to get sensors
            try:
                # Method 1: GetScannedSensorsFound (official)
                self.sensors = list(self.base.GetScannedSensorsFound())
                print(f"  GetScannedSensorsFound: {len(self.sensors)} sensors")
            except Exception as e:
                print(f"  GetScannedSensorsFound failed: {e}")
                self.sensors = []
            
            # Method 2: Try GetPairedSensors as fallback
            if len(self.sensors) == 0:
                try:
                    paired = list(self.base.GetPairedSensors())
                    print(f"  GetPairedSensors: {len(paired)} sensors")
                    if len(paired) > 0:
                        print("  Using paired sensors instead of scanned sensors")
                        self.sensors = paired
                except Exception as e:
                    print(f"  GetPairedSensors also failed: {e}")
            
            if len(self.sensors) == 0:
                print("✗ No sensors found")
                print("\nTroubleshooting:")
                print("  - Ensure sensors are powered on (remove from charger)")
                print("  - Press the button on each sensor to wake them up")
                print("  - Check sensors are paired (use PairSensor if needed)")
                print("  - Verify sensors are in range")
                print("  - LED should be blinking green if paired")
                return False
            
            print(f"✓ Found {len(self.sensors)} sensor(s)")
            
            # Display sensor info
            for i, sensor in enumerate(self.sensors):
                try:
                    pair_num = sensor.PairNumber
                    sid = sensor.Properties.Sid
                    mode = sensor.Configuration.ModeString
                    print(f"  Sensor {i+1}: Pair#{pair_num}, SID:{sid}, Mode:{mode}")
                except Exception as e:
                    print(f"  Sensor {i+1}: Info unavailable ({e})")
            
            return True
            
        except Exception as e:
            print(f"✗ Scan error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def select_all_sensors(self) -> bool:
            """
            Select all scanned sensors for data collection.
            
            Official API method: SelectAllSensors()
            
            Returns:
                True if successful
            """
            try:
                success = self.base.SelectAllSensors()
                if success:
                    print(f"✓ Selected all {len(self.sensors)} sensors")
                return success
            except Exception as e:
                print(f"✗ Error selecting sensors: {e}")
                return False

    def scan_sensors(self) -> bool:
        """Scan using EXACT demo pattern"""
        if not self.is_connected:
            return False
        
        try:
            print("Scanning for sensors...")
            
            # Use their exact pattern with .Result
            try:
                f = self.base.ScanSensors().Result
            except Exception as e:
                print("Scan attempt retry...")
                import time
                time.sleep(1)
                try:
                    f = self.base.ScanSensors().Result
                except:
                    pass
            
            self.sensors = list(self.base.GetScannedSensorsFound())
            
            if len(self.sensors) == 0:
                print("✗ No sensors found")
                return False
            
            print(f"✓ Found {len(self.sensors)} sensor(s)")
            
            # Display info
            for i, sensor in enumerate(self.sensors):
                print(f"  Sensor {i+1}: Pair#{sensor.PairNumber}, {sensor.Configuration.ModeString}")
            
            # CRITICAL: Select sensors using their EXACT loop pattern
            for i in range(len(self.sensors)):
                self.base.SelectSensor(i)
            
            return True
            
        except Exception as e:
            print(f"✗ Scan error: {e}")
            return False


    def configure(self, enable_start_trigger: bool = False, 
                enable_stop_trigger: bool = False) -> bool:
        """Configure using EXACT demo pattern"""
        if not self.is_connected:
            return False
        
        try:
            state = self.base.GetPipelineState()
            
            if state == 'Armed':
                self._parse_channels()
                self.is_configured = True
                return True
            
            elif state == 'Connected':
                print("Configuring...")
                
                # Call Configure - EXACTLY like demo
                self.base.Configure(enable_start_trigger, enable_stop_trigger)
                
                # Wait and check - EXACTLY like demo
                import time
                time.sleep(0.5)
                state_05 = self.base.GetPipelineState()
                print(f"  State after 0.5s: {state_05}")
                
                time.sleep(1.0)
                state_15 = self.base.GetPipelineState()
                print(f"  State after 1.5s: {state_15}")
                
                time.sleep(1.0)
                state_25 = self.base.GetPipelineState()
                print(f"  State after 2.5s: {state_25}")
                
                # Check configured
                configured = self.base.IsPipelineConfigured()
                print(f"  IsPipelineConfigured: {configured}")
                
                if configured:
                    self._parse_channels()
                    self.is_configured = True
                    
                    if state_25 != 'Armed':
                        print(f"  ⚠ WARNING: State is '{state_25}' not 'Armed'")
                        return False
                    
                    print(f"✓ Configured - State: {state_25}")
                    return True
                else:
                    print("  ✗ Configuration failed")
                    return False
            
            else:
                print(f"✗ Cannot configure from state: {state}")
                return False
                
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            import traceback
            traceback.print_exc()
            return False
                        
    def _parse_channels(self):
        """
        Parse channel information after configuration.
        
        Uses official API properties:
        - sensor.TrignoChannels: List of ChannelTrigno objects
        - channel.Id: GUID for data parsing
        - channel.Name: Channel name (e.g., "EMG 1")
        - channel.Type: ChannelTypes enum (EMG, ACC, GYRO, etc.)
        - channel.IsEnabled: Whether channel is active
        """
        self.channel_info = {}
        self.channel_guids = []
        
        emg_count = 0
        acc_count = 0
        gyro_count = 0
        
        for sensor_idx, sensor in enumerate(self.sensors):
            # Get all channels for this sensor
            channels = list(sensor.TrignoChannels)
            
            for channel in channels:
                # Only use enabled channels
                if not channel.IsEnabled:
                    continue
                
                guid = channel.Id
                name = channel.Name
                chan_type = str(channel.Type)  # Convert enum to string
                sample_rate = channel.SampleRate
                unit = str(channel.Unit)
                
                self.channel_guids.append(guid)
                self.channel_info[guid] = {
                    'name': name,
                    'type': chan_type,
                    'sample_rate': sample_rate,
                    'unit': unit,
                    'sensor_index': sensor_idx
                }
                
                # Count by type
                if 'EMG' in chan_type:
                    emg_count += 1
                elif 'ACC' in chan_type:
                    acc_count += 1
                elif 'GYRO' in chan_type:
                    gyro_count += 1
        
        print(f"  Channel breakdown:")
        print(f"    EMG: {emg_count}")
        print(f"    ACC: {acc_count}")
        print(f"    GYRO: {gyro_count}")
    
    def start_streaming(self, yt_data: bool = False) -> bool:
        """
        Start data streaming - checking state like official demo
        """
        if not self.is_configured:
            print("✗ Not configured - call configure() first")
            return False
        
        try:
            # CRITICAL: Check state before starting (like official demo does)
            state = self.base.GetPipelineState()
            print(f"Starting data stream (current state: {state})...")
            
            # Their demo checks if Armed before calling Start
            if state != 'Armed':
                print(f"✗ Cannot start from state '{state}'")
                print(f"  Pipeline must be in 'Armed' state")
                print(f"\n  Troubleshooting:")
                print(f"    1. Call configure() again")
                print(f"    2. Check if Configure() succeeded")
                print(f"    3. Verify no errors during configuration")
                return False
            
            # Now call Start (same as them)
            self.base.Start(yt_data)
            
            import time
            time.sleep(0.5)
            
            self.is_streaming = True
            
            # Verify we're running
            final_state = self.base.GetPipelineState()
            print(f"  Pipeline state: {final_state}")
            
            if final_state == "Running":
                print("✓ Data streaming started")
                return True
            else:
                print(f"  ⚠ Warning: Expected 'Running', got '{final_state}'")
                return False
                
        except Exception as e:
            print(f"✗ Start error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def poll_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Poll data from sensors.
        
        Official API methods:
        - CheckDataQueue(): Check if data ready
        - PollData(): Get Dictionary<Guid, List<double>>
        
        Returns:
            Dictionary mapping channel GUIDs (as strings) to data arrays,
            or None if no data available
        """
        if not self.is_streaming:
            return None
        
        try:
            # Official API call - check if data ready
            if not self.base.CheckDataQueue():
                return None
            
            # Official API call - get data
            # Returns C# Dictionary<Guid, List<double>>
            data_dict = self.base.PollData()
            
            # Convert to Python format
            result = {}
            for guid, values in data_dict.items():
                # Convert GUID to string for dictionary key
                guid_str = str(guid)
                # Convert C# List to numpy array
                result[guid_str] = np.array(list(values))
            
            return result
            
        except Exception as e:
            # Don't print every poll error (too noisy)
            return None
    
    def stop_streaming(self):
        """
        Stop data streaming.
        
        Official API method: Stop()
        Pipeline transitions from Running to Armed.
        """
        if self.is_streaming:
            try:
                self.base.Stop()
                self.is_streaming = False
                print("✓ Stopped streaming")
                
                state = self.base.GetPipelineState()
                print(f"  Pipeline state: {state}")
                
            except Exception as e:
                print(f"⚠ Stop error: {e}")
    
    def reset_pipeline(self):
        """
        Reset/disarm pipeline.
        
        Official API method: ResetPipeline()
        Pipeline transitions from Armed to Connected.
        Allows scanning/pairing after stopping collection.
        """
        if self.is_configured:
            try:
                self.base.ResetPipeline()
                self.is_configured = False
                print("✓ Pipeline reset")
                
                state = self.base.GetPipelineState()
                print(f"  Pipeline state: {state}")
                
            except Exception as e:
                print(f"⚠ Reset error: {e}")
    
    def disconnect(self):
        """Cleanup and disconnect"""
        if self.is_streaming:
            self.stop_streaming()
        
        if self.is_configured:
            self.reset_pipeline()
        
        print("✓ Disconnected")
    
    def get_emg_channel_guids(self) -> List[str]:
        """Get list of EMG channel GUIDs (as strings)"""
        return [
            str(guid) for guid, info in self.channel_info.items() 
            if 'EMG' in info['type']
        ]
    
    def get_imu_channel_guids(self) -> List[str]:
        """Get list of IMU channel GUIDs (ACC + GYRO, as strings)"""
        return [
            str(guid) for guid, info in self.channel_info.items() 
            if 'ACC' in info['type'] or 'GYRO' in info['type']
        ]
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state"""
        return self.base.GetPipelineState()
    
    def get_total_packets(self) -> int:
        """Get total data packets collected"""
        return self.base.GetTotalPackets()


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
        raise FileNotFoundError(
            f"Key file not found: {key_path}\n"
            f"Create this file with your Delsys API key"
        )
    
    if not license_path.exists():
        raise FileNotFoundError(
            f"License file not found: {license_path}\n"
            f"Create this file with your Delsys license"
        )
    
    key = key_path.read_text().strip()
    license = license_path.read_text().strip()
    
    if not key or not license:
        raise ValueError("Key or license file is empty")
    
    return key, license


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Testing Delsys Client (Official API)")
    print("="*60)
    
    try:
        # Load credentials
        print("\nLoading credentials...")
        KEY, LICENSE = load_credentials()
        print(f"✓ Key: {KEY[:10]}... ({len(KEY)} chars)")
        print(f"✓ License: {LICENSE[:10]}... ({len(LICENSE)} chars)")
        
        # Create client
        print("\nCreating client...")
        client = DelsysClient(KEY, LICENSE)
        print("✓ Client created")
        
        # Test connection
        print("\n" + "="*60)
        print("Test 1: Connection")
        print("="*60)
        if not client.connect():
            print("✗ Connection failed")
            exit(1)
        
        # Test scan
        print("\n" + "="*60)
        print("Test 2: Sensor Scan")
        print("="*60)
        if not client.scan_sensors():
            print("✗ Scan failed")
            exit(1)
        
        # Test configuration
        print("\n" + "="*60)
        print("Test 3: Configuration")
        print("="*60)
        if not client.configure():
            print("✗ Configuration failed")
            exit(1)
        
        # Test streaming (brief)
        print("\n" + "="*60)
        print("Test 4: Data Streaming (5 seconds)")
        print("="*60)
        
        if client.start_streaming():
            import time
            packets_received = 0
            
            for i in range(50):  # 5 seconds at 10 Hz
                data = client.poll_data()
                if data:
                    packets_received += 1
                    print(f"\r  Packets: {packets_received}, Channels: {len(data)}", 
                          end='', flush=True)
                time.sleep(1)
            
            print(f"\n✓ Received {packets_received} data packets")
            client.stop_streaming()
        
        # Cleanup
        client.disconnect()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Credential error: {e}")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()