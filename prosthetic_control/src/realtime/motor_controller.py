"""
Arduino Motor Controller

Sends gesture predictions to Arduino via serial port.
"""

import serial
import serial.tools.list_ports
import time
from typing import Optional


class MotorController:
    """
    Controls prosthetic hand motors via Arduino serial communication.
    
    Gesture mapping:
        0 = Neutral
        1 = Pinching
        2 = Grasping
        3 = Zipping (reserved for future use)
    """
    
    # Gesture name to Arduino command mapping
    GESTURE_MAP = {
        "Neutral": 0,
        "Pinching": 1,
        "Grasping": 2,
        "Zipping": 3  # Reserved for future gestures
    }
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        """
        Initialize motor controller.
        
        Args:
            port: Serial port (e.g., 'COM3' or '/dev/ttyACM0'). 
                  If None, will auto-detect Arduino.
            baudrate: Serial communication speed (default: 9600)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.last_gesture = None
        self.connected = False
        
    def find_arduino(self) -> Optional[str]:
        """Auto-detect Arduino port."""
        print("\nSearching for Arduino...")
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Look for Arduino in port description
            if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                print(f"  Found possible Arduino: {port.device} ({port.description})")
                return port.device
        
        # If no Arduino-specific port found, list all available
        if ports:
            print("  No Arduino detected. Available ports:")
            for port in ports:
                print(f"    {port.device}: {port.description}")
        else:
            print("  No serial ports found")
        
        return None
    
    def connect(self) -> bool:
        """
        Connect to Arduino.
        
        Returns:
            True if connection successful
        """
        # Auto-detect port if not specified
        if self.port is None:
            self.port = self.find_arduino()
            if self.port is None:
                print("\n✗ Could not find Arduino")
                print("  Please specify port manually: --motor-port COM3")
                return False
        
        try:
            print(f"\nConnecting to Arduino on {self.port}...")
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset after connection
            
            # Clear any startup messages
            self.serial.reset_input_buffer()
            
            self.connected = True
            print(f"✓ Connected to Arduino on {self.port}")
            
            # Send initial neutral command
            self.send_gesture("Neutral")
            
            return True
            
        except serial.SerialException as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def send_gesture(self, gesture_name: str) -> bool:
        """
        Send gesture command to Arduino.
        
        Args:
            gesture_name: Name of gesture ("Neutral", "Pinching", "Grasping", "Zipping")
        
        Returns:
            True if command sent successfully
        """
        if not self.connected or self.serial is None:
            return False
        
        # Don't send duplicate commands
        if gesture_name == self.last_gesture:
            return True
        
        # Get gesture code
        gesture_code = self.GESTURE_MAP.get(gesture_name)
        if gesture_code is None:
            print(f"⚠️  Unknown gesture: {gesture_name}")
            return False
        
        try:
            # Send gesture code as single byte
            self.serial.write(bytes([gesture_code]))
            self.serial.flush()
            
            self.last_gesture = gesture_name
            return True
            
        except serial.SerialException as e:
            print(f"✗ Send error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial is not None:
            try:
                # Send neutral before disconnecting
                self.send_gesture("Neutral")
                time.sleep(0.1)
                
                self.serial.close()
                print("✓ Disconnected from Arduino")
            except:
                pass
            
            self.serial = None
            self.connected = False


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("MOTOR CONTROLLER TEST")
    print("="*70)
    
    # Create controller
    motor = MotorController()
    
    # Connect
    if not motor.connect():
        exit(1)
    
    # Test gestures
    print("\nTesting gestures...")
    gestures = ["Neutral", "Pinching", "Grasping", "Neutral"]
    
    for gesture in gestures:
        print(f"\n  Sending: {gesture}")
        motor.send_gesture(gesture)
        time.sleep(2)
    
    # Disconnect
    motor.disconnect()
    
    print("\n✓ Test complete")