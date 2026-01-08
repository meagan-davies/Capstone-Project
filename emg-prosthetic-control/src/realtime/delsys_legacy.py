
# ============================================================
# DELSYS API CONNECTION
# ============================================================

class DelsysClient:
    """
    Client for connecting to Delsys Trigno system.
    
    Default Delsys ports:
    - Command port: 50040
    - EMG data port: 50043 (2000 Hz)
    - Auxiliary (IMU) data port: 50044 (148.148 Hz)
    """
    
    def __init__(self, host='localhost', cmd_port=50040, 
                 emg_port=50043, aux_port=50044):
        self.host = host
        self.cmd_port = cmd_port
        self.emg_port = emg_port
        self.aux_port = aux_port
        
        self.cmd_socket = None
        self.emg_socket = None
        self.aux_socket = None
        
        self.is_streaming = False
        
    def connect(self):
        """Establish connection to Delsys system"""
        try:
            # Command socket
            self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_socket.connect((self.host, self.cmd_port))
            print(f"✓ Connected to command port: {self.host}:{self.cmd_port}")
            
            # EMG data socket
            self.emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.emg_socket.connect((self.host, self.emg_port))
            print(f"✓ Connected to EMG port: {self.host}:{self.emg_port}")
            
            # Auxiliary (IMU) data socket
            self.aux_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.aux_socket.connect((self.host, self.aux_port))
            print(f"✓ Connected to AUX port: {self.host}:{self.aux_port}")
            
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send command to Delsys system"""
        if self.cmd_socket:
            self.cmd_socket.send(f"{command}\r\n\r\n".encode())
            time.sleep(0.1)
            
    def start_streaming(self):
        """Start data streaming"""
        self.send_command("START")
        self.is_streaming = True
        print("✓ Started streaming")
        
    def stop_streaming(self):
        """Stop data streaming"""
        self.send_command("STOP")
        self.is_streaming = False
        print("✓ Stopped streaming")
        
    def disconnect(self):
        """Close all connections"""
        if self.is_streaming:
            self.stop_streaming()
        
        if self.cmd_socket:
            self.cmd_socket.close()
        if self.emg_socket:
            self.emg_socket.close()
        if self.aux_socket:
            self.aux_socket.close()
        
        print("✓ Disconnected")