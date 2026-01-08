# ============================================================
# MAIN REAL-TIME SYSTEM
# ============================================================

class RealtimeEMGSystem:
    """
    Complete real-time EMG+IMU classification system.
    Integrates Delsys client, data processor, and classifier.
    """
    
    def __init__(self, model_path, scaler_path, 
                 n_emg_channels=4, n_imu_channels=6,
                 host='localhost'):
        
        self.client = DelsysClient(host=host)
        self.processor = RealtimeProcessor(
            n_emg_channels=n_emg_channels,
            n_imu_channels=n_imu_channels
        )
        self.classifier = RealtimeClassifier(model_path, scaler_path)
        
        self.is_running = False
        self.emg_thread = None
        self.imu_thread = None
        
    def connect(self):
        """Connect to Delsys system"""
        return self.client.connect()
    
    def _stream_emg_data(self):
        """Thread function to stream EMG data"""
        bytes_per_sample = 4  # float32
        bytes_per_channel = bytes_per_sample * self.processor.n_emg_channels
        
        while self.is_running:
            try:
                data = self.client.emg_socket.recv(bytes_per_channel)
                if len(data) == bytes_per_channel:
                    # Unpack float values
                    values = struct.unpack('f' * self.processor.n_emg_channels, data)
                    self.processor.add_emg_sample(np.array(values))
            except Exception as e:
                if self.is_running:
                    print(f"EMG streaming error: {e}")
                break
    
    def _stream_imu_data(self):
        """Thread function to stream IMU data"""
        bytes_per_sample = 4  # float32
        bytes_per_channel = bytes_per_sample * self.processor.n_imu_channels
        
        while self.is_running:
            try:
                data = self.client.aux_socket.recv(bytes_per_channel)
                if len(data) == bytes_per_channel:
                    values = struct.unpack('f' * self.processor.n_imu_channels, data)
                    self.processor.add_imu_sample(np.array(values))
            except Exception as e:
                if self.is_running:
                    print(f"IMU streaming error: {e}")
                break
    
    def start(self):
        """Start real-time classification"""
        if not self.client.is_streaming:
            self.client.start_streaming()
        
        self.is_running = True
        
        # Start streaming threads
        self.emg_thread = Thread(target=self._stream_emg_data, daemon=True)
        self.imu_thread = Thread(target=self._stream_imu_data, daemon=True)
        
        self.emg_thread.start()
        self.imu_thread.start()
        
        print("\n" + "="*60)
        print("REAL-TIME CLASSIFICATION STARTED")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                if self.processor.is_window_ready():
                    features = self.processor.extract_window_features()
                    
                    if features is not None:
                        pred_label, pred_proba = self.classifier.predict_smoothed(features)
                        class_name = self.classifier.get_class_name(pred_label)
                        confidence = pred_proba[pred_label] * 100
                        
                        # Display prediction
                        print(f"\r{class_name:12s} | Confidence: {confidence:5.1f}%", 
                              end='', flush=True)
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
    
    def stop(self):
        """Stop real-time classification"""
        self.is_running = False
        
        if self.emg_thread:
            self.emg_thread.join(timeout=1)
        if self.imu_thread:
            self.imu_thread.join(timeout=1)
        
        self.client.disconnect()
        print("✓ System stopped")