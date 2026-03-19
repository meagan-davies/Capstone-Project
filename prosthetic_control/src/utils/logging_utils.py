# src/utils/logging_utils.py

import csv
import os
from pathlib import Path
from datetime import datetime

class CSVLogger:
    """
    Simple CSV logger for real-time predictions.
    Automatically creates timestamped files and writes rows.
    """
    
    def __init__(self, folder: str = "logs", prefix: str = "predictions"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.folder / f"{prefix}_{timestamp}.csv"
        
        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "PredictedLabel", "ClassName", "Confidence"])
        
        print(f"✓ Logging predictions to: {self.file_path}")
    
    def log(self, pred_label: int, class_name: str, confidence: float):
        """Write a prediction row to the CSV"""
        self.writer.writerow([datetime.now().isoformat(), pred_label, class_name, confidence])
        # Optional: flush to disk every time to avoid data loss
        self.file.flush()
    
    def close(self):
        """Close the CSV file"""
        self.file.close()
        print(f"✓ Logger closed: {self.file_path}")
