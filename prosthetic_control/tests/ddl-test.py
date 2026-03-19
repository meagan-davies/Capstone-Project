import clr
import sys
from pathlib import Path

dll_path = Path("resources/DelsysAPI/resources")
print(f"DLL directory: {dll_path.absolute()}")
print(f"Exists: {dll_path.exists()}")

# Add to path
sys.path.insert(0, str(dll_path.absolute()))

# Try to load
try:
    clr.AddReference("DelsysAPI")
    print("✓ DLL loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load DLL:")
    print(f"  {e}")