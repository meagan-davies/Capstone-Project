#!/usr/bin/env python
"""Real-time classification script"""

import argparse
from src.realtime.system import RealtimeEMGSystem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='model_latest')
    parser.add_argument('--host', default='localhost')
    args = parser.parse_args()
    
    # Load and run system
    system = RealtimeEMGSystem(...)
    system.connect_and_setup()
    system.start()

if __name__ == "__main__":
    main()