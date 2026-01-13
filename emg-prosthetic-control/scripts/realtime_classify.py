"""
Real-time classification script
"""

import argparse
from src.realtime.system import RealtimeEMGSystem


def main():
    parser = argparse.ArgumentParser(description="Run real-time EMG classification")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained_model.pkl"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        required=True,
        help="Path to scaler.pkl"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store prediction logs"
    )

    args = parser.parse_args()

    system = RealtimeEMGSystem(
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )

    if not system.setup():
        print("✗ System setup failed")
        sys.exit(1)

    system.start(log_folder=args.log_dir)


if __name__ == "__main__":
    main()