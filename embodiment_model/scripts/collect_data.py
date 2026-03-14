"""
Collect embodiment data from sensors
"""

import argparse
import sys
from pathlib import Path
import time
import json
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "shared"))

from sensors.leap_motion import LeapMotionCapture
from sensors.bioradio import BioRadioCapture
from sensors.apple_watch import AppleWatchCapture
from src.data.data_loader import EmbodimentSession, save_embodiment_session


def collect_trial(participant_id: str, 
                  condition: str, 
                  trial_number: int,
                  duration: float = 30.0) -> EmbodimentSession:
    """
    Collect a single trial of embodiment data
    
    Args:
        participant_id: Participant ID
        condition: Experimental condition
        trial_number: Trial number
        duration: Trial duration (seconds)
    
    Returns:
        EmbodimentSession object
    """
    print(f"\n{'='*60}")
    print(f"Collecting Trial {trial_number}")
    print(f"Participant: {participant_id}")
    print(f"Condition: {condition}")
    print(f"Duration: {duration}s")
    print(f"{'='*60}\n")
    
    # Initialize sensors
    leap = LeapMotionCapture()
    bioradio = BioRadioCapture()
    watch = AppleWatchCapture()
    
    # Countdown
    print("Starting in...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO!\n")
    
    # Start recording
    leap.start_recording()
    bioradio.start_recording()
    watch.start_recording()
    
    start_time = time.time()
    
    # Record for specified duration
    try:
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\rRecording... {remaining:.1f}s remaining  ", end='')
            
            # Collect samples
            leap.get_frame()
            bioradio.get_sample()
            watch.get_sample()
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")
    
    # Stop recording
    leap_data = leap.stop_recording()
    bioradio_data = bioradio.stop_recording()
    watch_data = watch.stop_recording()
    
    print("\n\n✓ Recording complete")
    
    # Get embodiment rating from user
    print("\nPlease rate your sense of embodiment:")
    while True:
        try:
            embodiment_score = float(input("  Embodiment (0-100): "))
            if 0 <= embodiment_score <= 100:
                break
            print("  Please enter a value between 0 and 100")
        except ValueError:
            print("  Please enter a valid number")
    
    # Optional VEQ scores
    print("\nOptional - VEQ subscales (press Enter to skip):")
    veq_scores = {}
    
    for subscale in ['ownership', 'agency', 'location']:
        response = input(f"  {subscale.capitalize()} (1-7): ")
        if response.strip():
            try:
                veq_scores[subscale] = int(response)
            except ValueError:
                pass
    
    # Create session object
    session = EmbodimentSession(
        participant_id=participant_id,
        condition=condition,
        trial_number=trial_number,
        leap_data=leap_data,
        bioradio_data=bioradio_data,
        watch_data=watch_data,
        embodiment_score=embodiment_score,
        veq_scores=veq_scores if veq_scores else None,
        timestamp=datetime.now().isoformat(),
        duration=duration
    )
    
    return session


def main(args):
    """Main data collection script"""
    
    print("\n" + "="*70)
    print(" "*15 + "EMBODIMENT DATA COLLECTION")
    print("="*70)
    
    # Prepare output directory
    output_dir = Path(args.output) / args.participant_id / f"condition_{args.condition}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Trial duration: {args.duration}s")
    print(f"Rest between trials: {args.rest}s")
    
    input("\nPress Enter to begin data collection...")
    
    # Collect trials
    for trial in range(1, args.n_trials + 1):
        try:
            # Collect trial
            session = collect_trial(
                args.participant_id,
                args.condition,
                trial,
                args.duration
            )
            
            # Save trial
            save_embodiment_session(session, output_dir)
            
            # Rest between trials
            if trial < args.n_trials:
                print(f"\n\nRest period: {args.rest}s")
                for i in range(int(args.rest), 0, -1):
                    print(f"\rNext trial in {i}s...  ", end='')
                    time.sleep(1)
                print("\n")
        
        except KeyboardInterrupt:
            print("\n\nData collection stopped by user")
            break
        
        except Exception as e:
            print(f"\n⚠ Error during trial {trial}: {e}")
            response = input("Continue with next trial? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "="*70)
    print("Data collection complete!")
    print(f"Data saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect embodiment data")
    
    parser.add_argument("--participant-id", type=str, required=True,
                       help="Participant ID (e.g., P01)")
    parser.add_argument("--condition", type=str, required=True,
                       help="Experimental condition (e.g., baseline, 100ms_lag)")
    parser.add_argument("--output", type=str, default="../../data/raw/embodiment",
                       help="Output directory")
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Number of trials to collect")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Trial duration (seconds)")
    parser.add_argument("--rest", type=float, default=10.0,
                       help="Rest between trials (seconds)")
    
    args = parser.parse_args()
    main(args)