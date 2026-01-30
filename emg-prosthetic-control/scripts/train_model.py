#!/usr/bin/env python
"""
Model Training Script for EMG+IMU Gesture Classification

This script trains an LDA classifier on EMG and IMU data from CSV files.

Usage:
    # Basic training
    python scripts/train_model.py --data data/20260113 --model-name my_model
    
    # Train on multiple folders
    python scripts/train_model.py --data data/session1 data/session2 --model-name combined
    
    # Quick test (no cross-validation)
    python scripts/train_model.py --data data/20260113 --no-cv --model-name test
    
    # Custom parameters
    python scripts/train_model.py --data data/20260113 --window 0.25 --overlap 0.15
"""

import sys
import argparse
import os
import glob
from pathlib import Path
from datetime import datetime
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_emg_imu
from src.features.windowing import window_and_extract_features


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EMG+IMU gesture classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_model.py --data data/20260113

  # Multiple folders
  python scripts/train_model.py --data data/session1 data/session2
  
  # Custom model name
  python scripts/train_model.py --data data/20260113 --model-name my_model
  
  # Skip cross-validation for faster training
  python scripts/train_model.py --data data/20260113 --no-cv
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        required=True,
        help='Data folder(s) containing CSV files'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name for saved model (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save models (default: models/)'
    )
    
    # Training parameters
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation (faster training)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='standard',
        choices=['standard', 'robust', 'minmax'],
        help='Type of scaler (default: standard)'
    )
    
    # Feature extraction parameters
    parser.add_argument(
        '--window',
        type=float,
        default=0.2,
        help='Window duration in seconds (default: 0.2)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.1,
        help='Window overlap in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--fs-emg',
        type=float,
        default=1259,
        help='EMG sampling frequency in Hz (default: 1259)'
    )
    parser.add_argument(
        '--fs-imu',
        type=float,
        default=148,
        help='IMU sampling frequency in Hz (default: 148)'
    )
    
    # Other
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def get_trials_from_folder(data_dir, verbose=False):
    """
    Scan folder for CSV files and parse class labels from filenames.
    
    Expected format: "0.1_name.csv", "1.2_name.csv", etc.
    First number is the class label.
    """
    trial_files = glob.glob(os.path.join(data_dir, "*.csv"))
    class_trials = {}
    
    for f in trial_files:
        basename = os.path.basename(f)
        try:
            # Extract class from filename (e.g., "0.1_trial.csv" -> class 0)
            class_label = int(basename.split(".")[0])
            class_trials.setdefault(class_label, []).append(f)
            
            if verbose:
                print(f"    {basename} -> Class {class_label}")
                
        except (ValueError, IndexError):
            print(f"  ⚠ Skipping file with unexpected format: {basename}")
            continue
    
    return class_trials


def get_trials_from_multiple_folders(data_dirs, verbose=False):
    """Scan multiple folders and combine all trials."""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    combined_trials = {}
    folder_info = {}
    
    print(f"Scanning {len(data_dirs)} folder(s) for data...")
    
    for folder in data_dirs:
        if not os.path.exists(folder):
            print(f"  ⚠ Warning: Folder not found: {folder}")
            continue
        
        print(f"\nFolder: {folder}")
        folder_trials = get_trials_from_folder(folder, verbose)
        
        if len(folder_trials) == 0:
            print(f"  No valid CSV files found")
            continue
        
        # Track metadata
        folder_info[folder] = {
            'n_classes': len(folder_trials),
            'n_files': sum(len(files) for files in folder_trials.values()),
            'classes': list(folder_trials.keys())
        }
        
        # Merge into combined trials
        for label, files in folder_trials.items():
            combined_trials.setdefault(label, []).extend(files)
            print(f"  Class {label}: {len(files)} files")
    
    return combined_trials, folder_info


def process_trial(csv_path, label, fs_emg, fs_imu, window_sec, overlap_sec, verbose=False):
    """Load and extract features from a single trial."""
    emg_dict, imu_dict, _, _, _ = load_emg_imu(csv_path, fs_emg, fs_imu)
    
    if verbose:
        print(f"    EMG sensors: {len(emg_dict)}, IMU sensors: {len(imu_dict)}")
    
    # Extract features
    X = window_and_extract_features(
        emg_dict, imu_dict, fs_emg, fs_imu, window_sec, overlap_sec
    )
    y = np.full(X.shape[0], label)
    
    return X, y


def load_all_trials(class_trials, fs_emg, fs_imu, window_sec, overlap_sec, verbose=False):
    """Load and process all trials."""
    X_all = []
    y_all = []
    
    for label, files in sorted(class_trials.items()):
        print(f"\nProcessing Class {label} ({len(files)} files)...")
        
        for file in files:
            print(f"  {os.path.basename(file)}")
            try:
                X_trial, y_trial = process_trial(
                    file, label, fs_emg, fs_imu, window_sec, overlap_sec, verbose
                )
                print(f"    → {X_trial.shape[0]} windows, {X_trial.shape[1]} features")
                X_all.append(X_trial)
                y_all.append(y_trial)
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    if len(X_all) == 0:
        raise ValueError("No data was successfully loaded!")
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    return X_all, y_all


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def prepare_data(X, y, test_size=0.2, scaler_type='standard', random_state=42):
    """Split and scale data."""
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Scale
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def train_evaluate(X_train, X_test, y_train, y_test, use_cv=True, cv_folds=5):
    """Train and evaluate LDA classifier."""
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    # Cross-validation
    if use_cv and len(np.unique(y_train)) > 1:
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train
    print("\nTraining model...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    
    if train_acc - test_acc > 0.10:
        print("⚠ Warning: Possible overfitting (train >> test)")
    
    print(f"\nMacro F1:    {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    print(f"\nPer-Class Metrics:")
    class_names = ["Neutral", "Pinching", "Grasping", "Zipping"]
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1_per_class, support)):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  {name:12s}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, N={s}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=class_names[:len(np.unique(y_test))],
                                zero_division=0))
    
    # Return results
    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'support': support,
        'confusion_matrix': cm,
    }
    
    return clf, results


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model_and_scaler(clf, scaler, results, model_dir="models", model_name=None, 
                          train_params=None):
    """Save trained model with metadata."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
    
    # Create model folder
    save_path = os.path.join(model_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_file = os.path.join(save_path, "trained_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"✓ Model: {model_file}")
    
    # Save scaler
    scaler_file = os.path.join(save_path, "scaler.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler: {scaler_file}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'model_type': type(clf).__name__,
        'n_features': clf.coef_.shape[1] if hasattr(clf, 'coef_') else None,
        'n_classes': len(clf.classes_) if hasattr(clf, 'classes_') else None,
        'classes': clf.classes_.tolist() if hasattr(clf, 'classes_') else None,
        'class_names': {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping"
        },
        'performance': {
            'test_accuracy': float(results['test_accuracy']),
            'train_accuracy': float(results['train_accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
        },
        'training_params': train_params or {}
    }
    
    metadata_file = os.path.join(save_path, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata: {metadata_file}")
    
    # Save README
    readme = f"""# Model: {model_name}

## Performance
- Test Accuracy: {results['test_accuracy']:.2%}
- Macro F1: {results['f1_macro']:.4f}
- Training Date: {metadata['timestamp']}

## Model Info
- Type: {metadata['model_type']}
- Features: {metadata['n_features']}
- Classes: {metadata['n_classes']}

## Usage
```bash
python scripts/realtime_classify.py --model-name {model_name}
```
"""
    
    readme_file = os.path.join(save_path, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme)
    print(f"✓ README: {readme_file}")
    
    return save_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    print("="*70)
    print("EMG + IMU CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Data folders: {', '.join(args.data)}")
    print(f"  Model name: {args.model_name or 'auto-generated'}")
    print(f"  Window: {args.window}s, Overlap: {args.overlap}s")
    print(f"  Test size: {args.test_size}")
    print(f"  Scaler: {args.scaler}")
    print(f"  Cross-validation: {'No' if args.no_cv else f'Yes ({args.cv_folds} folds)'}")
    
    # Step 1: Load data
    print(f"\n{'='*70}")
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    class_trials, folder_info = get_trials_from_multiple_folders(args.data, args.verbose)
    
    if len(class_trials) == 0:
        print("\n✗ No valid CSV files found!")
        print("\nCheck that:")
        print("  - Folder paths are correct")
        print("  - CSV files exist")
        print("  - Filenames start with class number (e.g., '0.1_trial.csv')")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Folders: {len(folder_info)}")
    print(f"Classes: {len(class_trials)}")
    for label, files in sorted(class_trials.items()):
        print(f"  Class {label}: {len(files)} files")
    
    # Step 2: Extract features
    print(f"\n{'='*70}")
    print("STEP 2: FEATURE EXTRACTION")
    print("="*70)
    
    X, y = load_all_trials(
        class_trials, args.fs_emg, args.fs_imu, 
        args.window, args.overlap, args.verbose
    )
    
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    
    # Class distribution
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"\nClass distribution:")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        pct = (count / len(y)) * 100
        print(f"  Class {label}: {count:5d} samples ({pct:5.1f}%)")
    
    # Step 3: Train
    print(f"\n{'='*70}")
    print("STEP 3: TRAINING")
    print("="*70)
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        X, y, test_size=args.test_size, scaler_type=args.scaler
    )
    
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    clf, results = train_evaluate(
        X_train, X_test, y_train, y_test,
        use_cv=not args.no_cv, cv_folds=args.cv_folds
    )
    
    # Step 4: Save
    print(f"\n{'='*70}")
    print("STEP 4: SAVING MODEL")
    print("="*70)
    
    train_params = {
        'window_sec': args.window,
        'overlap_sec': args.overlap,
        'fs_emg': args.fs_emg,
        'fs_imu': args.fs_imu,
        'test_size': args.test_size,
        'scaler_type': args.scaler,
        'cv_folds': args.cv_folds if not args.no_cv else None,
    }
    
    save_path = save_model_and_scaler(
        clf, scaler, results,
        model_dir=args.model_dir,
        model_name=args.model_name,
        train_params=train_params
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Model saved: {save_path}")
    print(f"✓ Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"✓ Macro F1: {results['f1_macro']:.4f}")
    
    # Target check
    target_acc = 0.85
    if results['test_accuracy'] >= target_acc:
        print(f"\n🎉 Target accuracy ({target_acc*100}%) achieved!")
    else:
        gap = (target_acc - results['test_accuracy']) * 100
        print(f"\n📊 Gap to target: {gap:.1f}%")
        print("\nSuggestions:")
        print("  - Collect more training data")
        print("  - Try different window/overlap parameters")
        print("  - Add more trials per gesture")
    
    print(f"\nTo use for real-time classification:")
    print(f"  python scripts/realtime_classify.py --model-name {args.model_name or Path(save_path).name}")


if __name__ == "__main__":
    main()