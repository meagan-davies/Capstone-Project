"""
Model Utilities

Handles:
- Saving model bundles
- Loading model bundles
- Model versioning
- Metadata management
"""

from __future__ import annotations

import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .classifier import ModelBundle


# Constants
DEFAULT_MODEL_DIR = Path("models/saved_models")
BUNDLE_FILENAME = "model_bundle.pkl"
METADATA_FILENAME = "metadata.json"

# Make objects JSON serializable
def _make_json_serializable(obj):
    """
    Convert numpy types to native Python types
    so they can be saved in JSON metadata.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]

    return obj

# Save Function
def save_model_bundle(bundle: ModelBundle, metrics: Dict[str, Any], model_name: Optional[str] = None, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    """
    Save a trained ModelBundle with metadata.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"

    save_path = model_dir / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Extract classifier type
    clf = bundle.pipeline.named_steps["clf"]
    model_type = type(clf).__name__

    # Convert metrics to JSON-safe format
    safe_metrics = _make_json_serializable(metrics)

    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "model_type": bundle.classifier_type,
        "scaler_type": bundle.scaler_type,
        "feature_count": bundle.feature_count,
        "n_classes": bundle.n_classes,
        "results": safe_metrics,
    }

    # Save bundle (pickle handles numpy fine)
    bundle_file = save_path / BUNDLE_FILENAME
    with open(bundle_file, "wb") as f:
        pickle.dump(bundle, f)

    # Save metadata (JSON-safe)
    metadata_file = save_path / METADATA_FILENAME
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Logging
    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"Location: {save_path}")
    print(f"Model: {model_type}")
    print(f"Scaler: {bundle.scaler_type}")

    acc = metrics.get("test_accuracy")
    if acc is not None:
        print(f"Test Accuracy: {float(acc):.2%}")

    print("=" * 60 + "\n")

    return save_path

# Load Function
def load_model_bundle(model_name: str | Path, model_dir: Path | str = DEFAULT_MODEL_DIR) -> ModelBundle:
    """
    Load ModelBundle from disk.
    """

    model_dir = Path(model_dir)
    model_path = Path(model_name)

    if not model_path.exists():
        model_path = model_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_name}\n"
            f"Available models: {list_available_models(model_dir)}"
        )

    bundle_file = model_path / BUNDLE_FILENAME

    if not bundle_file.exists():
        raise FileNotFoundError(f"Bundle file missing: {bundle_file}")

    with open(bundle_file, "rb") as f:
        bundle: ModelBundle = pickle.load(f)

    print(f"✓ Loaded model from: {model_path}")

    return bundle

# List Models
def list_available_models(model_dir: Path | str = DEFAULT_MODEL_DIR) -> List[str]:

    model_dir = Path(model_dir)

    if not model_dir.exists():
        return []

    models = []

    for item in model_dir.iterdir():
        if item.is_dir() and (item / BUNDLE_FILENAME).exists():
            models.append(item.name)

    return sorted(models)

# Metadata Inspection
def get_model_info(model_name: str, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:

    model_dir = Path(model_dir)
    metadata_file = model_dir / model_name / METADATA_FILENAME

    if not metadata_file.exists():
        return {}

    with open(metadata_file, "r") as f:
        return json.load(f)

# Model Comparison
def compare_models(model_names: List[str], model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    header = f"{'Model':<25} {'Type':<15} {'Accuracy':<10} {'Created'}"
    print(header)
    print("-" * 80)

    for name in model_names:

        info = get_model_info(name, model_dir)

        if not info:
            print(f"{name:<25} NOT FOUND")
            continue

        model_type = info.get("model_type", "Unknown")
        acc = info.get("results", {}).get("test_accuracy", 0)
        created = info.get("timestamp", "Unknown")[:10]

        print(f"{name:<25} {model_type:<15} {float(acc):<10.2%} {created}")

# Convenience Loader for Realtime
def load_for_inference(model_name: str | Path, model_dir: Path | str = DEFAULT_MODEL_DIR):

    bundle = load_model_bundle(model_name, model_dir)

    return bundle.pipeline, bundle

# CLI Test
if __name__ == "__main__":

    print("Available models:")
    models = list_available_models()

    if models:
        for m in models:
            print(f"  - {m}")
    else:
        print("None found")

    print("\n✓ model_utils ready")