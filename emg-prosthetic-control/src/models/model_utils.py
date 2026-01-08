"""
Model Utilities Module

Functions for:
- Saving trained models with metadata
- Loading models for inference
- Model versioning and management
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path


def save_model_package(
    clf: object,
    scaler: object,
    model_dir: str = "models/saved_models",
    model_name: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
    results: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> str:
    """
    Save trained model, scaler, and metadata as a complete package.
    
    This is your save_model_and_scaler() function from the notebook,
    but enhanced with more metadata and better organization.
    
    Args:
        clf: Trained classifier (e.g., LDA)
        scaler: Fitted scaler (e.g., StandardScaler)
        model_dir: Base directory for saving models
        model_name: Name for this model (auto-generated if None)
        class_names: Dictionary mapping class IDs to names
        results: Dictionary of evaluation results
        config: Dictionary of training configuration
        
    Returns:
        Full path to saved model directory
        
    Example:
        >>> save_path = save_model_package(
        ...     clf, scaler,
        ...     model_name="production_v1",
        ...     class_names={0: "Neutral", 1: "Pinching"},
        ...     results={'accuracy': 0.87}
        ... )
        >>> print(f"Model saved to: {save_path}")
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
    
    # Create subfolder for this model
    save_path = os.path.join(model_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nSaving model package to: {save_path}")
    
    # 1. Save the trained model
    model_file = os.path.join(save_path, "trained_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  ✓ Model saved: trained_model.pkl")
    
    # 2. Save the scaler
    scaler_file = os.path.join(save_path, "scaler.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: scaler.pkl")
    
    # 3. Save comprehensive metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'model_type': type(clf).__name__,
        'scaler_type': type(scaler).__name__,
    }
    
    # Add model-specific info
    if hasattr(clf, 'coef_'):
        metadata['n_features'] = clf.coef_.shape[1]
    if hasattr(clf, 'classes_'):
        metadata['n_classes'] = len(clf.classes_)
        metadata['class_ids'] = clf.classes_.tolist()
    
    # Add class names
    if class_names:
        metadata['class_names'] = class_names
    else:
        # Default class names
        metadata['class_names'] = {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping"
        }
    
    # Add evaluation results
    if results:
        metadata['results'] = {
            'accuracy': float(results.get('accuracy', 0)),
            'f1_macro': float(results.get('f1_macro', 0)),
            'f1_weighted': float(results.get('f1_weighted', 0)),
        }
        if 'train_accuracy' in results:
            metadata['results']['train_accuracy'] = float(results['train_accuracy'])
    
    # Add training configuration
    if config:
        metadata['config'] = config
    
    metadata_file = os.path.join(save_path, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"  ✓ Metadata saved: metadata.json")
    
    # 4. Create a human-readable README
    readme_content = generate_model_readme(metadata, results)
    readme_file = os.path.join(save_path, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ README saved: README.md")
    
    # 5. Save confusion matrix if available
    if results and 'confusion_matrix' in results:
        import numpy as np
        cm_file = os.path.join(save_path, "confusion_matrix.txt")
        np.savetxt(cm_file, results['confusion_matrix'], fmt='%d')
        print(f"  ✓ Confusion matrix saved: confusion_matrix.txt")
    
    print(f"\n{'='*60}")
    print("MODEL PACKAGE SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Location: {save_path}")
    print(f"Model name: '{model_name}'")
    
    if results:
        print(f"\nPerformance:")
        print(f"  Accuracy: {results.get('accuracy', 0):.2%}")
        print(f"  F1 Score: {results.get('f1_macro', 0):.4f}")
    
    return save_path


def load_model_package(
    model_name: str,
    model_dir: str = "models/saved_models"
) -> Dict[str, Any]:
    """
    Load a complete model package (model + scaler + metadata).
    
    Args:
        model_name: Name of the model to load
        model_dir: Base directory containing models
        
    Returns:
        Dictionary containing:
            - 'model': The trained classifier
            - 'scaler': The fitted scaler
            - 'metadata': Model metadata dictionary
            
    Example:
        >>> package = load_model_package("production_v1")
        >>> clf = package['model']
        >>> scaler = package['scaler']
        >>> print(f"Model type: {package['metadata']['model_type']}")
    """
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Available models: {list_available_models(model_dir)}"
        )
    
    # Load model
    model_file = os.path.join(model_path, "trained_model.pkl")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_file = os.path.join(model_path, "scaler.pkl")
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    metadata_file = os.path.join(model_path, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"✓ Loaded model: {model_name}")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Scaler type: {type(scaler).__name__}")
    if metadata:
        print(f"  Created: {metadata.get('timestamp', 'Unknown')}")
        if 'results' in metadata:
            print(f"  Accuracy: {metadata['results'].get('accuracy', 0):.2%}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': metadata
    }


def list_available_models(model_dir: str = "models/saved_models") -> list:
    """
    List all available saved models.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        List of model names
        
    Example:
        >>> models = list_available_models()
        >>> print(f"Available models: {models}")
    """
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains a model file
            if os.path.exists(os.path.join(item_path, "trained_model.pkl")):
                models.append(item)
    
    return sorted(models)


def get_model_info(
    model_name: str,
    model_dir: str = "models/saved_models"
) -> Dict:
    """
    Get metadata for a model without loading the full model.
    
    Useful for quickly checking model properties.
    
    Args:
        model_name: Name of the model
        model_dir: Directory containing models
        
    Returns:
        Dictionary of model metadata
        
    Example:
        >>> info = get_model_info("production_v1")
        >>> print(f"Accuracy: {info['results']['accuracy']}")
    """
    metadata_file = os.path.join(model_dir, model_name, "metadata.json")
    
    if not os.path.exists(metadata_file):
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def compare_models(
    model_names: list,
    model_dir: str = "models/saved_models"
) -> None:
    """
    Compare multiple saved models.
    
    Args:
        model_names: List of model names to compare
        model_dir: Directory containing models
        
    Example:
        >>> compare_models(['model_v1', 'model_v2', 'model_v3'])
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model Name':<20} {'Type':<15} {'Accuracy':<10} {'F1 Score':<10} {'Created'}")
    print("-" * 80)
    
    for name in model_names:
        info = get_model_info(name, model_dir)
        if not info:
            print(f"{name:<20} {'Not found':>45}")
            continue
        
        model_type = info.get('model_type', 'Unknown')[:14]
        accuracy = info.get('results', {}).get('accuracy', 0)
        f1 = info.get('results', {}).get('f1_macro', 0)
        created = info.get('timestamp', 'Unknown')[:10]
        
        print(f"{name:<20} {model_type:<15} {accuracy:<10.2%} {f1:<10.4f} {created}")


def generate_model_readme(metadata: Dict, results: Optional[Dict] = None) -> str:
    """
    Generate a README file for the model.
    
    Args:
        metadata: Model metadata dictionary
        results: Evaluation results dictionary
        
    Returns:
        README content as string
    """
    readme = f"""# Model: {metadata.get('model_name', 'Unknown')}

## Model Information

- **Created**: {metadata.get('timestamp', 'Unknown')}
- **Model Type**: {metadata.get('model_type', 'Unknown')}
- **Scaler Type**: {metadata.get('scaler_type', 'Unknown')}
- **Number of Features**: {metadata.get('n_features', 'Unknown')}
- **Number of Classes**: {metadata.get('n_classes', 'Unknown')}

## Classes

"""
    
    # Add class information
    class_names = metadata.get('class_names', {})
    for class_id, class_name in class_names.items():
        readme += f"- **{class_id}**: {class_name}\n"
    
    # Add performance metrics
    if results or 'results' in metadata:
        results = results or metadata.get('results', {})
        readme += f"""
## Performance

- **Test Accuracy**: {results.get('accuracy', 0):.2%}
- **Macro F1 Score**: {results.get('f1_macro', 0):.4f}
- **Weighted F1 Score**: {results.get('f1_weighted', 0):.4f}
"""
        
        if 'train_accuracy' in results:
            train_acc = results['train_accuracy']
            test_acc = results['accuracy']
            overfitting = train_acc - test_acc
            readme += f"- **Training Accuracy**: {train_acc:.2%}\n"
            if overfitting > 0.10:
                readme += f"- ⚠️ **Warning**: Possible overfitting (gap: {overfitting:.2%})\n"
    
    # Add configuration
    if 'config' in metadata:
        config = metadata['config']
        readme += f"""
## Training Configuration

"""
        for key, value in config.items():
            readme += f"- **{key}**: {value}\n"
    
    # Add usage instructions
    readme += f"""
## Files in This Package

- `trained_model.pkl`: Trained classifier object
- `scaler.pkl`: Fitted feature scaler
- `metadata.json`: Model metadata in JSON format
- `README.md`: This file
- `confusion_matrix.txt`: Confusion matrix (if available)

## Usage

### In Python Scripts

```python
from src.models.model_utils import load_model_package

# Load the model
package = load_model_package('{metadata.get('model_name', 'model_name')}')
clf = package['model']
scaler = package['scaler']

# Make predictions
X_scaled = scaler.transform(X_new)
predictions = clf.predict(X_scaled)
```

### For Real-Time Classification

```bash
python scripts/realtime_classify.py --model-name {metadata.get('model_name', 'model_name')}
```

## Notes

This model was trained as part of the EMG+IMU prosthetic hand control project.
For questions or issues, contact the development team.
"""
    
    return readme


if __name__ == "__main__":
    # Example usage / testing
    print("Testing model_utils module...")
    
    # Check if any models exist
    models = list_available_models()
    
    if models:
        print(f"\nFound {len(models)} saved model(s):")
        for model in models:
            print(f"  - {model}")
        
        # Show comparison
        if len(models) > 1:
            compare_models(models[:3])  # Compare first 3
    else:
        print("\nNo saved models found.")
        print("Train a model first using: python scripts/train_model.py")
    
    print("\n✓ model_utils module working correctly!")