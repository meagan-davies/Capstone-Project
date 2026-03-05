"""
Model Training

Provides:
- Training pipeline
- Cross-validation
- Evaluation metrics
- Model bundle container
- Feature validation utilities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Model Bundle Dataclass
@dataclass
class ModelBundle:
    """
    Container for trained model and metadata.
    """

    pipeline: Pipeline
    feature_count: int
    class_names: Optional[List[str]]
    cv_scores: Optional[np.ndarray]
    scaler_type: str
    n_classes: int

    def predict(self, X: np.ndarray):
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self.pipeline.predict_proba(X)

# Pipeline Builder
def build_pipeline(scaler_type: str = "robust") -> Pipeline:
    """
    Create sklearn pipeline with scaler + LDA classifier.
    """

    scaler_map = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
    }

    if scaler_type not in scaler_map:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    pipeline = Pipeline([
        ("scaler", scaler_map[scaler_type]),
        ("clf", LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )),
    ])

    return pipeline

# Training Function
def train_model(X: np.ndarray, y: np.ndarray, scaler_type: str = "robust", test_size: float = 0.3, random_state: int = 42,
    cv_folds: int = 5, class_names: Optional[List[str]] = None, verbose: bool = True,) -> Tuple[ModelBundle, Dict, Dict]:
    """
    Train model with cross-validation and evaluation.

    Returns:
        bundle: ModelBundle
        metrics: evaluation metrics dictionary
        config: training configuration dictionary
    """

    if verbose:
        print("\n===== Training Model =====")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Pipeline
    pipeline = build_pipeline(scaler_type=scaler_type)

    # Cross-validation
    cv_scores = None

    if cv_folds > 1 and len(np.unique(y_train)) > 1:
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
        )

        if verbose:
            print(f"CV F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit Final Model
    pipeline.fit(X_train, y_train)

    if verbose:
        print(f"✓ Model trained on {len(X_train)} samples")

    # Evaluation
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    f1_macro = f1_score(y_test, y_test_pred, average="macro")
    f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test,
        y_test_pred,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_test_pred)

    if verbose:

        print("\n===== Evaluation =====")

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")

        if train_acc - test_acc > 0.10:
            print("⚠ Possible overfitting detected")

        print(f"\nMacro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")

        if class_names:
            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    y_test_pred,
                    target_names=class_names,
                    zero_division=0,
                )
            )

    # Metrics Dictionary
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1_per_class,
        "support": support,
        "confusion_matrix": cm,
        "cv_scores": cv_scores,
    }

    # Training Config
    config = {
        "scaler_type": scaler_type,
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }

    # Create Bundle
    bundle = ModelBundle(
        pipeline=pipeline,
        feature_count=X.shape[1],
        class_names=class_names,
        cv_scores=cv_scores,
        scaler_type=scaler_type,
        n_classes=len(np.unique(y)),
    )

    return bundle, metrics, config


# Feature Validation Utility
def validate_feature_shape(bundle: ModelBundle, X: np.ndarray):
    """
    Ensure feature size matches training.
    Prevents realtime bugs.
    """

    if X.shape[1] != bundle.feature_count:
        raise RuntimeError(
            f"Feature mismatch: got {X.shape[1]}, "
            f"expected {bundle.feature_count}"
        )

# Development Test
if __name__ == "__main__":

    print("Testing training module...")

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=32,
        n_classes=4,
        n_informative=20,
        random_state=42,
    )

    bundle, metrics, config = train_model(
        X,
        y,
        class_names=["Neutral", "Pinch", "Grasp", "Zip"],
        verbose=True,
    )

    print("\n✓ Training module working")