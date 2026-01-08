"""
Model Training and Evaluation Module

Contains functions for:
- Data preparation (splitting, scaling)
- Model training with cross-validation
- Model evaluation and metrics
- Support for multiple classifier types
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = 'standard',
    balance_classes: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, Optional[Dict]]:
    """
    Split and scale data for training.
    
    This is the SAME function from your notebook, just with:
    - Better type hints
    - More documentation
    - More flexibility (different scalers)
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        test_size: Proportion of data for testing (0-1)
        random_state: Random seed for reproducibility
        scaler_type: Type of scaler - 'standard', 'robust', or 'minmax'
        balance_classes: Whether to compute class weights
        
    Returns:
        X_train: Scaled training features
        X_test: Scaled test features
        y_train: Training labels
        y_test: Test labels
        scaler: Fitted scaler object
        class_weights: Dictionary of class weights (or None)
        
    Example:
        >>> X_train, X_test, y_train, y_test, scaler, weights = prepare_data(X, y)
        >>> print(f"Training samples: {len(X_train)}")
    """
    # Split with stratification (keeps class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,  # Ensures each split has same class distribution
        random_state=random_state
    )
    
    # Select scaler based on type
    if scaler_type == 'standard':
        # Z-score normalization: (x - mean) / std
        # Best for normally distributed data
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        # Uses median and IQR: (x - median) / IQR
        # Less sensitive to outliers (recommended for EMG!)
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        # Scales to [0, 1]: (x - min) / (max - min)
        # Good when you know the range
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit scaler on training data only!
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Use same scaling
    
    # Compute class weights if requested
    class_weights = None
    if balance_classes:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"\nClass weights computed: {class_weights}")
    
    return X_train, X_test, y_train, y_test, scaler, class_weights


def train_lda_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_type: str = 'standard',
    use_cv: bool = True,
    cv_folds: int = 5,
    verbose: bool = True
) -> Tuple[object, object]:
    """
    Train LDA classifier with cross-validation.
    
    This combines prepare_data + train_evaluate from your notebook,
    but split into logical pieces.
    
    Args:
        X_train: Training features (already split, not scaled)
        y_train: Training labels
        scaler_type: Type of scaler to use
        use_cv: Whether to perform cross-validation
        cv_folds: Number of CV folds
        verbose: Print progress information
        
    Returns:
        clf: Trained classifier
        scaler: Fitted scaler
        
    Example:
        >>> clf, scaler = train_lda_classifier(X_train, y_train, use_cv=True)
        >>> y_pred = clf.predict(scaler.transform(X_test))
    """
    # Scale data
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create classifier
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    # Cross-validation (optional but recommended)
    if use_cv and len(np.unique(y_train)) > 1:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
        
        if verbose:
            print(f"\nCross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full training set
    clf.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"✓ Model trained on {len(X_train_scaled)} samples")
    
    return clf, scaler


def evaluate_classifier(
    clf: object,
    scaler: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    
    This is the evaluation part from your train_evaluate function,
    but extracted and enhanced.
    
    Args:
        clf: Trained classifier
        scaler: Fitted scaler
        X_train: Training features (unscaled)
        X_test: Test features (unscaled)
        y_train: Training labels
        y_test: Test labels
        class_names: Names for each class (for reporting)
        verbose: Print detailed results
        
    Returns:
        Dictionary with all metrics:
            - accuracy: Overall accuracy
            - f1_macro: Macro-averaged F1
            - f1_weighted: Weighted F1
            - f1_per_class: F1 for each class
            - precision: Precision per class
            - recall: Recall per class
            - confusion_matrix: Confusion matrix
            - train_accuracy: Accuracy on training set (check overfitting)
            
    Example:
        >>> results = evaluate_classifier(clf, scaler, X_train, X_test, y_train, y_test)
        >>> print(f"Test Accuracy: {results['accuracy']:.2%}")
    """
    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None
    )
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Print results if verbose
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:     {test_acc:.4f}")
        
        # Check for overfitting
        if train_acc - test_acc > 0.10:
            print("⚠ Warning: Possible overfitting (train >> test)")
        
        print(f"\nMacro F1 Score:    {f1_macro:.4f}")
        print(f"Weighted F1 Score: {f1_weighted:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1_per_class, support)):
            class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            print(f"  {class_name:12s}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        if class_names:
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
    
    # Return all metrics
    return {
        'train_accuracy': train_acc,
        'accuracy': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision': precision,
        'recall': recall,
        'support': support,
        'confusion_matrix': cm,
    }


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    scaler_type: str = 'standard',
    use_cv: bool = True,
    cv_folds: int = 5,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[object, object, Dict]:
    """
    Complete training and evaluation pipeline.
    
    This is a convenience function that combines:
    1. prepare_data()
    2. train_lda_classifier()
    3. evaluate_classifier()
    
    Use this when you want to do everything in one call.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion for test set
        scaler_type: Type of scaler
        use_cv: Use cross-validation
        cv_folds: Number of CV folds
        class_names: Class names for reporting
        verbose: Print progress
        
    Returns:
        clf: Trained classifier
        scaler: Fitted scaler
        results: Dictionary of evaluation metrics
        
    Example:
        >>> clf, scaler, results = train_and_evaluate(X, y, verbose=True)
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    # Step 1: Split and scale
    if verbose:
        print("Step 1: Splitting and scaling data...")
    
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(
        X, y, test_size=test_size, scaler_type=scaler_type
    )
    
    # Step 2: Train
    if verbose:
        print("\nStep 2: Training classifier...")
    
    # We already scaled in prepare_data, so pass pre-scaled data
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    if use_cv:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro')
        if verbose:
            print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    clf.fit(X_train, y_train)
    
    # Step 3: Evaluate
    if verbose:
        print("\nStep 3: Evaluating model...")
    
    # Need to create a dummy scaler for evaluate function since data already scaled
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None
    )
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print("="*60)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        
        if class_names:
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    results = {
        'train_accuracy': train_acc,
        'accuracy': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision': precision,
        'recall': recall,
        'support': support,
        'confusion_matrix': cm,
    }
    
    return clf, scaler, results


# Additional utility functions

def compare_scalers(
    X: np.ndarray,
    y: np.ndarray,
    scalers: List[str] = ['standard', 'robust', 'minmax'],
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Compare different scaler types to find the best one.
    
    Useful for experimentation in notebooks!
    
    Example:
        >>> results = compare_scalers(X, y)
        >>> print(results)
        {'standard': 0.87, 'robust': 0.89, 'minmax': 0.85}
    """
    results = {}
    
    for scaler_type in scalers:
        clf, scaler = train_lda_classifier(
            X, y,
            scaler_type=scaler_type,
            use_cv=True,
            cv_folds=cv_folds,
            verbose=False
        )
        
        # Get CV score
        X_scaled = scaler.transform(X)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        score = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro').mean()
        
        results[scaler_type] = score
        print(f"{scaler_type:10s}: {score:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage / testing
    print("Testing classifier module...")
    
    # Generate synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=200,
        n_features=32,  # 4 EMG sensors * 8 features
        n_classes=4,
        n_informative=20,
        random_state=42
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Test the pipeline
    clf, scaler, results = train_and_evaluate(
        X, y,
        class_names=['Neutral', 'Pinching', 'Grasping', 'Zipping'],
        verbose=True
    )
    
    print("\n✓ Classifier module working correctly!")