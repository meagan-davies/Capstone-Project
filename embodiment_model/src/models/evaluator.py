"""
Model evaluation and validation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Optional
from pathlib import Path

from .regressor import EmbodimentRegressor


class EmbodimentEvaluator:
    """
    Evaluate embodiment model performance
    """
    
    def __init__(self, model: EmbodimentRegressor):
        """
        Initialize evaluator
        
        Args:
            model: Trained EmbodimentRegressor
        """
        self.model = model
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                participant_ids: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X: Feature matrix
            y_true: True embodiment scores
            participant_ids: Optional participant IDs
        
        Returns:
            Dict of evaluation metrics
        """
        # Predict
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        results = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true),
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Per-participant metrics if IDs provided
        if participant_ids is not None:
            results['per_participant'] = self._evaluate_per_participant(
                y_true, y_pred, participant_ids
            )
        
        return results
    
    def _evaluate_per_participant(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  participant_ids: np.ndarray) -> Dict:
        """Calculate per-participant metrics"""
        per_participant = {}
        
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            
            if np.sum(mask) >= 2:  # Need at least 2 samples
                per_participant[pid] = {
                    'r2': r2_score(y_true[mask], y_pred[mask]),
                    'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'n_samples': np.sum(mask)
                }
        
        return per_participant
    
    def print_evaluation(self, results: Dict):
        """
        Print evaluation results
        
        Args:
            results: Results from evaluate()
        """
        print(f"\n{'='*60}")
        print("Model Evaluation Results")
        print(f"{'='*60}")
        print(f"R² Score:       {results['r2']:.3f}")
        print(f"MAE:            {results['mae']:.2f} points")
        print(f"RMSE:           {results['rmse']:.2f} points")
        print(f"Correlation:    {results['correlation']:.3f}")
        print(f"Mean Error:     {results['mean_error']:.2f} points")
        print(f"Std Error:      {results['std_error']:.2f} points")
        
        if 'per_participant' in results:
            print(f"\nPer-Participant Performance:")
            print(f"{'Participant':<15} {'R²':<10} {'MAE':<10} {'N':<5}")
            print(f"{'-'*45}")
            
            for pid, metrics in results['per_participant'].items():
                print(f"{pid:<15} {metrics['r2']:<10.3f} {metrics['mae']:<10.2f} {metrics['n_samples']:<5}")
        
        print(f"{'='*60}\n")
    
    def plot_results(self, results: Dict, save_path: Optional[Path] = None):
        """
        Plot evaluation results
        
        Args:
            results: Results from evaluate()
            save_path: Optional path to save figure
        """
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('True Embodiment Score')
        axes[0, 0].set_ylabel('Predicted Embodiment Score')
        axes[0, 0].set_title(f"Predicted vs Actual (R² = {results['r2']:.3f})")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_xlim([0, 100])
        axes[0, 0].set_ylim([0, 100])
        
        # Plot 2: Residuals
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Embodiment Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Error distribution
        axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Prediction Error (points)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f"Error Distribution (MAE = {results['mae']:.2f})")
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Per-participant performance (if available)
        if 'per_participant' in results:
            pids = list(results['per_participant'].keys())
            r2_values = [results['per_participant'][pid]['r2'] for pid in pids]
            
            axes[1, 1].bar(range(len(pids)), r2_values, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Participant')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Per-Participant Performance')
            axes[1, 1].set_xticks(range(len(pids)))
            axes[1, 1].set_xticklabels(pids, rotation=45, ha='right')
            axes[1, 1].axhline(y=0, color='k', linestyle='-', lw=0.5)
            axes[1, 1].grid(alpha=0.3, axis='y')
        else:
            axes[1, 1].text(0.5, 0.5, 'No per-participant data', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[Path] = None):
        """
        Plot feature importance
        
        Args:
            top_k: Number of top features to plot
            save_path: Optional path to save figure
        """
        importance = self.model.get_feature_importance()
        
        if importance is None:
            print("Feature importance not available for this model type")
            return
        
        if self.model.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        else:
            feature_names = self.model.feature_names
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:top_k]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, importance[sorted_idx], edgecolor='black', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Most Important Features')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_control_accuracy_relationship(self, X: np.ndarray, 
                                             y_true: np.ndarray,
                                             control_feature_idx: int):
        """
        Analyze relationship between control accuracy and embodiment
        
        Args:
            X: Feature matrix
            y_true: True embodiment scores
            control_feature_idx: Index of control accuracy feature
        """
        y_pred = self.model.predict(X)
        control_accuracy = X[:, control_feature_idx]
        
        # Calculate correlations
        corr_true = np.corrcoef(control_accuracy, y_true)[0, 1]
        corr_pred = np.corrcoef(control_accuracy, y_pred)[0, 1]
        
        print(f"\nControl Accuracy vs Embodiment:")
        print(f"  Correlation (true):      {corr_true:.3f}")
        print(f"  Correlation (predicted): {corr_pred:.3f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # True embodiment
        axes[0].scatter(control_accuracy, y_true, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].set_xlabel('Control Accuracy')
        axes[0].set_ylabel('True Embodiment Score')
        axes[0].set_title(f'Control vs True Embodiment (r = {corr_true:.3f})')
        axes[0].grid(alpha=0.3)
        
        # Predicted embodiment
        axes[1].scatter(control_accuracy, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1].set_xlabel('Control Accuracy')
        axes[1].set_ylabel('Predicted Embodiment Score')
        axes[1].set_title(f'Control vs Predicted Embodiment (r = {corr_pred:.3f})')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def cross_validate_conditions(model: EmbodimentRegressor,
                              X: np.ndarray,
                              y: np.ndarray,
                              conditions: np.ndarray) -> Dict:
    """
    Evaluate model performance across different experimental conditions
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True scores
        conditions: Condition labels
    
    Returns:
        Dict of per-condition results
    """
    results = {}
    
    print(f"\nEvaluating across conditions...")
    
    for condition in np.unique(conditions):
        mask = conditions == condition
        
        if np.sum(mask) < 2:
            continue
        
        X_cond = X[mask]
        y_cond = y[mask]
        
        y_pred = model.predict(X_cond)
        
        results[condition] = {
            'r2': r2_score(y_cond, y_pred),
            'mae': mean_absolute_error(y_cond, y_pred),
            'n_samples': np.sum(mask),
            'mean_true': np.mean(y_cond),
            'mean_pred': np.mean(y_pred)
        }
        
        print(f"  {condition:<20} R²={results[condition]['r2']:.3f}  MAE={results[condition]['mae']:.2f}  N={results[condition]['n_samples']}")
    
    return results