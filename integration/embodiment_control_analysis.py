"""
Analyze relationship between control accuracy and embodiment
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "embodiment_model"))

from src.data.data_loader import load_embodiment_sessions
from src.data.dataset_builder import build_feature_matrix
from src.models.trainer import load_model


def analyze_control_embodiment_relationship(model_path: str, data_dir: str, output_dir: str = None):
    """
    Analyze relationship between control accuracy and embodiment
    
    Args:
        model_path: Path to trained embodiment model
        data_dir: Directory containing embodiment data
        output_dir: Optional output directory for plots
    """
    print("\n" + "="*70)
    print(" "*10 + "CONTROL ACCURACY vs EMBODIMENT ANALYSIS")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, cv_results, metadata = load_model(model_path)
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    sessions = load_embodiment_sessions(data_dir)
    
    if len(sessions) == 0:
        print("❌ No sessions found")
        return
    
    # Build feature matrix
    print("\nBuilding feature matrix...")
    X, y, participant_ids, feature_names = build_feature_matrix(sessions)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X)
    
    # Extract control accuracy features
    control_feature_indices = [
        i for i, name in enumerate(feature_names)
        if any(keyword in name.lower() for keyword in ['tracking', 'smoothness', 'latency', 'control', 'error'])
    ]
    
    if not control_feature_indices:
        print("⚠ No control accuracy features found")
        return
    
    print(f"\nFound {len(control_feature_indices)} control accuracy features:")
    for idx in control_feature_indices:
        print(f"  - {feature_names[idx]}")
    
    # Analysis
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    results = {}
    
    for idx in control_feature_indices:
        feature_name = feature_names[idx]
        feature_values = X[:, idx]
        
        # Correlation with true embodiment
        corr_true, p_true = pearsonr(feature_values, y)
        
        # Correlation with predicted embodiment
        corr_pred, p_pred = pearsonr(feature_values, y_pred)
        
        results[feature_name] = {
            'corr_true': corr_true,
            'p_true': p_true,
            'corr_pred': corr_pred,
            'p_pred': p_pred
        }
        
        print(f"\n{feature_name}:")
        print(f"  True embodiment:      r = {corr_true:6.3f}, p = {p_true:.4f}")
        print(f"  Predicted embodiment: r = {corr_pred:6.3f}, p = {p_pred:.4f}")
    
    # Condition analysis
    print("\n" + "="*70)
    print("ANALYSIS BY CONDITION")
    print("="*70)
    
    conditions = np.array([s.condition for s in sessions])
    
    condition_results = {}
    for condition in np.unique(conditions):
        mask = conditions == condition
        
        if np.sum(mask) < 3:
            continue
        
        y_cond = y[mask]
        y_pred_cond = y_pred[mask]
        
        # Mean control accuracy for this condition
        control_means = {}
        for idx in control_feature_indices:
            control_means[feature_names[idx]] = np.mean(X[mask, idx])
        
        condition_results[condition] = {
            'n_samples': np.sum(mask),
            'embodiment_true_mean': np.mean(y_cond),
            'embodiment_true_std': np.std(y_cond),
            'embodiment_pred_mean': np.mean(y_pred_cond),
            'embodiment_pred_std': np.std(y_pred_cond),
            'control_features': control_means
        }
        
        print(f"\n{condition}:")
        print(f"  N = {condition_results[condition]['n_samples']}")
        print(f"  True embodiment:  {condition_results[condition]['embodiment_true_mean']:.1f} ± {condition_results[condition]['embodiment_true_std']:.1f}")
        print(f"  Pred embodiment:  {condition_results[condition]['embodiment_pred_mean']:.1f} ± {condition_results[condition]['embodiment_pred_std']:.1f}")
    
    # Visualization
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Control accuracy vs true embodiment (scatter)
    if control_feature_indices:
        main_control_idx = control_feature_indices[0]
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X[:, main_control_idx], y, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax1.set_xlabel(feature_names[main_control_idx])
        ax1.set_ylabel('True Embodiment Score')
        ax1.set_title(f'Control vs True Embodiment\n(r = {results[feature_names[main_control_idx]]["corr_true"]:.3f})')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Control accuracy vs predicted embodiment
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(X[:, main_control_idx], y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax2.set_xlabel(feature_names[main_control_idx])
        ax2.set_ylabel('Predicted Embodiment Score')
        ax2.set_title(f'Control vs Predicted Embodiment\n(r = {results[feature_names[main_control_idx]]["corr_pred"]:.3f})')
        ax2.grid(alpha=0.3)
    
    # Plot 3: Correlation heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    control_names = [feature_names[i] for i in control_feature_indices]
    corr_matrix = np.array([[results[name]['corr_true'], results[name]['corr_pred']] 
                           for name in control_names])
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=['True', 'Predicted'], yticklabels=control_names,
                cbar_kws={'label': 'Correlation'}, ax=ax3)
    ax3.set_title('Control Features vs Embodiment')
    
    # Plot 4: Embodiment by condition
    ax4 = fig.add_subplot(gs[1, :])
    condition_names = list(condition_results.keys())
    true_means = [condition_results[c]['embodiment_true_mean'] for c in condition_names]
    true_stds = [condition_results[c]['embodiment_true_std'] for c in condition_names]
    pred_means = [condition_results[c]['embodiment_pred_mean'] for c in condition_names]
    pred_stds = [condition_results[c]['embodiment_pred_std'] for c in condition_names]
    
    x = np.arange(len(condition_names))
    width = 0.35
    
    ax4.bar(x - width/2, true_means, width, yerr=true_stds, label='True', alpha=0.7, capsize=5)
    ax4.bar(x + width/2, pred_means, width, yerr=pred_stds, label='Predicted', alpha=0.7, capsize=5)
    ax4.set_xlabel('Condition')
    ax4.set_ylabel('Embodiment Score')
    ax4.set_title('Mean Embodiment by Condition')
    ax4.set_xticks(x)
    ax4.set_xticklabels(condition_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Plot 5: True vs Predicted scatter by condition
    ax5 = fig.add_subplot(gs[2, 0])
    for condition in np.unique(conditions):
        mask = conditions == condition
        ax5.scatter(y[mask], y_pred[mask], label=condition, alpha=0.6, s=50)
    
    ax5.plot([0, 100], [0, 100], 'k--', lw=2, alpha=0.5)
    ax5.set_xlabel('True Embodiment')
    ax5.set_ylabel('Predicted Embodiment')
    ax5.set_title('True vs Predicted by Condition')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    ax5.set_xlim([0, 100])
    ax5.set_ylim([0, 100])
    
    # Plot 6: Residuals vs control accuracy
    if control_feature_indices:
        ax6 = fig.add_subplot(gs[2, 1])
        residuals = y_pred - y
        ax6.scatter(X[:, main_control_idx], residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax6.axhline(y=0, color='r', linestyle='--', lw=2)
        ax6.set_xlabel(feature_names[main_control_idx])
        ax6.set_ylabel('Prediction Error')
        ax6.set_title('Residuals vs Control Accuracy')
        ax6.grid(alpha=0.3)
    
    # Plot 7: Distribution comparison
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(y, bins=20, alpha=0.5, label='True', edgecolor='black')
    ax7.hist(y_pred, bins=20, alpha=0.5, label='Predicted', edgecolor='black')
    ax7.set_xlabel('Embodiment Score')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Score Distributions')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    plt.suptitle('Control Accuracy vs Embodiment Analysis', fontsize=16, y=0.995)
    
    if output_dir:
        output_path = Path(output_dir) / "control_embodiment_analysis.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")
    
    return results, condition_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze control-embodiment relationship")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained embodiment model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to embodiment data directory")
    parser.add_argument("--output", type=str,
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    analyze_control_embodiment_relationship(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()