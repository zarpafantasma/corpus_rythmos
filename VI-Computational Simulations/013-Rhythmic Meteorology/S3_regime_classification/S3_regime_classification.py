#!/usr/bin/env python3
"""
S3: Atmospheric Regime Classification by α
==========================================

RTM-Atmo predicts that α serves as a transport class indicator,
allowing automatic classification of atmospheric regimes:

- α < 1.5: Advective/fragmented (convective, disturbed)
- α ~ 1.5-2.0: Hierarchical (fronts, baroclinic waves)
- α ~ 2.0-2.5: Coherent (mature cyclones, organized systems)
- α > 2.5: Strongly coherent (blocking, persistent patterns)

This simulation:
1. Demonstrates regime classification based on α
2. Shows transitions between classes
3. Validates classification skill
4. Maps α to operational weather types

THEORETICAL MODEL - requires validation with operational analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# REGIME CLASSES
# =============================================================================

REGIME_CLASSES = {
    'Advective': {
        'alpha_range': (0.8, 1.5),
        'examples': ['Tropical disturbance', 'Disorganized convection', 'Trade wind flow'],
        'color': 'orange',
        'description': 'Fast decorrelation, fragmented structures'
    },
    'Hierarchical': {
        'alpha_range': (1.5, 2.0),
        'examples': ['Baroclinic wave', 'Frontal zone', 'MCS'],
        'color': 'blue',
        'description': 'Moderate organization, multi-scale interaction'
    },
    'Coherent': {
        'alpha_range': (2.0, 2.5),
        'examples': ['Mature cyclone', 'Strong jet', 'Organized storm'],
        'color': 'green',
        'description': 'Organized, persistent features'
    },
    'Strongly Coherent': {
        'alpha_range': (2.5, 3.5),
        'examples': ['Blocking high', 'Cut-off low', 'Persistent ridge'],
        'color': 'purple',
        'description': 'Quasi-stationary, long-lived patterns'
    }
}


def classify_regime(alpha):
    """Classify regime based on α value."""
    for class_name, params in REGIME_CLASSES.items():
        if params['alpha_range'][0] <= alpha < params['alpha_range'][1]:
            return class_name
    if alpha >= 2.5:
        return 'Strongly Coherent'
    return 'Advective'


# =============================================================================
# WEATHER PATTERN DATABASE
# =============================================================================

WEATHER_PATTERNS = {
    # Advective
    'Easterly Wave': {'alpha': 1.2, 'true_class': 'Advective'},
    'Trade Cumulus': {'alpha': 1.1, 'true_class': 'Advective'},
    'Sea Breeze': {'alpha': 1.3, 'true_class': 'Advective'},
    'Afternoon Thunderstorms': {'alpha': 1.4, 'true_class': 'Advective'},
    
    # Hierarchical
    'Cold Front': {'alpha': 1.7, 'true_class': 'Hierarchical'},
    'Warm Front': {'alpha': 1.6, 'true_class': 'Hierarchical'},
    'Squall Line': {'alpha': 1.8, 'true_class': 'Hierarchical'},
    'Developing Low': {'alpha': 1.9, 'true_class': 'Hierarchical'},
    'MCS': {'alpha': 1.7, 'true_class': 'Hierarchical'},
    
    # Coherent
    'Mature Extratropical': {'alpha': 2.2, 'true_class': 'Coherent'},
    'Category 1 Hurricane': {'alpha': 2.1, 'true_class': 'Coherent'},
    'Jet Streak': {'alpha': 2.3, 'true_class': 'Coherent'},
    'Subtropical High': {'alpha': 2.2, 'true_class': 'Coherent'},
    
    # Strongly Coherent
    'Blocking High': {'alpha': 2.8, 'true_class': 'Strongly Coherent'},
    'Major Hurricane': {'alpha': 2.6, 'true_class': 'Strongly Coherent'},
    'Rex Block': {'alpha': 3.0, 'true_class': 'Strongly Coherent'},
    'Polar Vortex': {'alpha': 2.9, 'true_class': 'Strongly Coherent'}
}


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_regime_transitions(duration_hours=240, seed=None):
    """
    Simulate α evolution through regime transitions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(duration_hours)
    alpha = np.zeros(duration_hours)
    regimes = []
    
    # Define regime sequence
    transitions = [
        (0, 60, 'Advective', 1.2),
        (60, 100, 'Hierarchical', 1.7),
        (100, 160, 'Coherent', 2.3),
        (160, 200, 'Strongly Coherent', 2.8),
        (200, 240, 'Coherent', 2.2)
    ]
    
    for start, end, regime, alpha_base in transitions:
        alpha[start:end] = alpha_base + 0.1 * np.random.randn(end - start)
        regimes.extend([regime] * (end - start))
    
    return t, alpha, regimes


def generate_classification_dataset(n_samples=200, noise=0.15, seed=None):
    """
    Generate dataset for classification validation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    
    for pattern_name, params in WEATHER_PATTERNS.items():
        n_per_pattern = n_samples // len(WEATHER_PATTERNS)
        
        for _ in range(n_per_pattern):
            alpha_measured = params['alpha'] + noise * np.random.randn()
            alpha_measured = max(0.8, min(3.5, alpha_measured))
            
            predicted_class = classify_regime(alpha_measured)
            
            data.append({
                'pattern': pattern_name,
                'alpha_true': params['alpha'],
                'alpha_measured': alpha_measured,
                'true_class': params['true_class'],
                'predicted_class': predicted_class
            })
    
    return pd.DataFrame(data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Atmospheric Regime Classification by α")
    print("=" * 70)
    
    output_dir = "/home/claude/020-Rhythmic_Meteorology/S3_regime_classification/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Part 1: α class boundaries
    # ===================
    
    print("\n1. Showing α class boundaries...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: α distribution by class
    ax = axes1[0, 0]
    
    alpha_range = np.linspace(0.5, 3.5, 100)
    
    for class_name, params in REGIME_CLASSES.items():
        low, high = params['alpha_range']
        ax.axvspan(low, high, alpha=0.3, color=params['color'], label=class_name)
        ax.axvline(x=low, color=params['color'], linestyle='--', alpha=0.7)
    
    # Mark weather patterns
    for pattern, params in WEATHER_PATTERNS.items():
        color = REGIME_CLASSES[params['true_class']]['color']
        ax.scatter([params['alpha']], [0.5], s=80, c=color, zorder=5)
    
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title('RTM-Atmo Regime Classification\nα determines transport class', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Weather patterns by α
    ax = axes1[0, 1]
    
    patterns = list(WEATHER_PATTERNS.keys())
    alphas = [WEATHER_PATTERNS[p]['alpha'] for p in patterns]
    colors = [REGIME_CLASSES[WEATHER_PATTERNS[p]['true_class']]['color'] for p in patterns]
    
    sorted_idx = np.argsort(alphas)
    patterns_sorted = [patterns[i] for i in sorted_idx]
    alphas_sorted = [alphas[i] for i in sorted_idx]
    colors_sorted = [colors[i] for i in sorted_idx]
    
    ax.barh(range(len(patterns_sorted)), alphas_sorted, color=colors_sorted, alpha=0.7)
    ax.set_yticks(range(len(patterns_sorted)))
    ax.set_yticklabels(patterns_sorted, fontsize=8)
    ax.set_xlabel('α', fontsize=11)
    ax.set_title('Weather Patterns by α\n(Color = regime class)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add class boundaries
    for class_name, params in REGIME_CLASSES.items():
        ax.axvline(x=params['alpha_range'][0], color='gray', linestyle=':', alpha=0.5)
    
    # Plot 3: Time series with regime transitions
    ax = axes1[1, 0]
    
    t, alpha, regimes = simulate_regime_transitions(seed=42)
    
    # Color by regime
    regime_colors = [REGIME_CLASSES[r]['color'] for r in regimes]
    
    for i in range(len(t) - 1):
        ax.plot([t[i], t[i+1]], [alpha[i], alpha[i+1]], 
                color=regime_colors[i], linewidth=2)
    
    # Add class boundaries
    for class_name, params in REGIME_CLASSES.items():
        ax.axhline(y=params['alpha_range'][0], color=params['color'], 
                   linestyle='--', alpha=0.5, label=class_name)
    
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('α', fontsize=11)
    ax.set_title('Regime Transitions Over Time\nα tracks weather evolution', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Classification accuracy
    ax = axes1[1, 1]
    
    df = generate_classification_dataset(n_samples=400, noise=0.15, seed=42)
    
    # Compute confusion matrix
    true_labels = df['true_class']
    pred_labels = df['predicted_class']
    
    classes = ['Advective', 'Hierarchical', 'Coherent', 'Strongly Coherent']
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.split()[0] for c in classes], rotation=45, ha='right')
    ax.set_yticklabels([c.split()[0] for c in classes])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Classification Confusion Matrix\n(Normalized)', fontsize=12)
    
    # Add accuracy values
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_regime_classification.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_regime_classification.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Detailed classification analysis
    # ===================
    
    print("\n2. Computing classification metrics...")
    
    # Overall accuracy
    accuracy = (df['true_class'] == df['predicted_class']).mean()
    
    # Per-class metrics
    class_metrics = []
    for class_name in classes:
        mask_true = df['true_class'] == class_name
        mask_pred = df['predicted_class'] == class_name
        
        tp = ((df['true_class'] == class_name) & (df['predicted_class'] == class_name)).sum()
        fp = ((df['true_class'] != class_name) & (df['predicted_class'] == class_name)).sum()
        fn = ((df['true_class'] == class_name) & (df['predicted_class'] != class_name)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    df_metrics = pd.DataFrame(class_metrics)
    
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df_metrics))
    width = 0.25
    
    ax.bar([i - width for i in x], df_metrics['precision'], width, 
           label='Precision', alpha=0.7)
    ax.bar([i for i in x], df_metrics['recall'], width, 
           label='Recall', alpha=0.7)
    ax.bar([i + width for i in x], df_metrics['f1_score'], width, 
           label='F1 Score', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.split()[0] for c in df_metrics['class']])
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Classification Performance by Regime\nOverall Accuracy: {accuracy:.1%}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_class_metrics.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df.to_csv(os.path.join(output_dir, 'S3_classification_data.csv'), index=False)
    df_metrics.to_csv(os.path.join(output_dir, 'S3_class_metrics.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S3: Atmospheric Regime Classification by α
===========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ATMO CLASSIFICATION SCHEME
------------------------------
α serves as a transport class indicator:

Advective (α = 0.8-1.5):
  - Fast decorrelation, fragmented
  - Examples: Easterly waves, trade cumulus, disturbances

Hierarchical (α = 1.5-2.0):
  - Moderate organization, multi-scale
  - Examples: Fronts, baroclinic waves, MCS

Coherent (α = 2.0-2.5):
  - Organized, persistent features
  - Examples: Mature cyclones, jets

Strongly Coherent (α = 2.5-3.5):
  - Quasi-stationary, long-lived
  - Examples: Blocking, major hurricanes

WEATHER PATTERN α VALUES
------------------------
"""
    
    for pattern, params in sorted(WEATHER_PATTERNS.items(), key=lambda x: x[1]['alpha']):
        summary += f"{pattern}: α = {params['alpha']:.1f} ({params['true_class']})\n"
    
    summary += f"""
CLASSIFICATION PERFORMANCE
--------------------------
Overall Accuracy: {accuracy:.1%}

Per-Class Metrics:
"""
    
    for _, row in df_metrics.iterrows():
        summary += f"  {row['class']}: P={row['precision']:.2f}, R={row['recall']:.2f}, F1={row['f1_score']:.2f}\n"
    
    summary += f"""
OPERATIONAL USE
---------------
1. Compute rolling α from reanalysis/satellite data
2. Classify regime automatically
3. Use regime for:
   - Forecast guidance (persistence vs. change)
   - Warning thresholds (regime-dependent)
   - Verification stratification

TRANSITION DETECTION
--------------------
Class changes (e.g., Advective → Hierarchical) indicate:
- Development (increasing α)
- Decay (decreasing α)
- Reorganization (non-monotonic α)

These transitions are key forecasting targets.
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall classification accuracy: {accuracy:.1%}")
    print("\nPer-class F1 scores:")
    for _, row in df_metrics.iterrows():
        print(f"  {row['class']}: {row['f1_score']:.2f}")
    print(f"\nOutputs: {output_dir}/")
    
    return df, df_metrics, accuracy


if __name__ == "__main__":
    main()
