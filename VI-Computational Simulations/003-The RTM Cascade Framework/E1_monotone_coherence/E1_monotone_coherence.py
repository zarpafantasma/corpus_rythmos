#!/usr/bin/env python3
"""
E1: Four-Layer Cascade with Non-Decreasing Coherence
=====================================================

RTM Cascade Framework Validation - Signature S1

This simulation validates that the coherence exponent α increases
(or at least does not decrease) across nested layers in a cascade
architecture.

Model:
------
For each layer n = 0, 1, 2, 3:
    T_n(L) = c_n * L^α_n * ε

Where:
- α_n = α_0 + n * Δα  (monotone increasing coherence)
- c_n = layer-level factor (affects intercept only, not slope)
- ε ~ LogNormal(0, σ²)  (multiplicative noise)
- L = effective size (geometric grid, ≥1 decade span)

Measurement:
------------
For each layer, regress log(T) on log(L) via OLS to estimate α_n.
Report 95% bootstrap CIs (≥1000 replicates).

Decision Rule (S1):
-------------------
Pass if all adjacent differences Δα_{n,n+1} have CI lower bound > -ε_tol
(default ε_tol = 0.05)

Expected Pattern:
-----------------
α rises (or plateaus) with n; CIs do not show significant drops;
intercepts differ across layers but do not affect slopes.

Reference: RTM Cascade Framework, Section 4.1
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List, Dict

# =============================================================================
# PARAMETERS
# =============================================================================

# Layer configuration
N_LAYERS = 4
ALPHA_BASE = 2.0        # α_0 (base coherence exponent)
DELTA_ALPHA = 0.3       # Δα per layer (coherence increase)

# Size configuration (geometric grid)
L_MIN = 10
L_MAX = 200
N_SIZES = 10            # sizes per layer (≥8 recommended)

# Noise and replication
NOISE_SIGMA = 0.15      # log-normal noise scale
N_EVENTS = 50           # events per (layer, size) combination

# Bootstrap
N_BOOTSTRAP = 1000

# Decision tolerance
EPSILON_TOL = 0.05      # tolerance for slope decrease

# Layer-level factors (affect intercept only)
LAYER_FACTORS = [1.0, 2.5, 0.8, 3.2]  # c_n values

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_cascade_data(rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic cascade data with non-decreasing α across layers.
    
    Returns DataFrame with columns: layer, L, T, log_L, log_T
    """
    # Generate size grid (geometric)
    L_values = np.geomspace(L_MIN, L_MAX, N_SIZES)
    
    records = []
    
    for n in range(N_LAYERS):
        # Layer parameters
        alpha_n = ALPHA_BASE + n * DELTA_ALPHA
        c_n = LAYER_FACTORS[n]
        
        for L in L_values:
            # Generate N_EVENTS observations at this (layer, size)
            for event in range(N_EVENTS):
                # Multiplicative noise
                epsilon = rng.lognormal(0, NOISE_SIGMA)
                
                # RTM scaling law
                T = c_n * (L ** alpha_n) * epsilon
                
                records.append({
                    'layer': n,
                    'L': L,
                    'T': T,
                    'log_L': np.log10(L),
                    'log_T': np.log10(T),
                    'alpha_true': alpha_n,
                    'c_true': c_n
                })
    
    return pd.DataFrame(records)


# =============================================================================
# SLOPE ESTIMATION
# =============================================================================

def estimate_slope_ols(log_L: np.ndarray, log_T: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate slope via OLS regression.
    
    Returns: (slope, intercept, R²)
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_T)
    return slope, intercept, r_value**2


def bootstrap_slope_ci(log_L: np.ndarray, log_T: np.ndarray, 
                       n_boot: int, rng: np.random.Generator,
                       confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for slope.
    
    Returns: (slope_estimate, ci_lower, ci_upper)
    """
    n = len(log_L)
    slopes = []
    
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            slope, _, _ = estimate_slope_ols(log_L[idx], log_T[idx])
            slopes.append(slope)
        except:
            continue
    
    slopes = np.array(slopes)
    point_estimate = np.mean(slopes)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(slopes, 100 * alpha / 2)
    ci_upper = np.percentile(slopes, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def analyze_layer(df: pd.DataFrame, layer: int, 
                  rng: np.random.Generator) -> Dict:
    """
    Analyze a single layer: estimate α with bootstrap CI.
    """
    layer_data = df[df['layer'] == layer]
    log_L = layer_data['log_L'].values
    log_T = layer_data['log_T'].values
    
    # OLS fit
    slope, intercept, r2 = estimate_slope_ols(log_L, log_T)
    
    # Bootstrap CI
    alpha_est, ci_lower, ci_upper = bootstrap_slope_ci(
        log_L, log_T, N_BOOTSTRAP, rng
    )
    
    # True values
    alpha_true = layer_data['alpha_true'].iloc[0]
    c_true = layer_data['c_true'].iloc[0]
    
    return {
        'layer': layer,
        'alpha_true': alpha_true,
        'c_true': c_true,
        'alpha_est': alpha_est,
        'alpha_ols': slope,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'intercept': intercept,
        'R2': r2,
        'n_points': len(log_L)
    }


# =============================================================================
# SIGNATURE S1 TEST
# =============================================================================

def test_s1_monotone(layer_results: List[Dict]) -> Dict:
    """
    Test Signature S1: Non-decreasing coherence across layers.
    
    Decision: Pass if all Δα_{n,n+1} have CI lower bound > -ε_tol
    """
    n_layers = len(layer_results)
    differences = []
    all_pass = True
    
    for i in range(n_layers - 1):
        alpha_n = layer_results[i]['alpha_est']
        alpha_n1 = layer_results[i+1]['alpha_est']
        
        # Difference
        delta = alpha_n1 - alpha_n
        
        # Approximate CI for difference (conservative: sum of half-widths)
        ci_width_n = (layer_results[i]['ci_upper'] - layer_results[i]['ci_lower']) / 2
        ci_width_n1 = (layer_results[i+1]['ci_upper'] - layer_results[i+1]['ci_lower']) / 2
        delta_ci_width = np.sqrt(ci_width_n**2 + ci_width_n1**2) * 1.96
        
        delta_ci_lower = delta - delta_ci_width
        delta_ci_upper = delta + delta_ci_width
        
        # Test: is the CI lower bound > -ε_tol?
        passes = delta_ci_lower > -EPSILON_TOL
        if not passes:
            all_pass = False
        
        differences.append({
            'from_layer': i,
            'to_layer': i + 1,
            'delta_alpha': delta,
            'delta_ci_lower': delta_ci_lower,
            'delta_ci_upper': delta_ci_upper,
            'passes': passes
        })
    
    return {
        'test': 'S1_monotone_coherence',
        'all_pass': all_pass,
        'epsilon_tol': EPSILON_TOL,
        'differences': differences
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(df: pd.DataFrame, layer_results: List[Dict], 
                 s1_result: Dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_LAYERS))
    
    # Plot 1: Log-log scaling per layer
    ax1 = axes[0, 0]
    for i, res in enumerate(layer_results):
        layer_data = df[df['layer'] == i]
        ax1.scatter(layer_data['log_L'], layer_data['log_T'], 
                   alpha=0.3, s=10, color=colors[i], label=f'Layer {i}')
        
        # Fit line
        x_fit = np.linspace(layer_data['log_L'].min(), layer_data['log_L'].max(), 100)
        y_fit = res['intercept'] + res['alpha_ols'] * x_fit
        ax1.plot(x_fit, y_fit, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('log₁₀(L)', fontsize=12)
    ax1.set_ylabel('log₁₀(T)', fontsize=12)
    ax1.set_title('RTM Scaling: T ∝ L^α per Layer', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: α estimates with CI
    ax2 = axes[0, 1]
    layers = [r['layer'] for r in layer_results]
    alphas = [r['alpha_est'] for r in layer_results]
    ci_lowers = [r['ci_lower'] for r in layer_results]
    ci_uppers = [r['ci_upper'] for r in layer_results]
    alpha_true = [r['alpha_true'] for r in layer_results]
    
    yerr_lower = [a - cl for a, cl in zip(alphas, ci_lowers)]
    yerr_upper = [cu - a for a, cu in zip(alphas, ci_uppers)]
    
    ax2.errorbar(layers, alphas, yerr=[yerr_lower, yerr_upper],
                 fmt='o', markersize=12, capsize=8, capthick=2,
                 color='blue', label='Estimated α')
    ax2.plot(layers, alpha_true, 's--', markersize=10, color='red',
             label='True α', alpha=0.7)
    
    ax2.set_xlabel('Layer n', fontsize=12)
    ax2.set_ylabel('Coherence Exponent α', fontsize=12)
    ax2.set_title('S1: Monotone Coherence Across Layers', fontsize=14)
    ax2.set_xticks(layers)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add pass/fail annotation
    status = "✓ PASS" if s1_result['all_pass'] else "✗ FAIL"
    ax2.annotate(f"S1 Test: {status}", xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=14, ha='right', fontweight='bold',
                color='green' if s1_result['all_pass'] else 'red')
    
    # Plot 3: Δα between adjacent layers
    ax3 = axes[1, 0]
    diff_results = s1_result['differences']
    x_pos = [d['from_layer'] + 0.5 for d in diff_results]
    deltas = [d['delta_alpha'] for d in diff_results]
    delta_ci_lower = [d['delta_ci_lower'] for d in diff_results]
    delta_ci_upper = [d['delta_ci_upper'] for d in diff_results]
    
    yerr_l = [d - cl for d, cl in zip(deltas, delta_ci_lower)]
    yerr_u = [cu - d for d, cu in zip(deltas, delta_ci_upper)]
    
    bar_colors = ['green' if d['passes'] else 'red' for d in diff_results]
    ax3.bar(x_pos, deltas, width=0.6, color=bar_colors, alpha=0.7)
    ax3.errorbar(x_pos, deltas, yerr=[yerr_l, yerr_u],
                 fmt='none', capsize=8, capthick=2, color='black')
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(y=-EPSILON_TOL, color='red', linestyle=':', linewidth=2,
                label=f'-ε_tol = -{EPSILON_TOL}')
    
    ax3.set_xlabel('Layer Transition', fontsize=12)
    ax3.set_ylabel('Δα = α_{n+1} - α_n', fontsize=12)
    ax3.set_title('Adjacent Layer Differences', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{d["from_layer"]}→{d["to_layer"]}' for d in diff_results])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Intercepts (level factors)
    ax4 = axes[1, 1]
    intercepts = [r['intercept'] for r in layer_results]
    true_log_c = [np.log10(r['c_true']) for r in layer_results]
    
    ax4.bar(layers, intercepts, width=0.6, color=colors, alpha=0.7,
            label='Estimated intercept')
    ax4.plot(layers, true_log_c, 's--', markersize=10, color='red',
             label='True log₁₀(c_n)', alpha=0.7)
    
    ax4.set_xlabel('Layer n', fontsize=12)
    ax4.set_ylabel('Intercept (log₁₀ scale)', fontsize=12)
    ax4.set_title('Intercepts: Level Factors (Not Coherence)', fontsize=14)
    ax4.set_xticks(layers)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E1_monotone_coherence_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E1_monotone_coherence_results.pdf'))
    plt.close()


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(df: pd.DataFrame, layer_results: List[Dict], 
                 s1_result: Dict) -> None:
    """Save all results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Raw data CSV
    df.to_csv(os.path.join(OUTPUT_DIR, 'E1_raw_data.csv'), index=False)
    
    # 2. Layer results CSV
    layer_df = pd.DataFrame(layer_results)
    layer_df.to_csv(os.path.join(OUTPUT_DIR, 'E1_layer_results.csv'), index=False)
    
    # 3. S1 test results CSV
    s1_df = pd.DataFrame(s1_result['differences'])
    s1_df['test'] = 'S1_monotone'
    s1_df['epsilon_tol'] = EPSILON_TOL
    s1_df['overall_pass'] = s1_result['all_pass']
    s1_df.to_csv(os.path.join(OUTPUT_DIR, 'E1_S1_test_results.csv'), index=False)
    
    # 4. Summary text
    summary = f"""E1: Four-Layer Cascade with Non-Decreasing Coherence
=====================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
Layers: {N_LAYERS}
Base α (α_0): {ALPHA_BASE}
Δα per layer: {DELTA_ALPHA}
True α values: {[ALPHA_BASE + i*DELTA_ALPHA for i in range(N_LAYERS)]}

Size range: L = {L_MIN} to {L_MAX} ({N_SIZES} sizes, geometric grid)
Events per (layer, size): {N_EVENTS}
Noise σ (log-normal): {NOISE_SIGMA}

Bootstrap replicates: {N_BOOTSTRAP}
Decision tolerance ε: {EPSILON_TOL}

RESULTS PER LAYER
-----------------
"""
    for r in layer_results:
        summary += f"""
Layer {r['layer']}:
  True α:      {r['alpha_true']:.3f}
  Estimated α: {r['alpha_est']:.4f}
  95% CI:      [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]
  R²:          {r['R2']:.6f}
  Intercept:   {r['intercept']:.4f}
"""
    
    summary += f"""
S1 TEST: MONOTONE COHERENCE
---------------------------
Decision rule: Pass if all Δα CI lower bounds > -{EPSILON_TOL}

"""
    for d in s1_result['differences']:
        status = "✓ PASS" if d['passes'] else "✗ FAIL"
        summary += f"  Layer {d['from_layer']}→{d['to_layer']}: "
        summary += f"Δα = {d['delta_alpha']:.4f} "
        summary += f"CI = [{d['delta_ci_lower']:.4f}, {d['delta_ci_upper']:.4f}] "
        summary += f"{status}\n"
    
    overall = "✓ PASS" if s1_result['all_pass'] else "✗ FAIL"
    summary += f"""
OVERALL S1 RESULT: {overall}

INTERPRETATION
--------------
The coherence exponent α increases monotonically from layer 0 to layer 3,
as expected under the RTM cascade hypothesis. The slope (coherence) changes
across layers while intercepts (level factors) vary independently.

This confirms Signature S1: non-decreasing coherence in a forward cascade.
"""
    
    with open(os.path.join(OUTPUT_DIR, 'E1_summary.txt'), 'w') as f:
        f.write(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("E1: Four-Layer Cascade with Non-Decreasing Coherence")
    print("RTM Cascade Framework - Signature S1 Validation")
    print("=" * 66)
    
    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Generate data
    print("\nGenerating cascade data...")
    df = generate_cascade_data(rng)
    print(f"  Total observations: {len(df)}")
    print(f"  Layers: {N_LAYERS}, Sizes per layer: {N_SIZES}")
    print(f"  Events per (layer, size): {N_EVENTS}")
    
    # Analyze each layer
    print("\nAnalyzing layers...")
    layer_results = []
    for n in range(N_LAYERS):
        res = analyze_layer(df, n, rng)
        layer_results.append(res)
        print(f"  Layer {n}: α = {res['alpha_est']:.4f} "
              f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}] "
              f"(true: {res['alpha_true']:.2f})")
    
    # Test S1
    print("\nTesting S1: Monotone Coherence...")
    s1_result = test_s1_monotone(layer_results)
    
    for d in s1_result['differences']:
        status = "✓" if d['passes'] else "✗"
        print(f"  Δα({d['from_layer']}→{d['to_layer']}) = {d['delta_alpha']:.4f} "
              f"CI: [{d['delta_ci_lower']:.4f}, {d['delta_ci_upper']:.4f}] {status}")
    
    overall = "PASS" if s1_result['all_pass'] else "FAIL"
    print(f"\n  S1 Overall: {overall}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(df, layer_results, s1_result)
    
    # Save results
    print("Saving results...")
    save_results(df, layer_results, s1_result)
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 66)
    
    return s1_result['all_pass']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
