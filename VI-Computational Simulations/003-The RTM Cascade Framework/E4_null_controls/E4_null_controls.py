#!/usr/bin/env python3
"""
E4: Null Controls (Flat Slopes and Symmetric Causality)
=======================================================

RTM Cascade Framework Validation - Negative Control

This simulation verifies that the S1 and S2 tests correctly return
NULL when there is no cascade structure:
- α is constant across all layers (no coherence increase)
- Coupling is symmetric or absent (no directional causality)

Model:
------
S1 Null: T_n(L) = c_n * L^α * ε  with constant α for all n
         (intercepts c_n vary, but slopes are identical)

S2 Null: Y_n(t) = ε_n(t)  (independent noise, no coupling)
         or symmetric coupling Y_n ↔ Y_m

Expected Outcome:
-----------------
- S1: Adjacent Δα CIs include 0 (no monotone trend)
- S2: TE and Granger symmetric or non-significant

This safeguards against false positives.

Reference: RTM Cascade Framework, Section 4.4
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
ALPHA_CONSTANT = 2.3    # Same α for all layers (no cascade)

# Size configuration (same as E1)
L_MIN = 10
L_MAX = 200
N_SIZES = 10
N_EVENTS = 50

# Noise
NOISE_SIGMA = 0.15

# Time series (same as E2)
N_SAMPLES = 2000
# NO COUPLING (or symmetric)

# Layer-level factors (vary to test intercept vs slope)
LAYER_FACTORS = [1.0, 2.5, 0.8, 3.2]

# Bootstrap and testing
N_BOOTSTRAP = 1000
N_SURROGATES = 500
EPSILON_TOL = 0.05
ALPHA_SIG = 0.05
MAX_LAG = 5

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# NULL MODEL - S1 (Flat Slopes)
# =============================================================================

def generate_null_s1_data(rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate data with CONSTANT α across all layers.
    Intercepts (c_n) vary, but slopes are identical.
    """
    L_values = np.geomspace(L_MIN, L_MAX, N_SIZES)
    records = []
    
    for n in range(N_LAYERS):
        # CONSTANT α for all layers
        alpha_n = ALPHA_CONSTANT
        c_n = LAYER_FACTORS[n]
        
        for L in L_values:
            for _ in range(N_EVENTS):
                epsilon = rng.lognormal(0, NOISE_SIGMA)
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


def test_null_s1(df: pd.DataFrame, rng: np.random.Generator) -> Dict:
    """
    Test S1 on null data: should show NO monotone trend (Δα ≈ 0).
    """
    layer_results = []
    
    for n in range(N_LAYERS):
        layer_data = df[df['layer'] == n]
        log_L = layer_data['log_L'].values
        log_T = layer_data['log_T'].values
        
        # OLS fit
        slope, intercept, r_value, _, _ = stats.linregress(log_L, log_T)
        
        # Bootstrap CI
        slopes = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(log_L), len(log_L), replace=True)
            s, _, _, _, _ = stats.linregress(log_L[idx], log_T[idx])
            slopes.append(s)
        
        alpha_est = np.mean(slopes)
        ci_lower = np.percentile(slopes, 2.5)
        ci_upper = np.percentile(slopes, 97.5)
        
        layer_results.append({
            'layer': n,
            'alpha_true': ALPHA_CONSTANT,
            'alpha_est': alpha_est,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'intercept': intercept,
            'R2': r_value**2
        })
    
    # Test for monotonicity (should FAIL - no trend)
    differences = []
    for i in range(N_LAYERS - 1):
        delta = layer_results[i+1]['alpha_est'] - layer_results[i]['alpha_est']
        ci_w1 = (layer_results[i]['ci_upper'] - layer_results[i]['ci_lower']) / 2
        ci_w2 = (layer_results[i+1]['ci_upper'] - layer_results[i+1]['ci_lower']) / 2
        delta_ci = np.sqrt(ci_w1**2 + ci_w2**2) * 1.96
        
        # For null: CI should INCLUDE 0
        includes_zero = (delta - delta_ci) < 0 and (delta + delta_ci) > 0
        
        differences.append({
            'from_layer': i,
            'to_layer': i + 1,
            'delta_alpha': delta,
            'delta_ci_lower': delta - delta_ci,
            'delta_ci_upper': delta + delta_ci,
            'includes_zero': includes_zero
        })
    
    # S1 NULL: PASS if all differences include 0 (no trend)
    all_include_zero = all(d['includes_zero'] for d in differences)
    
    return {
        'test': 'S1_null_flat_slopes',
        'null_confirmed': all_include_zero,
        'layer_results': layer_results,
        'differences': differences
    }


# =============================================================================
# NULL MODEL - S2 (Symmetric/No Causality)
# =============================================================================

def generate_null_s2_data(rng: np.random.Generator) -> np.ndarray:
    """
    Generate INDEPENDENT time series (no coupling).
    """
    Y = rng.normal(0, 1, (N_LAYERS, N_SAMPLES))
    return Y


def discretize_series(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize for TE."""
    bins = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    return np.digitize(x, bins) - 1


def estimate_te(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
    """Estimate Transfer Entropy."""
    src = discretize_series(source)
    tgt = discretize_series(target)
    
    y_t = tgt[lag:]
    y_past = tgt[:-lag]
    x_past = src[:-lag]
    
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True, axis=0)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    H_yy = entropy(np.column_stack([y_t, y_past]))
    H_y_past = entropy(y_past.reshape(-1, 1))
    H_cond_y = H_yy - H_y_past
    
    H_yyx = entropy(np.column_stack([y_t, y_past, x_past]))
    H_yx_past = entropy(np.column_stack([y_past, x_past]))
    H_cond_yx = H_yyx - H_yx_past
    
    return max(0, H_cond_y - H_cond_yx)


def granger_test(source: np.ndarray, target: np.ndarray, max_lag: int = 5) -> Tuple[float, float]:
    """Granger causality test."""
    n = len(target)
    Y_t = target[max_lag:]
    
    X_r = np.column_stack([target[max_lag-i-1:n-i-1] for i in range(max_lag)])
    X_f = np.column_stack([X_r, *[source[max_lag-i-1:n-i-1].reshape(-1,1) for i in range(max_lag)]])
    
    try:
        beta_r, _, _, _ = np.linalg.lstsq(X_r, Y_t, rcond=None)
        RSS_r = np.sum((Y_t - X_r @ beta_r)**2)
        
        beta_f, _, _, _ = np.linalg.lstsq(X_f, Y_t, rcond=None)
        RSS_f = np.sum((Y_t - X_f @ beta_f)**2)
        
        n_obs = len(Y_t)
        F = ((RSS_r - RSS_f) / max_lag) / (RSS_f / (n_obs - 2*max_lag))
        p = 1 - stats.f.cdf(F, max_lag, n_obs - 2*max_lag)
        return F, p
    except:
        return 0.0, 1.0


def test_null_s2(Y: np.ndarray, rng: np.random.Generator) -> Dict:
    """
    Test S2 on null data: should show SYMMETRIC causality (both directions non-significant).
    """
    results = []
    
    for n in range(1, N_LAYERS):
        # Forward
        te_fwd = estimate_te(Y[n-1], Y[n])
        gc_fwd_f, gc_fwd_p = granger_test(Y[n-1], Y[n], MAX_LAG)
        
        # Reverse
        te_rev = estimate_te(Y[n], Y[n-1])
        gc_rev_f, gc_rev_p = granger_test(Y[n], Y[n-1], MAX_LAG)
        
        # For null: NEITHER should be significant (or both should be similar)
        fwd_sig = gc_fwd_p < ALPHA_SIG
        rev_sig = gc_rev_p < ALPHA_SIG
        
        # Symmetric if both non-significant OR both significant with similar F
        symmetric = (not fwd_sig and not rev_sig) or (fwd_sig and rev_sig and abs(gc_fwd_f - gc_rev_f) < 2)
        
        results.append({
            'from_layer': n-1,
            'to_layer': n,
            'te_forward': te_fwd,
            'te_reverse': te_rev,
            'gc_forward_F': gc_fwd_f,
            'gc_forward_p': gc_fwd_p,
            'gc_reverse_F': gc_rev_f,
            'gc_reverse_p': gc_rev_p,
            'symmetric': symmetric
        })
    
    # S2 NULL: PASS if all pairs are symmetric
    all_symmetric = all(r['symmetric'] for r in results)
    
    return {
        'test': 'S2_null_symmetric',
        'null_confirmed': all_symmetric,
        'results': results
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(df_s1: pd.DataFrame, Y_s2: np.ndarray,
                 s1_result: Dict, s2_result: Dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_LAYERS))
    
    # Plot 1: S1 Null - Log-log scaling (slopes should be equal)
    ax1 = axes[0, 0]
    for n in range(N_LAYERS):
        layer_data = df_s1[df_s1['layer'] == n]
        ax1.scatter(layer_data['log_L'], layer_data['log_T'],
                   alpha=0.3, s=10, color=colors[n], label=f'Layer {n}')
    
    ax1.set_xlabel('log₁₀(L)', fontsize=12)
    ax1.set_ylabel('log₁₀(T)', fontsize=12)
    ax1.set_title('S1 Null: Parallel Lines (Same Slope, Different Intercepts)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: S1 Null - α estimates (should be flat)
    ax2 = axes[0, 1]
    layer_res = s1_result['layer_results']
    layers = [r['layer'] for r in layer_res]
    alphas = [r['alpha_est'] for r in layer_res]
    ci_lo = [r['ci_lower'] for r in layer_res]
    ci_hi = [r['ci_upper'] for r in layer_res]
    
    ax2.errorbar(layers, alphas, yerr=[[a-cl for a,cl in zip(alphas,ci_lo)],
                                        [ch-a for a,ch in zip(alphas,ci_hi)]],
                 fmt='o', markersize=12, capsize=8, color='blue', label='Estimated α')
    ax2.axhline(y=ALPHA_CONSTANT, color='red', linestyle='--', linewidth=2,
                label=f'True α = {ALPHA_CONSTANT}')
    
    ax2.set_xlabel('Layer n', fontsize=12)
    ax2.set_ylabel('Coherence Exponent α', fontsize=12)
    ax2.set_title('S1 Null: Flat α Across Layers (No Cascade)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    status1 = "✓ NULL CONFIRMED" if s1_result['null_confirmed'] else "✗ FALSE POSITIVE"
    ax2.annotate(status1, xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=12, ha='right', fontweight='bold',
                color='green' if s1_result['null_confirmed'] else 'red')
    
    # Plot 3: S2 Null - Time series (independent)
    ax3 = axes[1, 0]
    for n in range(N_LAYERS):
        ax3.plot(Y_s2[n, :300], color=colors[n], alpha=0.8, label=f'Layer {n}')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Y_n(t)', fontsize=12)
    ax3.set_title('S2 Null: Independent Time Series (No Coupling)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: S2 Null - Granger F-stats (should be symmetric/low)
    ax4 = axes[1, 1]
    s2_res = s2_result['results']
    x_pos = np.arange(len(s2_res))
    width = 0.35
    
    gc_fwd = [r['gc_forward_F'] for r in s2_res]
    gc_rev = [r['gc_reverse_F'] for r in s2_res]
    
    ax4.bar(x_pos - width/2, gc_fwd, width, label='Forward', color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, gc_rev, width, label='Reverse', color='red', alpha=0.7)
    
    ax4.set_xlabel('Layer Pair', fontsize=12)
    ax4.set_ylabel('Granger F-statistic', fontsize=12)
    ax4.set_title('S2 Null: Symmetric Granger (No Directional Causality)', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{r["from_layer"]}↔{r["to_layer"]}' for r in s2_res])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    status2 = "✓ NULL CONFIRMED" if s2_result['null_confirmed'] else "✗ FALSE POSITIVE"
    ax4.annotate(status2, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=12, ha='right', fontweight='bold',
                color='green' if s2_result['null_confirmed'] else 'red')
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E4_null_controls_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E4_null_controls_results.pdf'))
    plt.close()


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(df_s1: pd.DataFrame, Y_s2: np.ndarray,
                 s1_result: Dict, s2_result: Dict) -> None:
    """Save all results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. S1 data
    df_s1.to_csv(os.path.join(OUTPUT_DIR, 'E4_S1_null_data.csv'), index=False)
    
    # 2. S1 results
    s1_df = pd.DataFrame(s1_result['layer_results'])
    s1_df['null_confirmed'] = s1_result['null_confirmed']
    s1_df.to_csv(os.path.join(OUTPUT_DIR, 'E4_S1_null_results.csv'), index=False)
    
    # 3. S2 time series
    ts_df = pd.DataFrame({f'Layer_{n}': Y_s2[n] for n in range(N_LAYERS)})
    ts_df.to_csv(os.path.join(OUTPUT_DIR, 'E4_S2_null_timeseries.csv'), index=False)
    
    # 4. S2 results
    s2_df = pd.DataFrame(s2_result['results'])
    s2_df['null_confirmed'] = s2_result['null_confirmed']
    s2_df.to_csv(os.path.join(OUTPUT_DIR, 'E4_S2_null_results.csv'), index=False)
    
    # 5. Summary
    summary = f"""E4: Null Controls (Flat Slopes and Symmetric Causality)
=====================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PURPOSE
-------
Verify that S1 and S2 tests correctly return NULL when there is no cascade.
This protects against false positives.

S1 NULL TEST: FLAT SLOPES
-------------------------
Model: T_n(L) = c_n * L^α * ε with CONSTANT α = {ALPHA_CONSTANT} for all layers.
       Intercepts c_n vary but slopes are identical.

Expected: Δα CIs should include 0 (no monotone trend)
Result: {'NULL CONFIRMED ✓' if s1_result['null_confirmed'] else 'FALSE POSITIVE ✗'}

Layer α estimates:
"""
    for r in s1_result['layer_results']:
        summary += f"  Layer {r['layer']}: α = {r['alpha_est']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]\n"
    
    summary += f"""
Differences:
"""
    for d in s1_result['differences']:
        status = "✓ includes 0" if d['includes_zero'] else "✗ excludes 0"
        summary += f"  Δα({d['from_layer']}→{d['to_layer']}) = {d['delta_alpha']:.4f} "
        summary += f"[{d['delta_ci_lower']:.4f}, {d['delta_ci_upper']:.4f}] {status}\n"
    
    summary += f"""
S2 NULL TEST: SYMMETRIC CAUSALITY
---------------------------------
Model: Y_n(t) = ε_n(t) (independent noise, no coupling)

Expected: Granger F-stats similar in both directions, neither significant
Result: {'NULL CONFIRMED ✓' if s2_result['null_confirmed'] else 'FALSE POSITIVE ✗'}

Causality tests:
"""
    for r in s2_result['results']:
        status = "✓ symmetric" if r['symmetric'] else "✗ asymmetric"
        summary += f"  {r['from_layer']}↔{r['to_layer']}: "
        summary += f"GC_fwd F={r['gc_forward_F']:.2f} (p={r['gc_forward_p']:.4f}) "
        summary += f"GC_rev F={r['gc_reverse_F']:.2f} (p={r['gc_reverse_p']:.4f}) {status}\n"
    
    summary += f"""
OVERALL NULL CONTROL STATUS
---------------------------
S1 Null: {'CONFIRMED' if s1_result['null_confirmed'] else 'FAILED'}
S2 Null: {'CONFIRMED' if s2_result['null_confirmed'] else 'FAILED'}

INTERPRETATION
--------------
Both null controls should be CONFIRMED to validate that the S1/S2 tests
do not produce false positives. If they fail, the methodology may be
detecting spurious patterns.
"""
    
    with open(os.path.join(OUTPUT_DIR, 'E4_summary.txt'), 'w') as f:
        f.write(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("E4: Null Controls (Flat Slopes and Symmetric Causality)")
    print("RTM Cascade Framework - Negative Control Validation")
    print("=" * 66)
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    # S1 Null Test
    print("\n--- S1 NULL TEST: Flat Slopes ---")
    print(f"  Constant α = {ALPHA_CONSTANT} for all layers")
    df_s1 = generate_null_s1_data(rng)
    s1_result = test_null_s1(df_s1, rng)
    
    for r in s1_result['layer_results']:
        print(f"  Layer {r['layer']}: α = {r['alpha_est']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    
    print(f"\n  Differences (should include 0):")
    for d in s1_result['differences']:
        status = "✓" if d['includes_zero'] else "✗"
        print(f"    Δα({d['from_layer']}→{d['to_layer']}) = {d['delta_alpha']:.4f} {status}")
    
    print(f"\n  S1 Null: {'CONFIRMED' if s1_result['null_confirmed'] else 'FAILED'}")
    
    # S2 Null Test
    print("\n--- S2 NULL TEST: Symmetric Causality ---")
    print("  Independent time series (no coupling)")
    Y_s2 = generate_null_s2_data(rng)
    s2_result = test_null_s2(Y_s2, rng)
    
    for r in s2_result['results']:
        status = "✓" if r['symmetric'] else "✗"
        print(f"  {r['from_layer']}↔{r['to_layer']}: "
              f"GC_fwd={r['gc_forward_F']:.2f} GC_rev={r['gc_reverse_F']:.2f} {status}")
    
    print(f"\n  S2 Null: {'CONFIRMED' if s2_result['null_confirmed'] else 'FAILED'}")
    
    # Create plots and save
    print("\nCreating plots...")
    create_plots(df_s1, Y_s2, s1_result, s2_result)
    
    print("Saving results...")
    save_results(df_s1, Y_s2, s1_result, s2_result)
    
    overall = s1_result['null_confirmed'] and s2_result['null_confirmed']
    print(f"\n{'=' * 66}")
    print(f"OVERALL NULL CONTROL: {'CONFIRMED ✓' if overall else 'FAILED ✗'}")
    print(f"{'=' * 66}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    
    return overall


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
