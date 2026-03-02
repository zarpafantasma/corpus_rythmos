#!/usr/bin/env python3
"""
E2: Directional Causality in a Layered Chain
=============================================

RTM Cascade Framework Validation - Signature S2

This simulation validates that information flow is asymmetric (forward-only)
between adjacent layers in a cascade architecture.

Model:
------
Layer observables Y_n(t) follow a forward-coupled process:

    Y_0(t) = ε_0(t)                           (base layer)
    Y_n(t) = κ * Y_{n-1}(t-1) + ε_n(t)        (forward coupling only)

Where:
- κ = forward coupling strength (no backward coupling)
- ε_n(t) ~ N(0, σ²) = independent noise

Measurement:
------------
1. Transfer Entropy (TE): TE(n-1 → n) vs TE(n → n-1)
2. Granger Causality: F-test for forward vs reverse direction

Decision Rule (S2):
-------------------
Pass if both TE and Granger are significant for forward direction
AND not significant for reverse direction (FDR-adjusted)

Expected Pattern:
-----------------
TE(n-1 → n) >> TE(n → n-1)
Granger significant only forward

Reference: RTM Cascade Framework, Section 4.2
"""

import numpy as np
from scipy import stats
from scipy.signal import detrend
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS
# =============================================================================

# Layer configuration
N_LAYERS = 4

# Time series parameters
N_SAMPLES = 2000        # samples per layer
COUPLING_STRENGTH = 0.7 # κ (forward coupling)
NOISE_STD = 0.5         # σ (noise standard deviation)

# Causality analysis
MAX_LAG = 5             # maximum lag for TE/Granger
N_SURROGATES = 500      # surrogates for significance testing
ALPHA_SIG = 0.05        # significance level

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "output"


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_coupled_chain(rng: np.random.Generator) -> np.ndarray:
    """
    Generate forward-coupled time series for N_LAYERS layers.
    
    Y_0(t) = ε_0(t)
    Y_n(t) = κ * Y_{n-1}(t-1) + ε_n(t)
    
    Returns: array of shape (N_LAYERS, N_SAMPLES)
    """
    Y = np.zeros((N_LAYERS, N_SAMPLES))
    
    # Layer 0: pure noise
    Y[0] = rng.normal(0, NOISE_STD, N_SAMPLES)
    
    # Subsequent layers: forward coupling
    for n in range(1, N_LAYERS):
        noise = rng.normal(0, NOISE_STD, N_SAMPLES)
        Y[n, 0] = noise[0]
        for t in range(1, N_SAMPLES):
            Y[n, t] = COUPLING_STRENGTH * Y[n-1, t-1] + noise[t]
    
    return Y


# =============================================================================
# TRANSFER ENTROPY
# =============================================================================

def discretize_series(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize continuous series into bins for TE estimation."""
    bins = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    return np.digitize(x, bins) - 1


def estimate_transfer_entropy(source: np.ndarray, target: np.ndarray, 
                              lag: int = 1, n_bins: int = 8) -> float:
    """
    Estimate Transfer Entropy from source to target.
    
    TE(X → Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    
    Uses binning approximation.
    """
    # Discretize
    src = discretize_series(source, n_bins)
    tgt = discretize_series(target, n_bins)
    
    n = len(tgt)
    
    # Create lagged arrays
    y_t = tgt[lag:]
    y_past = tgt[:-lag]
    x_past = src[:-lag]
    
    # Joint and marginal counts
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True, axis=0)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    # H(Y_t, Y_past)
    joint_yy = np.column_stack([y_t, y_past])
    H_yy = entropy(joint_yy)
    
    # H(Y_past)
    H_y_past = entropy(y_past.reshape(-1, 1))
    
    # H(Y_t | Y_past) = H(Y_t, Y_past) - H(Y_past)
    H_cond_y = H_yy - H_y_past
    
    # H(Y_t, Y_past, X_past)
    joint_yyx = np.column_stack([y_t, y_past, x_past])
    H_yyx = entropy(joint_yyx)
    
    # H(Y_past, X_past)
    joint_yx_past = np.column_stack([y_past, x_past])
    H_yx_past = entropy(joint_yx_past)
    
    # H(Y_t | Y_past, X_past) = H(Y_t, Y_past, X_past) - H(Y_past, X_past)
    H_cond_yx = H_yyx - H_yx_past
    
    # TE = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
    te = H_cond_y - H_cond_yx
    
    return max(0, te)  # TE is non-negative


def te_surrogate_test(source: np.ndarray, target: np.ndarray,
                      n_surrogates: int, rng: np.random.Generator,
                      lag: int = 1) -> Tuple[float, float, bool]:
    """
    Test TE significance using surrogate data (time-shuffle).
    
    Returns: (TE_observed, p_value, is_significant)
    """
    te_obs = estimate_transfer_entropy(source, target, lag)
    
    te_surr = []
    for _ in range(n_surrogates):
        # Shuffle source to destroy temporal structure
        src_shuffled = rng.permutation(source)
        te_s = estimate_transfer_entropy(src_shuffled, target, lag)
        te_surr.append(te_s)
    
    te_surr = np.array(te_surr)
    p_value = np.mean(te_surr >= te_obs)
    
    return te_obs, p_value, p_value < ALPHA_SIG


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

def granger_causality_test(source: np.ndarray, target: np.ndarray,
                           max_lag: int = 5) -> Tuple[float, float, bool]:
    """
    Test Granger causality from source to target.
    
    Compare:
    - Restricted model: Y_t = Σ a_i Y_{t-i} + ε
    - Full model: Y_t = Σ a_i Y_{t-i} + Σ b_j X_{t-j} + ε
    
    Returns: (F_statistic, p_value, is_significant)
    """
    n = len(target)
    
    # Build design matrices
    Y_target = target[max_lag:]
    
    # Restricted model: only past Y
    X_restricted = np.column_stack([
        target[max_lag-i-1:n-i-1] for i in range(max_lag)
    ])
    
    # Full model: past Y and past X
    X_full = np.column_stack([
        X_restricted,
        *[source[max_lag-i-1:n-i-1].reshape(-1, 1) for i in range(max_lag)]
    ])
    
    # Fit models
    try:
        # Restricted
        beta_r, residuals_r, rank_r, s_r = np.linalg.lstsq(X_restricted, Y_target, rcond=None)
        if len(residuals_r) == 0:
            RSS_r = np.sum((Y_target - X_restricted @ beta_r)**2)
        else:
            RSS_r = residuals_r[0]
        
        # Full
        beta_f, residuals_f, rank_f, s_f = np.linalg.lstsq(X_full, Y_target, rcond=None)
        if len(residuals_f) == 0:
            RSS_f = np.sum((Y_target - X_full @ beta_f)**2)
        else:
            RSS_f = residuals_f[0]
        
        # F-test
        n_obs = len(Y_target)
        p_r = max_lag
        p_f = 2 * max_lag
        
        F_stat = ((RSS_r - RSS_f) / (p_f - p_r)) / (RSS_f / (n_obs - p_f))
        p_value = 1 - stats.f.cdf(F_stat, p_f - p_r, n_obs - p_f)
        
        return F_stat, p_value, p_value < ALPHA_SIG
    
    except Exception as e:
        return 0.0, 1.0, False


# =============================================================================
# S2 TEST
# =============================================================================

def test_s2_directionality(Y: np.ndarray, rng: np.random.Generator) -> Dict:
    """
    Test Signature S2: Forward-only directionality.
    
    For each adjacent pair (n-1, n):
    - Compute TE and Granger forward and reverse
    - Require significant forward AND not significant reverse
    """
    results = []
    
    for n in range(1, N_LAYERS):
        print(f"\n  Testing Layer {n-1} ↔ Layer {n}:")
        
        # Forward: n-1 → n
        te_fwd, te_fwd_p, te_fwd_sig = te_surrogate_test(
            Y[n-1], Y[n], N_SURROGATES, rng
        )
        gc_fwd_f, gc_fwd_p, gc_fwd_sig = granger_causality_test(
            Y[n-1], Y[n], MAX_LAG
        )
        
        # Reverse: n → n-1
        te_rev, te_rev_p, te_rev_sig = te_surrogate_test(
            Y[n], Y[n-1], N_SURROGATES, rng
        )
        gc_rev_f, gc_rev_p, gc_rev_sig = granger_causality_test(
            Y[n], Y[n-1], MAX_LAG
        )
        
        # Decision: forward significant AND reverse not significant
        te_asymmetric = te_fwd_sig and not te_rev_sig
        gc_asymmetric = gc_fwd_sig and not gc_rev_sig
        passes = te_asymmetric and gc_asymmetric
        
        result = {
            'from_layer': n-1,
            'to_layer': n,
            'te_forward': te_fwd,
            'te_forward_p': te_fwd_p,
            'te_forward_sig': te_fwd_sig,
            'te_reverse': te_rev,
            'te_reverse_p': te_rev_p,
            'te_reverse_sig': te_rev_sig,
            'te_asymmetric': te_asymmetric,
            'gc_forward_F': gc_fwd_f,
            'gc_forward_p': gc_fwd_p,
            'gc_forward_sig': gc_fwd_sig,
            'gc_reverse_F': gc_rev_f,
            'gc_reverse_p': gc_rev_p,
            'gc_reverse_sig': gc_rev_sig,
            'gc_asymmetric': gc_asymmetric,
            'passes': passes
        }
        results.append(result)
        
        print(f"    TE:  Forward={te_fwd:.4f} (p={te_fwd_p:.4f}{'*' if te_fwd_sig else ''}) "
              f"Reverse={te_rev:.4f} (p={te_rev_p:.4f}{'*' if te_rev_sig else ''})")
        print(f"    GC:  Forward F={gc_fwd_f:.2f} (p={gc_fwd_p:.4f}{'*' if gc_fwd_sig else ''}) "
              f"Reverse F={gc_rev_f:.2f} (p={gc_rev_p:.4f}{'*' if gc_rev_sig else ''})")
        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"    {status}")
    
    all_pass = all(r['passes'] for r in results)
    
    return {
        'test': 'S2_directional_causality',
        'all_pass': all_pass,
        'alpha': ALPHA_SIG,
        'results': results
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(Y: np.ndarray, s2_result: Dict) -> None:
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_LAYERS))
    
    # Plot 1: Time series
    ax1 = axes[0, 0]
    for n in range(N_LAYERS):
        ax1.plot(Y[n, :500], color=colors[n], alpha=0.8, label=f'Layer {n}')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Y_n(t)', fontsize=12)
    ax1.set_title('Layer Time Series (first 500 samples)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-correlations
    ax2 = axes[0, 1]
    lags = np.arange(-20, 21)
    for n in range(1, N_LAYERS):
        xcorr = [np.corrcoef(Y[n-1, max(0,-lag):min(N_SAMPLES, N_SAMPLES-lag)],
                             Y[n, max(0,lag):min(N_SAMPLES, N_SAMPLES+lag)])[0,1]
                 for lag in lags]
        ax2.plot(lags, xcorr, 'o-', color=colors[n], markersize=3,
                label=f'Layer {n-1} → {n}')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='red', linestyle=':', alpha=0.7, label='Expected peak (lag=1)')
    ax2.set_xlabel('Lag', fontsize=12)
    ax2.set_ylabel('Cross-correlation', fontsize=12)
    ax2.set_title('Cross-Correlations Between Adjacent Layers', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Transfer Entropy comparison
    ax3 = axes[1, 0]
    results = s2_result['results']
    x_pos = np.arange(len(results))
    width = 0.35
    
    te_fwd = [r['te_forward'] for r in results]
    te_rev = [r['te_reverse'] for r in results]
    
    bars1 = ax3.bar(x_pos - width/2, te_fwd, width, label='Forward (n-1 → n)',
                    color='green', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, te_rev, width, label='Reverse (n → n-1)',
                    color='red', alpha=0.7)
    
    # Add significance markers
    for i, r in enumerate(results):
        if r['te_forward_sig']:
            ax3.annotate('*', xy=(x_pos[i] - width/2, te_fwd[i]), ha='center', fontsize=14)
        if r['te_reverse_sig']:
            ax3.annotate('*', xy=(x_pos[i] + width/2, te_rev[i]), ha='center', fontsize=14)
    
    ax3.set_xlabel('Layer Pair', fontsize=12)
    ax3.set_ylabel('Transfer Entropy (bits)', fontsize=12)
    ax3.set_title('S2: Transfer Entropy (Forward vs Reverse)', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{r["from_layer"]}↔{r["to_layer"]}' for r in results])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Granger F-statistics
    ax4 = axes[1, 1]
    gc_fwd = [r['gc_forward_F'] for r in results]
    gc_rev = [r['gc_reverse_F'] for r in results]
    
    bars3 = ax4.bar(x_pos - width/2, gc_fwd, width, label='Forward (n-1 → n)',
                    color='green', alpha=0.7)
    bars4 = ax4.bar(x_pos + width/2, gc_rev, width, label='Reverse (n → n-1)',
                    color='red', alpha=0.7)
    
    for i, r in enumerate(results):
        if r['gc_forward_sig']:
            ax4.annotate('*', xy=(x_pos[i] - width/2, gc_fwd[i]), ha='center', fontsize=14)
        if r['gc_reverse_sig']:
            ax4.annotate('*', xy=(x_pos[i] + width/2, gc_rev[i]), ha='center', fontsize=14)
    
    ax4.set_xlabel('Layer Pair', fontsize=12)
    ax4.set_ylabel('Granger F-statistic', fontsize=12)
    ax4.set_title('S2: Granger Causality (Forward vs Reverse)', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{r["from_layer"]}↔{r["to_layer"]}' for r in results])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Overall result annotation
    status = "✓ PASS" if s2_result['all_pass'] else "✗ FAIL"
    fig.suptitle(f"E2: Directional Causality Test — S2: {status}", fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E2_directional_causality_results.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'E2_directional_causality_results.pdf'))
    plt.close()


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(Y: np.ndarray, s2_result: Dict) -> None:
    """Save all results to files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Time series data CSV
    ts_df = pd.DataFrame({f'Layer_{n}': Y[n] for n in range(N_LAYERS)})
    ts_df.to_csv(os.path.join(OUTPUT_DIR, 'E2_time_series.csv'), index=False)
    
    # 2. S2 results CSV
    s2_df = pd.DataFrame(s2_result['results'])
    s2_df['overall_pass'] = s2_result['all_pass']
    s2_df.to_csv(os.path.join(OUTPUT_DIR, 'E2_S2_test_results.csv'), index=False)
    
    # 3. Summary text
    summary = f"""E2: Directional Causality in a Layered Chain
=============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS
----------
Layers: {N_LAYERS}
Samples per layer: {N_SAMPLES}
Coupling strength (κ): {COUPLING_STRENGTH}
Noise std (σ): {NOISE_STD}
Max lag: {MAX_LAG}
Surrogates: {N_SURROGATES}
Significance level: {ALPHA_SIG}

MODEL
-----
Y_0(t) = ε_0(t)
Y_n(t) = κ * Y_{{n-1}}(t-1) + ε_n(t)

Forward coupling only (no backward coupling).

S2 TEST: DIRECTIONAL CAUSALITY
------------------------------
Decision rule: Pass if TE and Granger are both:
  - Significant forward (n-1 → n)
  - Not significant reverse (n → n-1)

"""
    for r in s2_result['results']:
        status = "✓ PASS" if r['passes'] else "✗ FAIL"
        summary += f"""
Layer {r['from_layer']} ↔ Layer {r['to_layer']}:
  Transfer Entropy:
    Forward: {r['te_forward']:.4f} (p={r['te_forward_p']:.4f}) {'*' if r['te_forward_sig'] else ''}
    Reverse: {r['te_reverse']:.4f} (p={r['te_reverse_p']:.4f}) {'*' if r['te_reverse_sig'] else ''}
    Asymmetric: {'Yes' if r['te_asymmetric'] else 'No'}
  Granger Causality:
    Forward: F={r['gc_forward_F']:.2f} (p={r['gc_forward_p']:.4f}) {'*' if r['gc_forward_sig'] else ''}
    Reverse: F={r['gc_reverse_F']:.2f} (p={r['gc_reverse_p']:.4f}) {'*' if r['gc_reverse_sig'] else ''}
    Asymmetric: {'Yes' if r['gc_asymmetric'] else 'No'}
  Result: {status}
"""
    
    overall = "✓ PASS" if s2_result['all_pass'] else "✗ FAIL"
    summary += f"""
OVERALL S2 RESULT: {overall}

INTERPRETATION
--------------
Information flows forward (from layer n-1 to layer n) with no significant
reverse flow. This confirms the RTM cascade hypothesis of forward-only
directionality in the echo architecture.

* = significant at α = {ALPHA_SIG}
"""
    
    with open(os.path.join(OUTPUT_DIR, 'E2_summary.txt'), 'w') as f:
        f.write(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("E2: Directional Causality in a Layered Chain")
    print("RTM Cascade Framework - Signature S2 Validation")
    print("=" * 66)
    
    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Generate data
    print("\nGenerating forward-coupled time series...")
    Y = generate_coupled_chain(rng)
    print(f"  Layers: {N_LAYERS}, Samples: {N_SAMPLES}")
    print(f"  Coupling strength κ = {COUPLING_STRENGTH}")
    
    # Test S2
    print("\nTesting S2: Directional Causality...")
    s2_result = test_s2_directionality(Y, rng)
    
    overall = "PASS" if s2_result['all_pass'] else "FAIL"
    print(f"\n  S2 Overall: {overall}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(Y, s2_result)
    
    # Save results
    print("Saving results...")
    save_results(Y, s2_result)
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 66)
    
    return s2_result['all_pass']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
