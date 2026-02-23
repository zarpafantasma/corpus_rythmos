#!/usr/bin/env python3
"""
T2: Slope-at-r Estimator for RTM Tests
======================================

From "Black Holes in the RTM Framework"

Estimates the RTM coherence exponent α from log-log regression of
observed process times τ versus effective sizes L at fixed radius.

Key equation: log(τ_obs) = α × log(L) + intercept + noise

The slope equals α(r), independent of GR redshift Z(r).
This is the core empirical handle for testing RTM activation.

Reference: Paper Sections 4.1-4.2, 8.1
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# PROFILE FUNCTIONS (from T1)
# =============================================================================

def alpha_logistic(r, alpha_0=1.0, alpha_inf=3.0, r_t=6.0, w=2.0):
    """Logistic profile for α(r)."""
    return alpha_inf + (alpha_0 - alpha_inf) / (1 + np.exp(-(r - r_t) / w))

def Z_schwarzschild(r, r_s=2.0):
    """Gravitational redshift factor."""
    r = np.asarray(r)
    r_safe = np.maximum(r, r_s * 1.001)
    return 1.0 / np.sqrt(1 - r_s / r_safe)

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_observations(r: float, L_values: np.ndarray, n_events: int = 50,
                          noise_sigma: float = 0.15, rng: np.random.Generator = None,
                          alpha_func=alpha_logistic, L_0: float = 1.0, T_0: float = 1.0):
    """
    Generate synthetic observations at radius r.
    
    τ_obs = Z(r) × (L/L_0)^α(r) × T_0 × ε
    
    Returns arrays of (L, τ) pairs with lognormal noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    alpha_r = alpha_func(r)
    Z_r = Z_schwarzschild(r)
    
    L_all = []
    tau_all = []
    
    for L in L_values:
        for _ in range(n_events):
            epsilon = rng.lognormal(0, noise_sigma)
            tau = Z_r * (L / L_0) ** alpha_r * T_0 * epsilon
            L_all.append(L)
            tau_all.append(tau)
    
    return np.array(L_all), np.array(tau_all), alpha_r, Z_r

# =============================================================================
# SLOPE ESTIMATION
# =============================================================================

def estimate_slope_ols(log_L: np.ndarray, log_tau: np.ndarray):
    """OLS regression to estimate slope α and intercept."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_tau)
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err
    }

def bootstrap_slope(log_L: np.ndarray, log_tau: np.ndarray, n_bootstrap: int = 1000,
                    rng: np.random.Generator = None):
    """Bootstrap estimation of slope with confidence intervals."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(log_L)
    slopes = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        slope, _, _, _, _ = stats.linregress(log_L[idx], log_tau[idx])
        slopes.append(slope)
    
    slopes = np.array(slopes)
    return {
        'mean': np.mean(slopes),
        'std': np.std(slopes),
        'ci_low': np.percentile(slopes, 2.5),
        'ci_high': np.percentile(slopes, 97.5),
        'slopes': slopes
    }

# =============================================================================
# TWO-RADIUS TEST
# =============================================================================

def two_radius_test(r_inner: float, r_outer: float, L_values: np.ndarray,
                    n_events: int = 50, noise_sigma: float = 0.15,
                    n_bootstrap: int = 1000, rng: np.random.Generator = None):
    """
    Test for slope difference between two radii.
    
    Δα = α(r_inner) - α(r_outer)
    
    Falsification: if 95% CI for Δα includes 0, no activation detected.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate data at both radii
    L_in, tau_in, alpha_in_true, Z_in = generate_observations(
        r_inner, L_values, n_events, noise_sigma, rng)
    L_out, tau_out, alpha_out_true, Z_out = generate_observations(
        r_outer, L_values, n_events, noise_sigma, rng)
    
    log_L_in, log_tau_in = np.log10(L_in), np.log10(tau_in)
    log_L_out, log_tau_out = np.log10(L_out), np.log10(tau_out)
    
    # Bootstrap slope differences
    n_in, n_out = len(L_in), len(L_out)
    delta_alphas = []
    
    for _ in range(n_bootstrap):
        idx_in = rng.choice(n_in, size=n_in, replace=True)
        idx_out = rng.choice(n_out, size=n_out, replace=True)
        
        slope_in, _, _, _, _ = stats.linregress(log_L_in[idx_in], log_tau_in[idx_in])
        slope_out, _, _, _, _ = stats.linregress(log_L_out[idx_out], log_tau_out[idx_out])
        
        delta_alphas.append(slope_in - slope_out)
    
    delta_alphas = np.array(delta_alphas)
    
    # OLS estimates
    ols_in = estimate_slope_ols(log_L_in, log_tau_in)
    ols_out = estimate_slope_ols(log_L_out, log_tau_out)
    
    return {
        'r_inner': r_inner,
        'r_outer': r_outer,
        'alpha_inner_true': alpha_in_true,
        'alpha_outer_true': alpha_out_true,
        'alpha_inner_est': ols_in['slope'],
        'alpha_outer_est': ols_out['slope'],
        'delta_alpha_true': alpha_in_true - alpha_out_true,
        'delta_alpha_est': np.mean(delta_alphas),
        'delta_alpha_ci_low': np.percentile(delta_alphas, 2.5),
        'delta_alpha_ci_high': np.percentile(delta_alphas, 97.5),
        'ci_excludes_zero': not (np.percentile(delta_alphas, 2.5) <= 0 <= np.percentile(delta_alphas, 97.5)),
        'significant': not (np.percentile(delta_alphas, 2.5) <= 0 <= np.percentile(delta_alphas, 97.5))
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create slope estimation visualizations."""
    
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    L_values = np.geomspace(0.1, 10, 10)
    
    # Plot 1: Log-log at multiple radii
    ax1 = axes[0, 0]
    colors = ['blue', 'orange', 'red']
    radii = [15, 8, 4]
    
    for r, color in zip(radii, colors):
        L, tau, alpha_true, Z = generate_observations(r, L_values, n_events=30, 
                                                       noise_sigma=0.15, rng=rng)
        log_L, log_tau = np.log10(L), np.log10(tau)
        
        # Fit
        result = estimate_slope_ols(log_L, log_tau)
        
        ax1.scatter(log_L, log_tau, alpha=0.3, s=15, color=color)
        L_fit = np.linspace(log_L.min(), log_L.max(), 100)
        tau_fit = result['slope'] * L_fit + result['intercept']
        ax1.plot(L_fit, tau_fit, color=color, linewidth=2, 
                 label=f'r={r}: α_true={alpha_true:.2f}, α_est={result["slope"]:.2f}')
    
    ax1.set_xlabel('log₁₀(L)', fontsize=12)
    ax1.set_ylabel('log₁₀(τ_obs)', fontsize=12)
    ax1.set_title('Slope Estimation at Different Radii', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovered α(r) profile
    ax2 = axes[0, 1]
    r_bins = np.array([4, 5, 6, 7, 8, 10, 12, 15, 18])
    alpha_est = []
    alpha_ci_low = []
    alpha_ci_high = []
    alpha_true_arr = []
    
    for r in r_bins:
        L, tau, alpha_true, _ = generate_observations(r, L_values, n_events=50, 
                                                       noise_sigma=0.15, rng=rng)
        log_L, log_tau = np.log10(L), np.log10(tau)
        bs = bootstrap_slope(log_L, log_tau, n_bootstrap=500, rng=rng)
        
        alpha_est.append(bs['mean'])
        alpha_ci_low.append(bs['ci_low'])
        alpha_ci_high.append(bs['ci_high'])
        alpha_true_arr.append(alpha_true)
    
    alpha_est = np.array(alpha_est)
    alpha_ci_low = np.array(alpha_ci_low)
    alpha_ci_high = np.array(alpha_ci_high)
    
    ax2.errorbar(r_bins, alpha_est, yerr=[alpha_est - alpha_ci_low, alpha_ci_high - alpha_est],
                 fmt='o', capsize=5, capthick=2, color='blue', label='Estimated α')
    
    r_fine = np.linspace(3, 20, 100)
    ax2.plot(r_fine, alpha_logistic(r_fine), 'r-', linewidth=2, label='True α(r)')
    
    ax2.set_xlabel('Radius r/r_s', fontsize=12)
    ax2.set_ylabel('Coherence exponent α', fontsize=12)
    ax2.set_title('Recovered α(r) Profile with 95% CI', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Two-radius test
    ax3 = axes[1, 0]
    test = two_radius_test(4, 12, L_values, n_events=50, noise_sigma=0.15,
                           n_bootstrap=1000, rng=rng)
    
    ax3.hist(np.random.default_rng(42).normal(test['delta_alpha_est'], 
                                               (test['delta_alpha_ci_high'] - test['delta_alpha_ci_low'])/4,
                                               1000),
             bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Null (Δα = 0)')
    ax3.axvline(x=test['delta_alpha_true'], color='green', linestyle='-', linewidth=2,
                label=f'True Δα = {test["delta_alpha_true"]:.2f}')
    ax3.axvline(x=test['delta_alpha_ci_low'], color='orange', linestyle=':', linewidth=2)
    ax3.axvline(x=test['delta_alpha_ci_high'], color='orange', linestyle=':', linewidth=2,
                label='95% CI')
    
    ax3.set_xlabel('Δα = α(r_inner) - α(r_outer)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Two-Radius Test: r={test["r_inner"]} vs r={test["r_outer"]}', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    status = "SIGNIFICANT" if test['significant'] else "NOT SIGNIFICANT"
    ax3.text(0.05, 0.95, f'Result: {status}', transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Sensitivity vs noise level
    ax4 = axes[1, 1]
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    ci_widths = []
    
    for noise in noise_levels:
        L, tau, _, _ = generate_observations(6, L_values, n_events=50, noise_sigma=noise, rng=rng)
        log_L, log_tau = np.log10(L), np.log10(tau)
        bs = bootstrap_slope(log_L, log_tau, n_bootstrap=500, rng=rng)
        ci_widths.append(bs['ci_high'] - bs['ci_low'])
    
    ax4.plot(noise_levels, ci_widths, 'bo-', markersize=8, linewidth=2)
    ax4.set_xlabel('Noise level σ', fontsize=12)
    ax4.set_ylabel('95% CI width for α', fontsize=12)
    ax4.set_title('Estimation Precision vs Noise', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'T2_slope_estimator.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'T2_slope_estimator.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("T2: Slope-at-r Estimator for RTM Tests")
    print("From: Black Holes in the RTM Framework")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    L_values = np.geomspace(0.1, 10, 10)
    
    # Multi-radius estimation
    print("\nSLOPE ESTIMATION AT MULTIPLE RADII")
    print("-" * 50)
    
    records = []
    r_bins = [4, 5, 6, 7, 8, 10, 12, 15, 18]
    
    for r in r_bins:
        L, tau, alpha_true, Z = generate_observations(r, L_values, n_events=50, 
                                                       noise_sigma=0.15, rng=rng)
        log_L, log_tau = np.log10(L), np.log10(tau)
        
        ols = estimate_slope_ols(log_L, log_tau)
        bs = bootstrap_slope(log_L, log_tau, n_bootstrap=1000, rng=rng)
        
        records.append({
            'r': r,
            'alpha_true': alpha_true,
            'alpha_est': bs['mean'],
            'alpha_ci_low': bs['ci_low'],
            'alpha_ci_high': bs['ci_high'],
            'intercept': ols['intercept'],
            'r_squared': ols['r_squared'],
            'Z_r': Z
        })
        
        print(f"  r = {r:2d}:  α_true = {alpha_true:.3f},  "
              f"α_est = {bs['mean']:.3f} [{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]")
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'T2_slope_estimates.csv'), index=False)
    
    # Two-radius test
    print("\n" + "=" * 66)
    print("TWO-RADIUS FALSIFICATION TEST")
    print("=" * 66)
    
    test = two_radius_test(4, 12, L_values, n_events=50, noise_sigma=0.15,
                           n_bootstrap=1000, rng=rng)
    
    print(f"\n  r_inner = {test['r_inner']}, r_outer = {test['r_outer']}")
    print(f"  α_inner (true) = {test['alpha_inner_true']:.3f}")
    print(f"  α_outer (true) = {test['alpha_outer_true']:.3f}")
    print(f"  Δα (true)      = {test['delta_alpha_true']:.3f}")
    print(f"  Δα (estimated) = {test['delta_alpha_est']:.3f} "
          f"[{test['delta_alpha_ci_low']:.3f}, {test['delta_alpha_ci_high']:.3f}]")
    print(f"\n  CI excludes 0: {test['ci_excludes_zero']}")
    print(f"  Activation detected: {'YES ✓' if test['significant'] else 'NO'}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Summary
    summary = f"""T2: Slope-at-r Estimator
========================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METHOD
------
1. Generate observations τ_obs = Z(r) × L^α(r) × ε
2. Fit log(τ) vs log(L) via OLS
3. Slope = α(r), independent of Z(r)
4. Bootstrap for 95% confidence intervals

TWO-RADIUS FALSIFICATION TEST
-----------------------------
Compare α(r_inner) vs α(r_outer)
Δα = α_inner - α_outer

Decision rule:
  - 95% CI for Δα excludes 0 → Activation detected
  - 95% CI includes 0 → No activation (null)

TEST RESULT (r=4 vs r=12)
-------------------------
Δα = {test['delta_alpha_est']:.3f} [{test['delta_alpha_ci_low']:.3f}, {test['delta_alpha_ci_high']:.3f}]
Activation: {'DETECTED ✓' if test['significant'] else 'NOT DETECTED'}

KEY INSIGHT
-----------
Slope in log(τ) vs log(L) equals α(r).
GR/kinematics only affect the intercept.
Radial evolution of slope → RTM activation.
"""
    
    with open(os.path.join(output_dir, 'T2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
