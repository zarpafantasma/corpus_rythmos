#!/usr/bin/env python3
"""
T3: GR vs RTM Decomposition
===========================

From "Black Holes in the RTM Framework"

Demonstrates the key decomposition principle:
- RTM affects SLOPES (how τ scales with L)
- GR/kinematics affect LEVELS (intercepts)

This separation allows clean testing: if slopes change with radius,
RTM is active; if only intercepts change, it's pure GR.

Key equation:
  log(τ_obs) = α(r) × log(L) + [log(Z(r)) + log(T_0/L_0^α)]
              \_slope_/          \___intercept (level)___/

Reference: Paper Sections 2.1, 4.4, Demo C
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def alpha_logistic(r, alpha_0=1.0, alpha_inf=3.0, r_t=6.0, w=2.0):
    """Logistic profile for α(r)."""
    return alpha_inf + (alpha_0 - alpha_inf) / (1 + np.exp(-(r - r_t) / w))

def alpha_constant(r, alpha_const=2.0):
    """Constant α (null case for RTM)."""
    return np.full_like(np.atleast_1d(r), alpha_const, dtype=float).squeeze()

def Z_schwarzschild(r, r_s=2.0):
    """Gravitational redshift factor."""
    r = np.asarray(r)
    r_safe = np.maximum(r, r_s * 1.001)
    return 1.0 / np.sqrt(1 - r_s / r_safe)

def generate_data(r_values, L_values, n_events=50, noise_sigma=0.15,
                  alpha_func=alpha_logistic, rng=None):
    """Generate observations at multiple radii."""
    if rng is None:
        rng = np.random.default_rng()
    
    records = []
    
    for r in r_values:
        alpha_r = alpha_func(r)
        Z_r = Z_schwarzschild(r)
        
        for L in L_values:
            for _ in range(n_events):
                epsilon = rng.lognormal(0, noise_sigma)
                tau = Z_r * (L ** alpha_r) * epsilon
                records.append({
                    'r': r,
                    'L': L,
                    'tau': tau,
                    'log_L': np.log10(L),
                    'log_tau': np.log10(tau),
                    'alpha_true': alpha_r,
                    'Z_true': Z_r
                })
    
    return pd.DataFrame(records)

def fit_per_radius(df):
    """Fit slope and intercept at each radius."""
    results = []
    
    for r in df['r'].unique():
        subset = df[df['r'] == r]
        slope, intercept, r_val, _, _ = stats.linregress(subset['log_L'], subset['log_tau'])
        
        results.append({
            'r': r,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_val ** 2,
            'alpha_true': subset['alpha_true'].iloc[0],
            'Z_true': subset['Z_true'].iloc[0],
            'log_Z': np.log10(subset['Z_true'].iloc[0])
        })
    
    return pd.DataFrame(results)

# =============================================================================
# SCENARIOS
# =============================================================================

def scenario_rtm_active(r_values, L_values, n_events=50, noise_sigma=0.15, rng=None):
    """
    Scenario A: RTM active (α varies with r)
    Expected: slopes change, intercepts change
    """
    return generate_data(r_values, L_values, n_events, noise_sigma,
                         alpha_func=alpha_logistic, rng=rng)

def scenario_rtm_inactive(r_values, L_values, n_events=50, noise_sigma=0.15, rng=None):
    """
    Scenario B: RTM inactive (α constant)
    Expected: slopes constant, intercepts change (due to Z(r))
    """
    return generate_data(r_values, L_values, n_events, noise_sigma,
                         alpha_func=alpha_constant, rng=rng)

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create decomposition visualizations."""
    
    rng = np.random.default_rng(42)
    r_values = np.array([4, 6, 8, 10, 12, 15])
    L_values = np.geomspace(0.1, 10, 8)
    
    # Generate both scenarios
    df_active = scenario_rtm_active(r_values, L_values, n_events=30, rng=rng)
    df_inactive = scenario_rtm_inactive(r_values, L_values, n_events=30, rng=rng)
    
    # Fit
    fits_active = fit_per_radius(df_active)
    fits_inactive = fit_per_radius(df_inactive)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: RTM Active
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(r_values)))
    
    for i, r in enumerate(r_values):
        subset = df_active[df_active['r'] == r]
        ax1.scatter(subset['log_L'], subset['log_tau'], alpha=0.3, s=15, color=colors[i])
        
        fit = fits_active[fits_active['r'] == r].iloc[0]
        x = np.linspace(subset['log_L'].min(), subset['log_L'].max(), 50)
        y = fit['slope'] * x + fit['intercept']
        ax1.plot(x, y, color=colors[i], linewidth=2, label=f'r={r}: α={fit["slope"]:.2f}')
    
    ax1.set_xlabel('log₁₀(L)', fontsize=12)
    ax1.set_ylabel('log₁₀(τ_obs)', fontsize=12)
    ax1.set_title('RTM ACTIVE: Slopes Vary with Radius', fontsize=14)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Slopes vs radius (active)
    ax2 = axes[0, 1]
    ax2.plot(fits_active['r'], fits_active['slope'], 'bo-', markersize=10, linewidth=2, label='Estimated')
    ax2.plot(fits_active['r'], fits_active['alpha_true'], 'r--', linewidth=2, label='True α(r)')
    ax2.set_xlabel('Radius r/r_s', fontsize=12)
    ax2.set_ylabel('Slope (= α)', fontsize=12)
    ax2.set_title('SLOPES Change → RTM Active', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Intercepts vs log(Z) (active)
    ax3 = axes[0, 2]
    ax3.plot(fits_active['log_Z'], fits_active['intercept'], 'go-', markersize=10, linewidth=2)
    ax3.set_xlabel('log₁₀(Z)', fontsize=12)
    ax3.set_ylabel('Intercept', fontsize=12)
    ax3.set_title('Intercepts Scale with Z(r)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Row 2: RTM Inactive (null)
    ax4 = axes[1, 0]
    
    for i, r in enumerate(r_values):
        subset = df_inactive[df_inactive['r'] == r]
        ax4.scatter(subset['log_L'], subset['log_tau'], alpha=0.3, s=15, color=colors[i])
        
        fit = fits_inactive[fits_inactive['r'] == r].iloc[0]
        x = np.linspace(subset['log_L'].min(), subset['log_L'].max(), 50)
        y = fit['slope'] * x + fit['intercept']
        ax4.plot(x, y, color=colors[i], linewidth=2, label=f'r={r}: α={fit["slope"]:.2f}')
    
    ax4.set_xlabel('log₁₀(L)', fontsize=12)
    ax4.set_ylabel('log₁₀(τ_obs)', fontsize=12)
    ax4.set_title('RTM INACTIVE: Slopes Constant (Parallel Lines)', fontsize=14)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Slopes vs radius (inactive)
    ax5 = axes[1, 1]
    ax5.plot(fits_inactive['r'], fits_inactive['slope'], 'bo-', markersize=10, linewidth=2, label='Estimated')
    ax5.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='True α = 2.0')
    ax5.set_xlabel('Radius r/r_s', fontsize=12)
    ax5.set_ylabel('Slope (= α)', fontsize=12)
    ax5.set_title('SLOPES Constant → No RTM Activation', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(1.5, 2.5)
    
    # Intercepts vs log(Z) (inactive)
    ax6 = axes[1, 2]
    ax6.plot(fits_inactive['log_Z'], fits_inactive['intercept'], 'go-', markersize=10, linewidth=2)
    
    # Fit intercept vs log(Z)
    slope_int, intercept_int, _, _, _ = stats.linregress(fits_inactive['log_Z'], fits_inactive['intercept'])
    x_fit = np.linspace(fits_inactive['log_Z'].min(), fits_inactive['log_Z'].max(), 50)
    y_fit = slope_int * x_fit + intercept_int
    ax6.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: slope = {slope_int:.2f}')
    
    ax6.set_xlabel('log₁₀(Z)', fontsize=12)
    ax6.set_ylabel('Intercept', fontsize=12)
    ax6.set_title('Intercepts ∝ log(Z) (GR Only)', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'T3_decomposition.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'T3_decomposition.pdf'))
    plt.close()
    
    return fits_active, fits_inactive

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("T3: GR vs RTM Decomposition")
    print("From: Black Holes in the RTM Framework")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    r_values = np.array([4, 6, 8, 10, 12, 15])
    L_values = np.geomspace(0.1, 10, 8)
    
    # Generate and analyze both scenarios
    print("\n" + "=" * 66)
    print("SCENARIO A: RTM ACTIVE (α varies with r)")
    print("=" * 66)
    
    df_active = scenario_rtm_active(r_values, L_values, n_events=50, rng=rng)
    fits_active = fit_per_radius(df_active)
    
    print(f"\n{'r':>4} | {'α_true':>8} | {'α_est':>8} | {'Intercept':>10} | {'log(Z)':>8}")
    print("-" * 50)
    for _, row in fits_active.iterrows():
        print(f"{row['r']:>4.0f} | {row['alpha_true']:>8.3f} | {row['slope']:>8.3f} | "
              f"{row['intercept']:>10.3f} | {row['log_Z']:>8.3f}")
    
    print(f"\n  Slope variation: {fits_active['slope'].max() - fits_active['slope'].min():.3f}")
    print("  → SLOPES CHANGE with radius ✓")
    
    print("\n" + "=" * 66)
    print("SCENARIO B: RTM INACTIVE (α constant = 2.0)")
    print("=" * 66)
    
    df_inactive = scenario_rtm_inactive(r_values, L_values, n_events=50, rng=rng)
    fits_inactive = fit_per_radius(df_inactive)
    
    print(f"\n{'r':>4} | {'α_true':>8} | {'α_est':>8} | {'Intercept':>10} | {'log(Z)':>8}")
    print("-" * 50)
    for _, row in fits_inactive.iterrows():
        print(f"{row['r']:>4.0f} | {row['alpha_true']:>8.3f} | {row['slope']:>8.3f} | "
              f"{row['intercept']:>10.3f} | {row['log_Z']:>8.3f}")
    
    print(f"\n  Slope variation: {fits_inactive['slope'].max() - fits_inactive['slope'].min():.3f}")
    print("  → SLOPES CONSTANT (only intercepts change) ✓")
    
    # Save data
    fits_active.to_csv(os.path.join(output_dir, 'T3_rtm_active.csv'), index=False)
    fits_inactive.to_csv(os.path.join(output_dir, 'T3_rtm_inactive.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Key decomposition
    print("\n" + "=" * 66)
    print("DECOMPOSITION PRINCIPLE")
    print("=" * 66)
    print("""
    log(τ_obs) = α(r) × log(L) + [log(Z(r)) + const]
                 \_____/          \_________________/
                  SLOPE              INTERCEPT
    
    RTM affects:  SLOPE (α changes with r)
    GR affects:   INTERCEPT (Z changes with r)
    
    Test logic:
      • Slopes vary with radius → RTM is active
      • Only intercepts vary   → Pure GR (no RTM)
    """)
    
    # Summary
    summary = f"""T3: GR vs RTM Decomposition
============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DECOMPOSITION EQUATION
----------------------
log(τ_obs) = α(r) × log(L) + [log(Z(r)) + const]
             \\_____/          \\_________________/
              SLOPE              INTERCEPT

KEY PRINCIPLE
-------------
- RTM affects SLOPES (coherence exponent α)
- GR affects INTERCEPTS (redshift factor Z)

SCENARIO A: RTM ACTIVE
----------------------
Slopes vary: {fits_active['slope'].min():.3f} to {fits_active['slope'].max():.3f}
→ Radial evolution of α detected

SCENARIO B: RTM INACTIVE
------------------------
Slopes constant: ~{fits_inactive['slope'].mean():.3f} ± {fits_inactive['slope'].std():.3f}
→ No RTM activation (pure GR)

OBSERVATIONAL STRATEGY
----------------------
1. Bin by radius
2. Fit log(τ) vs log(L) in each bin
3. If slopes change → RTM is active
4. If only intercepts change → Pure GR
"""
    
    with open(os.path.join(output_dir, 'T3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
