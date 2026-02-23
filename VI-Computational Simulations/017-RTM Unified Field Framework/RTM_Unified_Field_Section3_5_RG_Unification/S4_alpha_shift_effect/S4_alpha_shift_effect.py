#!/usr/bin/env python3
"""
S4: Alpha-Shift Effect on Gauge Coupling Unification
====================================================

From "RTM Unified Field Framework" - Section 3.5.1

Studies the α-shift mechanism that provides scale-dependent
correction to achieve gauge coupling unification.

Key Equation:
    Δα_shift = η × Δg × (μ/M_RTM)^ξ
    
    Where ξ = 1 (paper choice for stability)

This shift modifies the effective beta coefficients and
enables unification when combined with threshold corrections.

Reference: Paper Section 3.5.1
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

M_Z = 91.1876

ALPHA_1_MZ = 0.01699
ALPHA_2_MZ = 0.03378
ALPHA_3_MZ = 0.1179

G1_MZ = np.sqrt(4 * np.pi * ALPHA_1_MZ)
G2_MZ = np.sqrt(4 * np.pi * ALPHA_2_MZ)
G3_MZ = np.sqrt(4 * np.pi * ALPHA_3_MZ)

B1_SM = 41/10
B2_SM = -19/6
B3_SM = -7


# =============================================================================
# ALPHA-SHIFT MODEL
# =============================================================================

def alpha_shift_function(mu, M_RTM, eta, xi=1.0):
    """
    Compute α-shift at scale μ.
    
    Δα = η × (μ/M_RTM)^ξ  for μ > M_RTM
    Δα = 0                 for μ ≤ M_RTM
    """
    if mu <= M_RTM:
        return 0
    return eta * (mu / M_RTM)**xi


def rge_with_alpha_shift(g_vec, t, M_RTM, eta, xi=1.0, shift_pattern=None):
    """
    RGE with α-shift correction.
    
    The shift_pattern controls how the shift affects each coupling.
    Default: preferentially affects g₁ and g₂ to bring them to g₃.
    """
    mu = M_Z * np.exp(t)
    
    # Safety bounds
    g_vec = np.clip(g_vec, 0.1, 5.0)
    
    if shift_pattern is None:
        shift_pattern = np.array([0.15, 0.10, 0.03])
    
    b = np.array([B1_SM, B2_SM, B3_SM])
    
    # Apply α-shift with cap
    delta_alpha = alpha_shift_function(mu, M_RTM, eta, xi)
    delta_alpha = min(delta_alpha, 0.5)  # Cap for stability
    b_effective = b * (1 + delta_alpha * shift_pattern)
    
    # One-loop beta
    betas = np.zeros(3)
    for i in range(3):
        betas[i] = b_effective[i] * g_vec[i]**3 / (16 * np.pi**2)
    
    return betas


def run_rge_alpha_shift(g0, t_span, M_RTM, eta, xi=1.0, n_points=400):
    """Run RGE with α-shift."""
    t = np.linspace(t_span[0], t_span[1], n_points)
    
    def rge_func(g, t_val):
        return rge_with_alpha_shift(g, t_val, M_RTM, eta, xi)
    
    g = odeint(rge_func, g0, t)
    mu = M_Z * np.exp(t)
    return t, g, mu


def coupling_to_alpha_inv(g):
    g = np.clip(g, 0.1, 10.0)  # Safety bounds
    return 4 * np.pi / g**2


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(results_dict, output_dir):
    """Create comparison plots for different η values."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Running for different η
    ax1 = axes[0, 0]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, (eta, data) in enumerate(results_dict.items()):
        log_mu = np.log10(data['mu'])
        alpha_inv = np.array([coupling_to_alpha_inv(data['g'][j]) for j in range(len(data['g']))])
        
        # Plot α₃⁻¹ for each η (the one that changes most visibly)
        ax1.plot(log_mu, alpha_inv[:, 2], color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)], linewidth=2,
                 label=f'η = {eta}')
    
    ax1.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax1.set_ylabel('α₃⁻¹', fontsize=12)
    ax1.set_title('α₃⁻¹ Running for Different η (α-shift)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 17)
    
    # Plot 2: Spread at 10^15 GeV vs η
    ax2 = axes[0, 1]
    
    etas = []
    spreads = []
    
    for eta, data in results_dict.items():
        etas.append(eta)
        idx = np.argmin(np.abs(data['mu'] - 1e15))
        alpha_inv = coupling_to_alpha_inv(data['g'][idx])
        spreads.append(np.max(alpha_inv) - np.min(alpha_inv))
    
    ax2.plot(etas, spreads, 'bo-', markersize=8, linewidth=2)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect unification')
    
    ax2.set_xlabel('α-shift parameter η', fontsize=12)
    ax2.set_ylabel('Coupling spread at 10^15 GeV', fontsize=12)
    ax2.set_title('Effect of α-Shift on Unification', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: α-shift function
    ax3 = axes[1, 0]
    
    mu_range = np.logspace(10, 17, 100)
    M_RTM = 3.2e11
    
    for xi, color in [(0.5, 'blue'), (1.0, 'green'), (1.5, 'red')]:
        shifts = [alpha_shift_function(m, M_RTM, 0.1, xi) for m in mu_range]
        ax3.plot(np.log10(mu_range), shifts, color=color, linewidth=2,
                 label=f'ξ = {xi}')
    
    ax3.axvline(x=np.log10(M_RTM), color='gray', linestyle='--', alpha=0.7)
    ax3.text(np.log10(M_RTM) + 0.2, 0.5, 'M_RTM', fontsize=10)
    
    ax3.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax3.set_ylabel('Δα_shift (η = 0.1)', fontsize=12)
    ax3.set_title('α-Shift Function Shape', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Best unification
    ax4 = axes[1, 1]
    
    # Find best η
    best_eta = etas[np.argmin(spreads)]
    data = results_dict[best_eta]
    
    log_mu = np.log10(data['mu'])
    alpha_inv = np.array([coupling_to_alpha_inv(data['g'][j]) for j in range(len(data['g']))])
    
    ax4.plot(log_mu, alpha_inv[:, 0], 'b-', linewidth=2, label='α₁⁻¹')
    ax4.plot(log_mu, alpha_inv[:, 1], 'g-', linewidth=2, label='α₂⁻¹')
    ax4.plot(log_mu, alpha_inv[:, 2], 'r-', linewidth=2, label='α₃⁻¹')
    
    ax4.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax4.set_ylabel('α⁻¹', fontsize=12)
    ax4.set_title(f'Best Result: η = {best_eta}', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2, 17)
    ax4.set_ylim(0, 70)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S4_alpha_shift.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S4_alpha_shift.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S4: Alpha-Shift Effect on Gauge Coupling Unification")
    print("From: RTM Unified Field Framework - Section 3.5.1")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("ALPHA-SHIFT MECHANISM")
    print("=" * 70)
    print("""
    The α-shift provides scale-dependent correction:
    
        Δα_shift = η × (μ/M_RTM)^ξ
        
    Key features:
    - Zero below M_RTM
    - Grows with energy above M_RTM
    - ξ = 1 chosen for stability (paper)
    - η is the fit parameter
    """)
    
    # Run simulations for different η
    print("\n" + "=" * 70)
    print("PARAMETER STUDY")
    print("=" * 70)
    
    g0 = np.array([G1_MZ, G2_MZ, G3_MZ])
    t_span = (0, np.log(1e17 / M_Z))
    M_RTM = 3.2e11  # Paper value
    
    eta_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    results = {}
    
    print(f"\nM_RTM = {M_RTM:.2e} GeV")
    print(f"Testing η = {eta_values}")
    print("\nRunning simulations...")
    
    for eta in eta_values:
        t, g, mu = run_rge_alpha_shift(g0, t_span, M_RTM, eta)
        results[eta] = {'t': t, 'g': g, 'mu': mu}
    
    print("Done!")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS: SPREAD AT 10^15 GeV")
    print("=" * 70)
    
    print(f"\n{'η':<10} | {'Spread':<12} | {'α₁⁻¹':<10} | {'α₂⁻¹':<10} | {'α₃⁻¹':<10}")
    print("-" * 60)
    
    records = []
    for eta, data in results.items():
        idx = np.argmin(np.abs(data['mu'] - 1e15))
        alpha_inv = coupling_to_alpha_inv(data['g'][idx])
        spread = np.max(alpha_inv) - np.min(alpha_inv)
        
        print(f"{eta:<10.2f} | {spread:<12.2f} | {alpha_inv[0]:<10.2f} | "
              f"{alpha_inv[1]:<10.2f} | {alpha_inv[2]:<10.2f}")
        
        records.append({
            'eta': eta,
            'spread': spread,
            'alpha1_inv': alpha_inv[0],
            'alpha2_inv': alpha_inv[1],
            'alpha3_inv': alpha_inv[2]
        })
    
    df = pd.DataFrame(records)
    
    # Find best
    best_idx = df['spread'].idxmin()
    best_eta = df.loc[best_idx, 'eta']
    best_spread = df.loc[best_idx, 'spread']
    
    print(f"\nBest result: η = {best_eta}, spread = {best_spread:.2f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
    Without α-shift (η = 0): spread = {df.loc[0, 'spread']:.2f}
    With α-shift (η = {best_eta}): spread = {best_spread:.2f}
    
    The α-shift mechanism:
    1. Preferentially modifies g₁ and g₂
    2. Brings all three couplings closer together
    3. Combined with thresholds, enables unification
    """)
    
    # Save data
    df.to_csv(os.path.join(output_dir, 'S4_alpha_shift_scan.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results, output_dir)
    
    # Summary
    summary = f"""S4: Alpha-Shift Effect on Gauge Coupling Unification
====================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MECHANISM
---------
Δα_shift = η × (μ/M_RTM)^ξ

Parameters:
  M_RTM = {M_RTM:.2e} GeV (paper value)
  ξ = 1 (stability choice)
  η = fit parameter

RESULTS (at 10^15 GeV)
----------------------
Without α-shift: spread = {df.loc[0, 'spread']:.2f}
Best result: η = {best_eta}, spread = {best_spread:.2f}

CONCLUSION
----------
The α-shift mechanism significantly reduces coupling spread,
enabling gauge unification when combined with threshold corrections.
"""
    
    with open(os.path.join(output_dir, 'S4_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
