#!/usr/bin/env python3
"""
S1: Gauge Coupling RGE Running
==============================

From "RTM Unified Field Framework" - Section 3.5

Implements two-loop Renormalization Group Equations for SM gauge couplings
g₁, g₂, g₃ and shows their running from M_Z to high scales.

Key Equations:
    dg_i/d(ln μ) = β_i^(1-loop) + β_i^(2-loop)
    
    One-loop: β_i = b_i × g_i³ / (16π²)
    Two-loop: adds b_ij × g_i³ g_j² / (16π²)²

Reference: Paper Section 3.5.1-3.5.2
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

# SM gauge couplings at M_Z (PDG values)
M_Z = 91.1876  # GeV
ALPHA_1_MZ = 0.01699  # U(1)_Y
ALPHA_2_MZ = 0.03378  # SU(2)_L
ALPHA_3_MZ = 0.1179   # SU(3)_C

# Convert to g_i (g² = 4π α)
G1_MZ = np.sqrt(4 * np.pi * ALPHA_1_MZ)  # ~0.462
G2_MZ = np.sqrt(4 * np.pi * ALPHA_2_MZ)  # ~0.651
G3_MZ = np.sqrt(4 * np.pi * ALPHA_3_MZ)  # ~1.217

# One-loop beta coefficients (SM)
# b_i for SU(3)×SU(2)×U(1) with normalization g₁ = √(5/3) g'
B1_SM = 41/10   # U(1)
B2_SM = -19/6   # SU(2)
B3_SM = -7      # SU(3)

# Two-loop beta coefficients (Machacek-Vaughn)
B_2LOOP = np.array([
    [199/50, 27/10, 44/5],   # g₁
    [9/10, 35/6, 12],        # g₂
    [11/10, 9/2, -26]        # g₃
])


# =============================================================================
# RGE FUNCTIONS
# =============================================================================

def beta_one_loop(g, b):
    """One-loop beta function: β = b × g³ / (16π²)"""
    return b * g**3 / (16 * np.pi**2)


def beta_two_loop(g_vec, b_vec, B_mat):
    """
    Two-loop beta function.
    
    β_i = (b_i g_i³ + Σ_j B_ij g_i³ g_j²) / (16π²)²
    """
    betas = np.zeros(3)
    for i in range(3):
        one_loop = b_vec[i] * g_vec[i]**3 / (16 * np.pi**2)
        two_loop = 0
        for j in range(3):
            two_loop += B_mat[i, j] * g_vec[i]**3 * g_vec[j]**2
        two_loop /= (16 * np.pi**2)**2
        betas[i] = one_loop + two_loop
    return betas


def rge_sm(g_vec, t):
    """RGE system for SM (t = ln(μ/M_Z))"""
    b_vec = np.array([B1_SM, B2_SM, B3_SM])
    return beta_two_loop(g_vec, b_vec, B_2LOOP)


def run_rge(g0, t_span, n_points=500):
    """
    Integrate RGEs from g0 over t_span.
    
    Parameters:
    -----------
    g0 : array
        Initial couplings [g1, g2, g3] at t=0
    t_span : tuple
        (t_min, t_max) where t = ln(μ/M_Z)
    
    Returns:
    --------
    t : array
        log scale values
    g : array
        Couplings at each scale (n_points × 3)
    mu : array
        Energy scale in GeV
    """
    t = np.linspace(t_span[0], t_span[1], n_points)
    g = odeint(rge_sm, g0, t)
    mu = M_Z * np.exp(t)
    return t, g, mu


def coupling_to_alpha_inv(g):
    """Convert coupling g to α⁻¹ = 4π/g²"""
    return 4 * np.pi / g**2


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(t, g, mu, output_dir):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert to α⁻¹
    alpha_inv = np.zeros_like(g)
    for i in range(3):
        alpha_inv[:, i] = coupling_to_alpha_inv(g[:, i])
    
    # Plot 1: α⁻¹ vs log₁₀(μ)
    ax1 = axes[0]
    log_mu = np.log10(mu)
    
    ax1.plot(log_mu, alpha_inv[:, 0], 'b-', linewidth=2, label='α₁⁻¹ (U(1))')
    ax1.plot(log_mu, alpha_inv[:, 1], 'g-', linewidth=2, label='α₂⁻¹ (SU(2))')
    ax1.plot(log_mu, alpha_inv[:, 2], 'r-', linewidth=2, label='α₃⁻¹ (SU(3))')
    
    ax1.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax1.set_ylabel('α⁻¹', fontsize=12)
    ax1.set_title('SM Gauge Coupling Running (Two-Loop)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 17)
    ax1.set_ylim(0, 70)
    
    # Mark key scales
    ax1.axvline(x=np.log10(M_Z), color='gray', linestyle='--', alpha=0.5)
    ax1.text(np.log10(M_Z) + 0.2, 65, 'M_Z', fontsize=10)
    
    # Plot 2: Coupling spread
    ax2 = axes[1]
    
    # Spread = max - min of α⁻¹
    spread = np.max(alpha_inv, axis=1) - np.min(alpha_inv, axis=1)
    
    ax2.plot(log_mu, spread, 'purple', linewidth=2)
    ax2.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax2.set_ylabel('α⁻¹ spread (max - min)', fontsize=12)
    ax2.set_title('Coupling Spread vs Scale', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Find minimum spread
    min_idx = np.argmin(spread)
    ax2.axvline(x=log_mu[min_idx], color='red', linestyle='--', alpha=0.7)
    ax2.text(log_mu[min_idx] + 0.5, spread[min_idx] + 5, 
             f'Min at μ ≈ 10^{log_mu[min_idx]:.1f} GeV', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_gauge_rge_running.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_gauge_rge_running.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S1: Gauge Coupling RGE Running")
    print("From: RTM Unified Field Framework - Section 3.5")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("INITIAL CONDITIONS (PDG at M_Z)")
    print("=" * 70)
    print(f"""
    M_Z = {M_Z} GeV
    
    α₁(M_Z) = {ALPHA_1_MZ} → g₁ = {G1_MZ:.4f}
    α₂(M_Z) = {ALPHA_2_MZ} → g₂ = {G2_MZ:.4f}
    α₃(M_Z) = {ALPHA_3_MZ} → g₃ = {G3_MZ:.4f}
    """)
    
    print("=" * 70)
    print("BETA FUNCTION COEFFICIENTS (SM)")
    print("=" * 70)
    print(f"""
    One-loop:
      b₁ = {B1_SM:.2f}
      b₂ = {B2_SM:.2f}
      b₃ = {B3_SM:.2f}
    
    Two-loop matrix B_ij: see Machacek-Vaughn
    """)
    
    # Run RGE
    print("=" * 70)
    print("RUNNING RGEs FROM M_Z TO 10^17 GeV")
    print("=" * 70)
    
    g0 = np.array([G1_MZ, G2_MZ, G3_MZ])
    t_span = (0, np.log(1e17 / M_Z))  # M_Z to 10^17 GeV
    
    print("\nIntegrating two-loop RGEs...")
    t, g, mu = run_rge(g0, t_span, n_points=500)
    print("Done!")
    
    # Results at key scales
    print("\n" + "=" * 70)
    print("RESULTS AT KEY SCALES")
    print("=" * 70)
    
    scales = [1e3, 1e6, 1e10, 1e14, 1e16]
    print(f"\n{'Scale (GeV)':<15} | {'α₁⁻¹':<10} | {'α₂⁻¹':<10} | {'α₃⁻¹':<10} | {'Spread':<10}")
    print("-" * 65)
    
    for scale in scales:
        idx = np.argmin(np.abs(mu - scale))
        alpha_inv = coupling_to_alpha_inv(g[idx])
        spread = np.max(alpha_inv) - np.min(alpha_inv)
        print(f"{scale:<15.0e} | {alpha_inv[0]:<10.2f} | {alpha_inv[1]:<10.2f} | "
              f"{alpha_inv[2]:<10.2f} | {spread:<10.2f}")
    
    # Find approximate unification
    alpha_inv_all = np.array([coupling_to_alpha_inv(g[i]) for i in range(len(g))])
    spread = np.max(alpha_inv_all, axis=1) - np.min(alpha_inv_all, axis=1)
    min_idx = np.argmin(spread)
    
    print(f"\nMinimum spread at μ ≈ {mu[min_idx]:.2e} GeV")
    print(f"  α₁⁻¹ = {alpha_inv_all[min_idx, 0]:.2f}")
    print(f"  α₂⁻¹ = {alpha_inv_all[min_idx, 1]:.2f}")
    print(f"  α₃⁻¹ = {alpha_inv_all[min_idx, 2]:.2f}")
    print(f"  Spread = {spread[min_idx]:.2f}")
    
    print("\n" + "=" * 70)
    print("SM DOES NOT UNIFY")
    print("=" * 70)
    print("""
    The Standard Model couplings do NOT unify at any scale.
    The minimum spread (~6) occurs around 10^14 GeV.
    
    This is the motivation for RTM threshold corrections
    (Section 3.5.2) which can achieve unification.
    """)
    
    # Save data
    df = pd.DataFrame({
        'log10_mu': np.log10(mu),
        'mu_GeV': mu,
        'g1': g[:, 0],
        'g2': g[:, 1],
        'g3': g[:, 2],
        'alpha1_inv': alpha_inv_all[:, 0],
        'alpha2_inv': alpha_inv_all[:, 1],
        'alpha3_inv': alpha_inv_all[:, 2],
        'spread': spread
    })
    df.to_csv(os.path.join(output_dir, 'S1_rge_running.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(t, g, mu, output_dir)
    
    # Summary
    summary = f"""S1: Gauge Coupling RGE Running
==============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INITIAL CONDITIONS (M_Z)
------------------------
α₁(M_Z) = {ALPHA_1_MZ}
α₂(M_Z) = {ALPHA_2_MZ}
α₃(M_Z) = {ALPHA_3_MZ}

BETA COEFFICIENTS (SM)
----------------------
b₁ = {B1_SM:.2f}
b₂ = {B2_SM:.2f}
b₃ = {B3_SM:.2f}

RESULTS
-------
Minimum spread at μ ≈ {mu[min_idx]:.2e} GeV
  α₁⁻¹ = {alpha_inv_all[min_idx, 0]:.2f}
  α₂⁻¹ = {alpha_inv_all[min_idx, 1]:.2f}
  α₃⁻¹ = {alpha_inv_all[min_idx, 2]:.2f}
  Spread = {spread[min_idx]:.2f}

CONCLUSION
----------
SM gauge couplings do NOT unify.
RTM threshold corrections are needed (Section 3.5.2).
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
