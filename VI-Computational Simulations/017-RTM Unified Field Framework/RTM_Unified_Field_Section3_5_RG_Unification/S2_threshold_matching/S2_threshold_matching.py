#!/usr/bin/env python3
"""
S2: Threshold Matching with RTM States
======================================

From "RTM Unified Field Framework" - Section 3.5.2

Implements exact one-loop threshold corrections at each RTM state's mass.

Key Concept:
    New states (scalars, fermions) modify beta coefficients above their mass.
    
    Effective b_i = b_i^SM + Σ_states Δb_i × θ(μ - M_state)

RTM Threshold Catalogue (from paper):
    - RTM scalar φ at M_φ
    - Heavy vector-like fermions
    - Additional Higgs-like scalars

Reference: Paper Section 3.5.2 "Threshold Catalogue and Matching"
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

M_Z = 91.1876  # GeV

# SM couplings at M_Z
ALPHA_1_MZ = 0.01699
ALPHA_2_MZ = 0.03378
ALPHA_3_MZ = 0.1179

G1_MZ = np.sqrt(4 * np.pi * ALPHA_1_MZ)
G2_MZ = np.sqrt(4 * np.pi * ALPHA_2_MZ)
G3_MZ = np.sqrt(4 * np.pi * ALPHA_3_MZ)

# SM one-loop coefficients
B1_SM = 41/10
B2_SM = -19/6
B3_SM = -7


# =============================================================================
# THRESHOLD CATALOGUE (from paper Table in Section 3.5.2)
# =============================================================================

# Format: (name, mass_GeV, Δb1, Δb2, Δb3)
RTM_THRESHOLDS = [
    ('RTM scalar φ', 3.2e11, 1/10, 1/6, 0),        # Gauge singlet + small mixing
    ('Heavy fermion F1', 1.0e12, 2/5, 0, 1/3),     # Vector-like quark
    ('Heavy fermion F2', 5.0e12, 1/5, 1/2, 0),     # Vector-like lepton
    ('Heavy scalar S', 1.0e13, 1/10, 1/6, 0),      # Extra Higgs doublet
    ('RTM vector V', 5.0e13, 0, 4/3, 2),           # Heavy gauge bosons
]


# =============================================================================
# RGE WITH THRESHOLDS
# =============================================================================

def get_effective_b(mu, include_thresholds=True):
    """
    Get effective beta coefficients at scale μ.
    
    Adds Δb_i for each state with M_state < μ.
    """
    b = np.array([B1_SM, B2_SM, B3_SM])
    
    if include_thresholds:
        for name, mass, db1, db2, db3 in RTM_THRESHOLDS:
            if mu > mass:
                b += np.array([db1, db2, db3])
    
    return b


def rge_with_thresholds(g_vec, t, include_thresholds=True):
    """RGE system with threshold matching."""
    mu = M_Z * np.exp(t)
    b = get_effective_b(mu, include_thresholds)
    
    # One-loop only for speed
    betas = np.zeros(3)
    for i in range(3):
        betas[i] = b[i] * g_vec[i]**3 / (16 * np.pi**2)
    
    return betas


def run_rge(g0, t_span, include_thresholds=True, n_points=500):
    """Integrate RGEs."""
    t = np.linspace(t_span[0], t_span[1], n_points)
    
    def rge_func(g, t_val):
        return rge_with_thresholds(g, t_val, include_thresholds)
    
    g = odeint(rge_func, g0, t)
    mu = M_Z * np.exp(t)
    return t, g, mu


def coupling_to_alpha_inv(g):
    """Convert g to α⁻¹."""
    return 4 * np.pi / g**2


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(results_sm, results_rtm, output_dir):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    t_sm, g_sm, mu_sm = results_sm
    t_rtm, g_rtm, mu_rtm = results_rtm
    
    log_mu_sm = np.log10(mu_sm)
    log_mu_rtm = np.log10(mu_rtm)
    
    # Convert to α⁻¹
    alpha_sm = np.array([coupling_to_alpha_inv(g_sm[i]) for i in range(len(g_sm))])
    alpha_rtm = np.array([coupling_to_alpha_inv(g_rtm[i]) for i in range(len(g_rtm))])
    
    # Plot 1: SM running
    ax1 = axes[0]
    ax1.plot(log_mu_sm, alpha_sm[:, 0], 'b-', linewidth=2, label='α₁⁻¹')
    ax1.plot(log_mu_sm, alpha_sm[:, 1], 'g-', linewidth=2, label='α₂⁻¹')
    ax1.plot(log_mu_sm, alpha_sm[:, 2], 'r-', linewidth=2, label='α₃⁻¹')
    
    ax1.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax1.set_ylabel('α⁻¹', fontsize=12)
    ax1.set_title('SM Only (No Thresholds)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 17)
    ax1.set_ylim(0, 70)
    
    # Plot 2: SM + RTM thresholds
    ax2 = axes[1]
    ax2.plot(log_mu_rtm, alpha_rtm[:, 0], 'b-', linewidth=2, label='α₁⁻¹')
    ax2.plot(log_mu_rtm, alpha_rtm[:, 1], 'g-', linewidth=2, label='α₂⁻¹')
    ax2.plot(log_mu_rtm, alpha_rtm[:, 2], 'r-', linewidth=2, label='α₃⁻¹')
    
    # Mark thresholds
    for name, mass, _, _, _ in RTM_THRESHOLDS:
        ax2.axvline(x=np.log10(mass), color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('log₁₀(μ/GeV)', fontsize=12)
    ax2.set_ylabel('α⁻¹', fontsize=12)
    ax2.set_title('SM + RTM Thresholds', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 17)
    ax2.set_ylim(0, 70)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_threshold_matching.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_threshold_matching.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S2: Threshold Matching with RTM States")
    print("From: RTM Unified Field Framework - Section 3.5.2")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RTM THRESHOLD CATALOGUE")
    print("=" * 70)
    print(f"\n{'State':<20} | {'Mass (GeV)':<12} | {'Δb₁':<8} | {'Δb₂':<8} | {'Δb₃':<8}")
    print("-" * 70)
    
    for name, mass, db1, db2, db3 in RTM_THRESHOLDS:
        print(f"{name:<20} | {mass:<12.2e} | {db1:<8.2f} | {db2:<8.2f} | {db3:<8.2f}")
    
    # Run RGEs
    print("\n" + "=" * 70)
    print("RUNNING RGEs")
    print("=" * 70)
    
    g0 = np.array([G1_MZ, G2_MZ, G3_MZ])
    t_span = (0, np.log(1e17 / M_Z))
    
    print("\n1. SM only...")
    results_sm = run_rge(g0, t_span, include_thresholds=False)
    
    print("2. SM + RTM thresholds...")
    results_rtm = run_rge(g0, t_span, include_thresholds=True)
    
    print("Done!")
    
    # Compare at high scale
    print("\n" + "=" * 70)
    print("COMPARISON AT 10^15 GeV")
    print("=" * 70)
    
    scale = 1e15
    
    _, g_sm, mu_sm = results_sm
    _, g_rtm, mu_rtm = results_rtm
    
    idx_sm = np.argmin(np.abs(mu_sm - scale))
    idx_rtm = np.argmin(np.abs(mu_rtm - scale))
    
    alpha_sm = coupling_to_alpha_inv(g_sm[idx_sm])
    alpha_rtm = coupling_to_alpha_inv(g_rtm[idx_rtm])
    
    spread_sm = np.max(alpha_sm) - np.min(alpha_sm)
    spread_rtm = np.max(alpha_rtm) - np.min(alpha_rtm)
    
    print(f"\n{'Coupling':<10} | {'SM':<12} | {'SM+RTM':<12}")
    print("-" * 40)
    print(f"{'α₁⁻¹':<10} | {alpha_sm[0]:<12.2f} | {alpha_rtm[0]:<12.2f}")
    print(f"{'α₂⁻¹':<10} | {alpha_sm[1]:<12.2f} | {alpha_rtm[1]:<12.2f}")
    print(f"{'α₃⁻¹':<10} | {alpha_sm[2]:<12.2f} | {alpha_rtm[2]:<12.2f}")
    print(f"{'Spread':<10} | {spread_sm:<12.2f} | {spread_rtm:<12.2f}")
    
    print("\n" + "=" * 70)
    print("EFFECT OF THRESHOLDS")
    print("=" * 70)
    print(f"""
    Spread reduction: {spread_sm:.2f} → {spread_rtm:.2f}
    Change: {(spread_rtm - spread_sm)/spread_sm * 100:.1f}%
    
    Note: Thresholds alone do NOT achieve unification.
    The α-shift mechanism (S4) is also required.
    """)
    
    # Save data
    df_thresh = pd.DataFrame({
        'state': [t[0] for t in RTM_THRESHOLDS],
        'mass_GeV': [t[1] for t in RTM_THRESHOLDS],
        'Db1': [t[2] for t in RTM_THRESHOLDS],
        'Db2': [t[3] for t in RTM_THRESHOLDS],
        'Db3': [t[4] for t in RTM_THRESHOLDS]
    })
    df_thresh.to_csv(os.path.join(output_dir, 'S2_threshold_catalogue.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results_sm, results_rtm, output_dir)
    
    # Summary
    summary = f"""S2: Threshold Matching with RTM States
======================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM THRESHOLDS
--------------
{len(RTM_THRESHOLDS)} new states between 10^11 - 10^14 GeV

EFFECT AT 10^15 GeV
-------------------
SM spread:      {spread_sm:.2f}
SM+RTM spread:  {spread_rtm:.2f}
Change:         {(spread_rtm - spread_sm)/spread_sm * 100:.1f}%

CONCLUSION
----------
Thresholds modify the running but alone do not achieve unification.
The α-shift mechanism (Section 3.5.1) provides additional correction.
"""
    
    with open(os.path.join(output_dir, 'S2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
