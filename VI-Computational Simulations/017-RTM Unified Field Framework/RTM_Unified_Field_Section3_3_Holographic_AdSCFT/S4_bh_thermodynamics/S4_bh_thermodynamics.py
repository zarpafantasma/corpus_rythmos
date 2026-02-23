#!/usr/bin/env python3
"""
S4: Black Hole Thermodynamics with RTM
======================================

From "RTM Unified Field Framework" - Section 3.3.3

Implements black hole thermodynamics with RTM modifications:
1. Modified Hawking temperature with α-dependence
2. Generalized Bekenstein bound

Key Equations (from paper):
    T_H = ℏc³/(8πGM) × α_H
    
    S_max = (2π/ℏc) × E × R / α
    
    "Maximal information storage scales inversely with the 
     local temporal-scaling exponent"

Reference: Paper Section 3.3.3 "Black Hole Thermodynamics"
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# PHYSICAL CONSTANTS (Natural units: G = c = ℏ = k_B = 1)
# =============================================================================

# In natural units
G_NEWTON = 1.0
C_LIGHT = 1.0
HBAR = 1.0
K_BOLTZ = 1.0

# Reference scale
M_PLANCK = 1.0
L_PLANCK = 1.0

# RTM parameters
ALPHA_REF = 2.0  # Reference α value


# =============================================================================
# STANDARD BLACK HOLE THERMODYNAMICS
# =============================================================================

def schwarzschild_radius(M, G=G_NEWTON, c=C_LIGHT):
    """
    Schwarzschild radius: r_s = 2GM/c²
    """
    return 2 * G * M / c**2


def hawking_temperature_standard(M, G=G_NEWTON, c=C_LIGHT, hbar=HBAR, k_B=K_BOLTZ):
    """
    Standard Hawking temperature.
    
    T_H = ℏc³ / (8πGMk_B)
    """
    return hbar * c**3 / (8 * np.pi * G * M * k_B)


def bekenstein_entropy_standard(M, G=G_NEWTON, c=C_LIGHT, hbar=HBAR, k_B=K_BOLTZ):
    """
    Bekenstein-Hawking entropy.
    
    S = A / (4 L_P²) = 4πG²M²/(ℏc)
    """
    A = 4 * np.pi * schwarzschild_radius(M)**2
    L_P_sq = G * hbar / c**3
    return A / (4 * L_P_sq)


def bekenstein_bound_standard(E, R, hbar=HBAR, c=C_LIGHT):
    """
    Standard Bekenstein bound.
    
    S_max ≤ 2πER / (ℏc)
    """
    return 2 * np.pi * E * R / (hbar * c)


# =============================================================================
# RTM-MODIFIED THERMODYNAMICS
# =============================================================================

def hawking_temperature_rtm(M, alpha, G=G_NEWTON, c=C_LIGHT, hbar=HBAR, k_B=K_BOLTZ):
    """
    RTM-modified Hawking temperature.
    
    T_H^RTM = T_H^standard × α_H
    
    From paper: "identifies α with horizon red-shift effects"
    """
    T_standard = hawking_temperature_standard(M, G, c, hbar, k_B)
    return T_standard * alpha


def bekenstein_bound_rtm(E, R, alpha, hbar=HBAR, c=C_LIGHT):
    """
    RTM Generalized Bekenstein Bound.
    
    S_max^RTM = (2π/ℏc) × E × R / α
    
    From paper: "maximal information storage scales inversely 
    with the local temporal-scaling exponent"
    """
    S_standard = bekenstein_bound_standard(E, R, hbar, c)
    return S_standard / alpha


def entropy_rtm(M, alpha, G=G_NEWTON, c=C_LIGHT, hbar=HBAR, k_B=K_BOLTZ):
    """
    RTM-modified black hole entropy.
    
    S^RTM = S_standard / α
    """
    S_standard = bekenstein_entropy_standard(M, G, c, hbar, k_B)
    return S_standard / alpha


def evaporation_time_rtm(M, alpha, G=G_NEWTON, c=C_LIGHT, hbar=HBAR):
    """
    RTM-modified black hole evaporation time.
    
    t_evap ∝ M³/α  (faster evaporation with higher α)
    """
    t_standard = 5120 * np.pi * G**2 * M**3 / (hbar * c**4)
    return t_standard / alpha


# =============================================================================
# INFORMATION BOUNDS AND BRANCH-JUMP CONSTRAINTS
# =============================================================================

def max_extractable_energy(M, alpha, eta=0.5):
    """
    Maximum extractable energy from RTM system.
    
    E_max = η × M × c² / α
    
    Higher α → less extractable energy (tighter bound).
    """
    return eta * M * C_LIGHT**2 / alpha


def branch_jump_threshold(M, alpha_initial, alpha_final, Delta_V=1.0):
    """
    Energy threshold for branch-jump transition.
    
    From paper: "enforcing limits on energy extraction 
    and branch-hop transitions"
    
    E_threshold = ΔV × M × |α_f - α_i|
    """
    return Delta_V * M * np.abs(alpha_final - alpha_initial)


def information_vault_capacity(M, alpha, R, c=C_LIGHT, hbar=HBAR):
    """
    Information capacity in RTM "vault" near horizon.
    
    From paper: "as α→∞ near singularity, proper time freezes 
    and information is stored in a finite-coherence vault"
    
    I_max = S_max^RTM (in bits)
    """
    E = M * c**2
    S_max = bekenstein_bound_rtm(E, R, alpha, hbar, c)
    return S_max / np.log(2)  # Convert to bits


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Hawking temperature vs α
    ax1 = axes[0, 0]
    
    M_test = 1.0  # Test mass
    alphas = np.linspace(1.0, 4.0, 100)
    
    T_standard = hawking_temperature_standard(M_test)
    T_rtm = [hawking_temperature_rtm(M_test, a) for a in alphas]
    
    ax1.plot(alphas, T_rtm, 'r-', linewidth=2, label='T_H^RTM(α)')
    ax1.axhline(y=T_standard, color='blue', linestyle='--', linewidth=2, 
                label='T_H^standard')
    
    ax1.set_xlabel('α (RTM exponent)', fontsize=12)
    ax1.set_ylabel('Hawking Temperature T_H', fontsize=12)
    ax1.set_title('RTM-Modified Hawking Temperature', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bekenstein bound vs α
    ax2 = axes[0, 1]
    
    E_test = 1.0
    R_test = 1.0
    
    S_standard = bekenstein_bound_standard(E_test, R_test)
    S_rtm = [bekenstein_bound_rtm(E_test, R_test, a) for a in alphas]
    
    ax2.plot(alphas, S_rtm, 'g-', linewidth=2, label='S_max^RTM(α)')
    ax2.axhline(y=S_standard, color='blue', linestyle='--', linewidth=2,
                label='S_max^standard')
    
    ax2.set_xlabel('α', fontsize=12)
    ax2.set_ylabel('Maximum Entropy S_max', fontsize=12)
    ax2.set_title('Generalized Bekenstein Bound', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy and temperature vs mass for different α
    ax3 = axes[1, 0]
    
    M_range = np.linspace(0.1, 10, 100)
    alpha_values = [2.0, 2.5, 3.0]
    colors = ['blue', 'green', 'red']
    
    for alpha, color in zip(alpha_values, colors):
        T = [hawking_temperature_rtm(m, alpha) for m in M_range]
        ax3.loglog(M_range, T, color=color, linewidth=2, label=f'α = {alpha}')
    
    ax3.set_xlabel('Black Hole Mass M', fontsize=12)
    ax3.set_ylabel('Temperature T_H', fontsize=12)
    ax3.set_title('Temperature vs Mass for Different α', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Information capacity vs α at RTM bands
    ax4 = axes[1, 1]
    
    rtm_bands = [2.0, 2.26, 2.47, 2.61, 2.72]
    M_test = 1.0
    R_test = schwarzschild_radius(M_test)
    
    I_capacity = [information_vault_capacity(M_test, a, R_test) for a in rtm_bands]
    
    ax4.bar(range(len(rtm_bands)), I_capacity, color='purple', alpha=0.7)
    ax4.set_xticks(range(len(rtm_bands)))
    ax4.set_xticklabels([f'α={a}' for a in rtm_bands], rotation=45)
    ax4.set_ylabel('Information Capacity (bits)', fontsize=12)
    ax4.set_title('Information Vault Capacity at RTM Bands', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S4_bh_thermodynamics.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S4_bh_thermodynamics.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S4: Black Hole Thermodynamics with RTM")
    print("From: RTM Unified Field Framework - Section 3.3.3")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RTM-MODIFIED THERMODYNAMICS")
    print("=" * 70)
    print("""
    Key modifications from paper:
    
    1. HAWKING TEMPERATURE:
       T_H^RTM = T_H^standard × α_H
       "identifies α with horizon red-shift effects"
       
    2. GENERALIZED BEKENSTEIN BOUND:
       S_max^RTM = (2π/ℏc) × E × R / α
       "maximal information storage scales inversely with α"
       
    3. INFORMATION VAULT:
       "as α→∞ near singularity, proper time freezes and 
        information is stored in a finite-coherence vault"
    """)
    
    # Hawking temperature comparison
    print("=" * 70)
    print("HAWKING TEMPERATURE")
    print("=" * 70)
    
    M_test = 1.0  # Planck masses
    T_std = hawking_temperature_standard(M_test)
    
    print(f"\n    For M = {M_test} M_P:")
    print(f"    T_H^standard = {T_std:.4f}")
    print(f"\n    | α   | T_H^RTM  | Ratio |")
    print(f"    |-----|----------|-------|")
    
    for alpha in [1.5, 2.0, 2.5, 3.0, 3.5]:
        T_rtm = hawking_temperature_rtm(M_test, alpha)
        print(f"    | {alpha:<3} | {T_rtm:<8.4f} | {T_rtm/T_std:<5.2f} |")
    
    # Bekenstein bound
    print("\n" + "=" * 70)
    print("GENERALIZED BEKENSTEIN BOUND")
    print("=" * 70)
    
    E_test = 1.0
    R_test = 1.0
    S_std = bekenstein_bound_standard(E_test, R_test)
    
    print(f"\n    For E = {E_test}, R = {R_test}:")
    print(f"    S_max^standard = {S_std:.4f}")
    print(f"\n    | α   | S_max^RTM | Ratio |")
    print(f"    |-----|-----------|-------|")
    
    for alpha in [1.5, 2.0, 2.5, 3.0, 3.5]:
        S_rtm = bekenstein_bound_rtm(E_test, R_test, alpha)
        print(f"    | {alpha:<3} | {S_rtm:<9.4f} | {S_rtm/S_std:<5.2f} |")
    
    print(f"""
    
    Key insight: Higher α → lower entropy bound
    This enforces stricter limits on information storage.
    """)
    
    # Branch-jump constraints
    print("=" * 70)
    print("BRANCH-JUMP CONSTRAINTS")
    print("=" * 70)
    
    M_system = 1.0
    
    print(f"\n    Energy thresholds for branch transitions:")
    print(f"    (For M = {M_system})")
    print(f"\n    | Transition  | E_threshold |")
    print(f"    |-------------|-------------|")
    
    transitions = [(2.0, 2.26), (2.26, 2.47), (2.47, 2.61), (2.61, 2.72)]
    for a_i, a_f in transitions:
        E_thresh = branch_jump_threshold(M_system, a_i, a_f)
        print(f"    | {a_i} → {a_f} | {E_thresh:<11.4f} |")
    
    # Information vault
    print("\n" + "=" * 70)
    print("INFORMATION VAULT CAPACITY")
    print("=" * 70)
    
    M_bh = 1.0
    R_bh = schwarzschild_radius(M_bh)
    
    print(f"\n    For M = {M_bh} M_P, R_s = {R_bh:.4f} L_P:")
    print(f"\n    | RTM Band | I_max (bits) |")
    print(f"    |----------|--------------|")
    
    rtm_bands = [2.0, 2.26, 2.47, 2.61, 2.72]
    for alpha in rtm_bands:
        I_max = information_vault_capacity(M_bh, alpha, R_bh)
        print(f"    | α = {alpha:<4} | {I_max:<12.2f} |")
    
    print("""
    
    Physical interpretation:
    - Higher α bands store less information
    - At α → ∞, capacity → 0 (frozen time)
    - This regulates singularity information paradox
    """)
    
    # Save data
    alphas = np.linspace(1.0, 4.0, 100)
    df = pd.DataFrame({
        'alpha': alphas,
        'T_H_rtm': [hawking_temperature_rtm(M_test, a) for a in alphas],
        'S_max_rtm': [bekenstein_bound_rtm(E_test, R_test, a) for a in alphas],
        'I_vault': [information_vault_capacity(M_bh, a, R_bh) for a in alphas]
    })
    df.to_csv(os.path.join(output_dir, 'S4_thermodynamics_vs_alpha.csv'), index=False)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Summary
    summary = f"""S4: Black Hole Thermodynamics with RTM
======================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM MODIFICATIONS
-----------------
T_H^RTM = T_H × α (temperature enhanced)
S_max^RTM = S_max / α (entropy bound tightened)
I_vault ∝ 1/α (information capacity reduced)

HAWKING TEMPERATURE
-------------------
Standard (M=1): {hawking_temperature_standard(1.0):.4f}
At α=2.0: {hawking_temperature_rtm(1.0, 2.0):.4f}
At α=3.0: {hawking_temperature_rtm(1.0, 3.0):.4f}

BEKENSTEIN BOUND
----------------
Standard (E=R=1): {bekenstein_bound_standard(1.0, 1.0):.4f}
At α=2.0: {bekenstein_bound_rtm(1.0, 1.0, 2.0):.4f}
At α=3.0: {bekenstein_bound_rtm(1.0, 1.0, 3.0):.4f}

PAPER VERIFICATION
------------------
✓ Modified Hawking temperature computed
✓ Generalized Bekenstein bound implemented
✓ Information vault capacity at RTM bands
✓ Branch-jump energy thresholds computed
"""
    
    with open(os.path.join(output_dir, 'S4_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
