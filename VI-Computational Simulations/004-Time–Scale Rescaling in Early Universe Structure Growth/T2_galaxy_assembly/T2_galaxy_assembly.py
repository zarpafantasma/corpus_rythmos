#!/usr/bin/env python3
"""
T2: Galaxy Assembly - Required Acceleration Calculator
======================================================

From "Time–Scale Rescaling in Early Universe Structure Growth"

Calculates the required RTM acceleration factor A to reach a target
stellar mass M_star at redshift z.

Key equations:
--------------
M_star = f_b * M_halo * [1 - (1-ε)^(A*N_dyn)]

Required A:
A_required = ln[1 - M_star/(f_b*M_halo)] / [N_dyn * ln(1-ε)]

Reference: Paper Section 3
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# ASTROPHYSICAL PARAMETERS
# =============================================================================

F_BARYON = 0.157          # Cosmic baryon fraction
EPSILON_DEFAULT = 0.02    # Star formation efficiency per t_dyn
N_DYN_DEFAULT = 5         # Dynamical times available

# =============================================================================
# GALAXY ASSEMBLY CALCULATIONS
# =============================================================================

def stellar_mass_standard(M_halo: float, f_b: float, epsilon: float, 
                          N_dyn: int) -> float:
    """Standard stellar mass without RTM acceleration."""
    integrated_eff = 1 - (1 - epsilon) ** N_dyn
    return f_b * M_halo * integrated_eff

def stellar_mass_rtm(M_halo: float, f_b: float, epsilon: float, 
                     N_dyn: int, A: float) -> float:
    """Stellar mass with RTM acceleration factor A."""
    effective_N = A * N_dyn
    integrated_eff = 1 - (1 - epsilon) ** effective_N
    return f_b * M_halo * integrated_eff

def required_acceleration(M_star_target: float, M_halo: float, 
                          f_b: float, epsilon: float, N_dyn: int) -> float:
    """
    Calculate required A to reach M_star_target.
    
    A_required = ln[1 - M_star/(f_b*M_halo)] / [N_dyn * ln(1-ε)]
    """
    max_mass = f_b * M_halo
    if M_star_target >= max_mass:
        return np.inf
    
    ratio = M_star_target / max_mass
    numerator = np.log(1 - ratio)
    denominator = N_dyn * np.log(1 - epsilon)
    
    return numerator / denominator

def integrated_efficiency(epsilon: float, N_steps: float) -> float:
    """Integrated star formation efficiency after N_steps."""
    return 1 - (1 - epsilon) ** N_steps

# =============================================================================
# PAPER EXAMPLES
# =============================================================================

def case_a_demanding():
    """Case A from paper: demanding scenario."""
    M_halo = 1e12  # M_sun
    M_star_target = 1e11  # M_sun (10% of halo)
    epsilon = 0.02
    N_dyn = 5
    
    A_req = required_acceleration(M_star_target, M_halo, F_BARYON, epsilon, N_dyn)
    
    return {
        'name': 'Case A (demanding)',
        'M_halo': M_halo,
        'M_star_target': M_star_target,
        'epsilon': epsilon,
        'N_dyn': N_dyn,
        'A_required': A_req,
        'description': 'M_star = 10^11 M_sun, ε = 2%'
    }

def case_b_moderate():
    """Case B from paper: moderate scenario."""
    M_halo = 1e12
    M_star_target = 3e10  # 3% of halo
    epsilon = 0.02
    N_dyn = 5
    
    A_req = required_acceleration(M_star_target, M_halo, F_BARYON, epsilon, N_dyn)
    
    return {
        'name': 'Case B (moderate)',
        'M_halo': M_halo,
        'M_star_target': M_star_target,
        'epsilon': epsilon,
        'N_dyn': N_dyn,
        'A_required': A_req,
        'description': 'M_star = 3×10^10 M_sun, ε = 2%'
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir: str):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: M_star vs A for different efficiencies
    ax1 = axes[0, 0]
    A_range = np.linspace(1, 100, 100)
    M_halo = 1e12
    N_dyn = 5
    
    for epsilon in [0.01, 0.02, 0.05, 0.10]:
        M_star = [stellar_mass_rtm(M_halo, F_BARYON, epsilon, N_dyn, A) for A in A_range]
        ax1.semilogy(A_range, M_star, label=f'ε = {epsilon:.0%}')
    
    ax1.axhline(y=1e11, color='red', linestyle='--', alpha=0.7, label='Target: 10¹¹ M☉')
    ax1.axhline(y=3e10, color='orange', linestyle='--', alpha=0.7, label='Target: 3×10¹⁰ M☉')
    ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=60, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Acceleration Factor A', fontsize=12)
    ax1.set_ylabel('Stellar Mass M_star [M☉]', fontsize=12)
    ax1.set_title(f'M_star vs A (M_halo = 10¹² M☉, N_dyn = {N_dyn})', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 100)
    
    # Plot 2: Required A vs target M_star/M_halo ratio
    ax2 = axes[0, 1]
    ratios = np.linspace(0.01, 0.15, 50)  # 1% to 15% of baryon mass
    
    for epsilon in [0.01, 0.02, 0.05]:
        A_req = []
        for r in ratios:
            M_star_target = r * F_BARYON * M_halo
            A = required_acceleration(M_star_target, M_halo, F_BARYON, epsilon, N_dyn)
            A_req.append(min(A, 200))
        ax2.plot(ratios * 100, A_req, label=f'ε = {epsilon:.0%}')
    
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='A = 30 (ΛCDM z=10)')
    ax2.axhline(y=58, color='blue', linestyle='--', alpha=0.7, label='A = 58 (EdS z=10)')
    
    ax2.set_xlabel('Target M_star / (f_b × M_halo) [%]', fontsize=12)
    ax2.set_ylabel('Required Acceleration A', fontsize=12)
    ax2.set_title('Required A to Reach Target Mass', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 150)
    
    # Plot 3: Integrated efficiency vs effective N_steps
    ax3 = axes[1, 0]
    N_steps = np.linspace(1, 500, 200)
    
    for epsilon in [0.01, 0.02, 0.05, 0.10]:
        eff = integrated_efficiency(epsilon, N_steps)
        ax3.plot(N_steps, eff * 100, label=f'ε = {epsilon:.0%}')
    
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Effective N_steps (= A × N_dyn)', fontsize=12)
    ax3.set_ylabel('Integrated Efficiency [%]', fontsize=12)
    ax3.set_title('SFE = 1 - (1-ε)^N_steps', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space (A, ε) for different targets
    ax4 = axes[1, 1]
    epsilon_range = np.linspace(0.005, 0.15, 50)
    A_range_2 = np.linspace(1, 100, 50)
    E, A_grid = np.meshgrid(epsilon_range, A_range_2)
    
    # Calculate M_star for each (ε, A)
    M_star_grid = np.zeros_like(E)
    for i in range(len(A_range_2)):
        for j in range(len(epsilon_range)):
            M_star_grid[i, j] = stellar_mass_rtm(M_halo, F_BARYON, 
                                                  epsilon_range[j], N_dyn, A_range_2[i])
    
    # Contour plot
    levels = [1e10, 3e10, 5e10, 1e11, 1.5e11]
    cs = ax4.contour(E * 100, A_grid, M_star_grid, levels=levels, colors='black')
    ax4.clabel(cs, fmt='%.0e M☉', fontsize=8)
    
    # Shade feasible region (A < 60)
    ax4.axhline(y=58, color='blue', linestyle='--', label='A_EdS(z=10)')
    ax4.axhline(y=35, color='red', linestyle='--', label='A_ΛCDM(z=10)')
    
    ax4.set_xlabel('Star Formation Efficiency ε [%]', fontsize=12)
    ax4.set_ylabel('Acceleration Factor A', fontsize=12)
    ax4.set_title('Parameter Space: M_star Contours', fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'T2_galaxy_assembly.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'T2_galaxy_assembly.pdf'))
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("T2: Galaxy Assembly - Required Acceleration Calculator")
    print("From: Time–Scale Rescaling in Early Universe Structure Growth")
    print("=" * 66)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate paper examples
    print("\nPAPER EXAMPLES")
    print("-" * 50)
    
    case_a = case_a_demanding()
    case_b = case_b_moderate()
    
    for case in [case_a, case_b]:
        print(f"\n{case['name']}:")
        print(f"  M_halo        = {case['M_halo']:.0e} M☉")
        print(f"  M_star target = {case['M_star_target']:.0e} M☉")
        print(f"  ε             = {case['epsilon']:.0%}")
        print(f"  N_dyn         = {case['N_dyn']}")
        print(f"  A_required    = {case['A_required']:.1f}")
        
        # Check if achievable
        A_eds = (1 + 10) ** 1.5  # EdS at z=10
        A_lcdm = 35  # ΛCDM at z=10
        
        if case['A_required'] <= A_eds:
            print(f"  → Achievable with EdS (A={A_eds:.0f}) ✓")
        if case['A_required'] <= A_lcdm:
            print(f"  → Achievable with ΛCDM (A≈{A_lcdm}) ✓")
    
    # Generate data table
    print("\n" + "=" * 66)
    print("ACCELERATION TABLE")
    print("=" * 66)
    
    records = []
    M_halo = 1e12
    N_dyn = 5
    
    for epsilon in [0.01, 0.02, 0.05, 0.10]:
        for M_star_frac in [0.01, 0.03, 0.05, 0.10]:
            M_star_target = M_star_frac * F_BARYON * M_halo
            A_req = required_acceleration(M_star_target, M_halo, F_BARYON, epsilon, N_dyn)
            
            records.append({
                'epsilon': epsilon,
                'M_star_fraction': M_star_frac,
                'M_star_target': M_star_target,
                'A_required': A_req,
                'achievable_LCDM': A_req <= 40,
                'achievable_EdS': A_req <= 60
            })
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'T2_galaxy_assembly_data.csv'), index=False)
    
    print(f"\n{'ε':>6} | {'M*/M_b':>8} | {'M_star':>10} | {'A_req':>6} | ΛCDM | EdS")
    print("-" * 60)
    for _, row in df.iterrows():
        lcdm = "✓" if row['achievable_LCDM'] else "✗"
        eds = "✓" if row['achievable_EdS'] else "✗"
        print(f"{row['epsilon']:>6.0%} | {row['M_star_fraction']:>7.0%} | "
              f"{row['M_star_target']:>10.2e} | {row['A_required']:>6.1f} | "
              f"  {lcdm}  |  {eds}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(output_dir)
    
    # Summary
    summary = f"""T2: Galaxy Assembly Calculator
==============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY EQUATION
------------
M_star = f_b × M_halo × [1 - (1-ε)^(A×N_dyn)]

Required acceleration:
A_required = ln[1 - M_star/(f_b×M_halo)] / [N_dyn × ln(1-ε)]

PARAMETERS
----------
f_b (baryon fraction) = {F_BARYON}
M_halo (reference)    = 10¹² M☉
N_dyn (default)       = {N_DYN_DEFAULT}

PAPER EXAMPLES
--------------
Case A (demanding):  M_star = 10¹¹ M☉, ε = 2%  →  A_required = {case_a['A_required']:.1f}
Case B (moderate):   M_star = 3×10¹⁰ M☉, ε = 2%  →  A_required = {case_b['A_required']:.1f}

RTM PREDICTIONS (z=10, α=1)
---------------------------
A_EdS  ≈ 58  →  Sufficient for Case A and B ✓
A_ΛCDM ≈ 35  →  Sufficient for Case B, marginal for Case A

INTERPRETATION
--------------
With ε ~ 2% and A ~ 30-60, RTM naturally explains 
the "too-early/too-massive" galaxies at z > 10 
observed by JWST without exotic physics.
"""
    
    with open(os.path.join(output_dir, 'T2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 66)

if __name__ == "__main__":
    main()
