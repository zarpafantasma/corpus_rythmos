#!/usr/bin/env python3
"""
S1: Arrhenius Classic vs RTM-Modified Kinetics
===============================================

Classical Arrhenius: k = A × exp(-E_a/RT)

RTM-Modified: k = A × (L/L_ref)^(-α) × exp(-E_a,eff/RT)

where:
- L = effective confinement length
- α = coherence exponent of the environment
- E_a,eff may also depend on environment (barrier reshaping)

This simulation demonstrates:
1. How RTM scaling adds a length-dependent pre-exponential factor
2. Temperature-independent confinement effects
3. Combined T and L effects on reaction rates
4. Extraction of α from isothermal confinement series

THEORETICAL MODEL - requires experimental validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

R = 8.314  # J/(mol·K)
kB = 1.381e-23  # J/K


# =============================================================================
# KINETIC MODELS
# =============================================================================

def arrhenius_classic(T, A, Ea):
    """
    Classic Arrhenius equation.
    
    k = A × exp(-E_a / RT)
    
    Parameters:
    -----------
    T : array
        Temperature (K)
    A : float
        Pre-exponential factor (s^-1)
    Ea : float
        Activation energy (J/mol)
    """
    return A * np.exp(-Ea / (R * T))


def arrhenius_rtm(T, L, A_0, Ea_0, alpha, L_ref=100e-9, delta_Ea=0):
    """
    RTM-modified Arrhenius equation.
    
    k = A_0 × (L/L_ref)^(-α) × exp(-E_a,eff / RT)
    
    where E_a,eff = E_a,0 + δE_a(α) allows for barrier reshaping.
    
    Parameters:
    -----------
    T : float or array
        Temperature (K)
    L : float
        Confinement length (m)
    A_0 : float
        Pre-exponential at reference scale (s^-1)
    Ea_0 : float
        Activation energy at reference (J/mol)
    alpha : float
        Coherence exponent
    L_ref : float
        Reference length (m), default 100 nm
    delta_Ea : float
        Barrier change per unit α deviation from 2
    """
    # Scale factor from RTM
    scale_factor = (L / L_ref) ** (-alpha)
    
    # Effective activation energy (optional barrier reshaping)
    Ea_eff = Ea_0 + delta_Ea * (alpha - 2.0)
    
    return A_0 * scale_factor * np.exp(-Ea_eff / (R * T))


def extract_apparent_parameters(T_data, k_data):
    """
    Fit Arrhenius to get apparent A and E_a.
    
    ln(k) = ln(A) - E_a/(R·T)
    """
    inv_T = 1.0 / T_data
    ln_k = np.log(k_data)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_k)
    
    Ea_apparent = -slope * R
    A_apparent = np.exp(intercept)
    
    return {
        'A': A_apparent,
        'Ea': Ea_apparent,
        'r_squared': r_value**2
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    print("=" * 70)
    print("S1: Arrhenius Classic vs RTM-Modified Kinetics")
    print("=" * 70)
    
    output_dir = "/home/claude/013-Rhythmic_Chemistry/S1_arrhenius_rtm/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Parameters
    # ===================
    
    # Reference reaction parameters (typical organic reaction)
    A_0 = 1e13          # s^-1 (pre-exponential)
    Ea_0 = 80e3         # J/mol (~80 kJ/mol)
    L_ref = 100e-9      # 100 nm reference
    
    # Temperature range
    T = np.linspace(300, 500, 50)  # K
    
    # Confinement scales (m)
    L_values = np.array([5, 10, 20, 50, 100, 200]) * 1e-9  # 5-200 nm
    
    # α values for different environments
    alpha_values = [1.5, 2.0, 2.3, 2.5]
    
    results = []
    
    # ===================
    # Part 1: Classic vs RTM at different T
    # ===================
    
    print("\n1. Comparing classic vs RTM Arrhenius...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Arrhenius plots at different L (fixed α)
    ax1 = axes1[0, 0]
    
    alpha_fixed = 2.2
    
    # Classic (bulk)
    k_classic = arrhenius_classic(T, A_0, Ea_0)
    ax1.plot(1000/T, np.log10(k_classic), 'k-', linewidth=2.5, 
             label='Classic (bulk)')
    
    # RTM at different L
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(L_values)))
    for L, color in zip(L_values, colors):
        k_rtm = arrhenius_rtm(T, L, A_0, Ea_0, alpha_fixed, L_ref)
        ax1.plot(1000/T, np.log10(k_rtm), '--', color=color, linewidth=1.5,
                 label=f'L = {L*1e9:.0f} nm')
    
    ax1.set_xlabel('1000/T (K⁻¹)', fontsize=11)
    ax1.set_ylabel('log₁₀(k / s⁻¹)', fontsize=11)
    ax1.set_title(f'Arrhenius Plot: Classic vs RTM (α = {alpha_fixed})\n'
                  f'Confinement shifts intercept, not slope', fontsize=12)
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: k vs L at fixed T (isothermal)
    ax2 = axes1[0, 1]
    
    T_fixed = 350  # K
    L_range = np.logspace(-9, -7, 50)  # 1 nm to 100 nm
    
    for alpha in alpha_values:
        k = arrhenius_rtm(T_fixed, L_range, A_0, Ea_0, alpha, L_ref)
        ax2.plot(L_range * 1e9, k, linewidth=2, label=f'α = {alpha}')
    
    # Classic reference (L-independent)
    k_classic_ref = arrhenius_classic(T_fixed, A_0, Ea_0)
    ax2.axhline(y=k_classic_ref, color='gray', linestyle='--', linewidth=2,
                label='Classic (no L dependence)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax2.set_ylabel('Rate Constant k (s⁻¹)', fontsize=11)
    ax2.set_title(f'Isothermal Rate vs Confinement (T = {T_fixed} K)\n'
                  f'RTM: k ∝ L^(-α)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Enhancement factor vs L
    ax3 = axes1[1, 0]
    
    for alpha in alpha_values:
        k_rtm = arrhenius_rtm(T_fixed, L_range, A_0, Ea_0, alpha, L_ref)
        enhancement = k_rtm / k_classic_ref
        ax3.plot(L_range * 1e9, enhancement, linewidth=2, label=f'α = {alpha}')
    
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax3.fill_between([1, 100], [1, 1], [1000, 1000], alpha=0.1, color='green')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax3.set_ylabel('Enhancement Factor (k_RTM / k_classic)', fontsize=11)
    ax3.set_title('Rate Enhancement by Nanoconfinement', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Annotate enhancement at 10 nm
    for alpha in alpha_values:
        enh_10nm = (10e-9 / L_ref) ** (-alpha)
        ax3.annotate(f'{enh_10nm:.0f}×', xy=(10, enh_10nm), fontsize=9)
    
    # Plot 4: Apparent vs true parameters
    ax4 = axes1[1, 1]
    
    # Simulate experiment: measure k at multiple T for different L
    T_exp = np.array([320, 340, 360, 380, 400])
    
    results_params = []
    
    for L in [10e-9, 50e-9, 100e-9]:
        k_measured = arrhenius_rtm(T_exp, L, A_0, Ea_0, 2.2, L_ref)
        # Add noise
        k_measured *= np.exp(0.05 * np.random.randn(len(T_exp)))
        
        params = extract_apparent_parameters(T_exp, k_measured)
        results_params.append({
            'L_nm': L * 1e9,
            'A_apparent': params['A'],
            'Ea_apparent': params['Ea'] / 1000,  # kJ/mol
            'A_ratio': params['A'] / A_0,
            'Ea_ratio': params['Ea'] / Ea_0
        })
        
        # Plot Arrhenius fit
        ax4.scatter(1000/T_exp, np.log10(k_measured), s=50, label=f'L = {L*1e9:.0f} nm')
    
    # Add classic
    k_classic_exp = arrhenius_classic(T_exp, A_0, Ea_0)
    ax4.scatter(1000/T_exp, np.log10(k_classic_exp), s=50, marker='s', 
                color='black', label='Bulk (classic)')
    
    ax4.set_xlabel('1000/T (K⁻¹)', fontsize=11)
    ax4.set_ylabel('log₁₀(k / s⁻¹)', fontsize=11)
    ax4.set_title('Apparent Arrhenius Parameters at Different L\n'
                  '(Same slope, different intercept)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_arrhenius_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_arrhenius_comparison.pdf'))
    plt.close()
    
    # ===================
    # Part 2: α extraction from confinement data
    # ===================
    
    print("\n2. Extracting α from isothermal confinement data...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate isothermal data
    alpha_true = 2.3
    L_exp = np.array([5, 10, 20, 50, 100, 200]) * 1e-9
    
    k_exp = arrhenius_rtm(T_fixed, L_exp, A_0, Ea_0, alpha_true, L_ref)
    k_exp *= np.exp(0.08 * np.random.randn(len(L_exp)))  # Add noise
    
    # Fit log-log
    log_L = np.log(L_exp)
    log_k = np.log(k_exp)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_L, log_k)
    alpha_recovered = -slope
    
    ax = axes2[0]
    ax.scatter(L_exp * 1e9, k_exp, s=80, c='blue', zorder=3)
    
    L_fit = np.logspace(-9, -7, 50)
    k_fit = np.exp(intercept) * L_fit ** slope
    ax.plot(L_fit * 1e9, k_fit, 'r--', linewidth=2, 
            label=f'Fit: α = {alpha_recovered:.2f}')
    
    k_true = arrhenius_rtm(T_fixed, L_fit, A_0, Ea_0, alpha_true, L_ref)
    ax.plot(L_fit * 1e9, k_true, 'g:', linewidth=2,
            label=f'True: α = {alpha_true}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Confinement L (nm)', fontsize=11)
    ax.set_ylabel('Rate Constant k (s⁻¹)', fontsize=11)
    ax.set_title(f'α Recovery from Isothermal Data (T = {T_fixed} K)\n'
                 f'True α = {alpha_true}, Recovered α = {alpha_recovered:.2f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Summary table
    ax = axes2[1]
    ax.axis('off')
    
    summary_text = f"""RTM vs Classic Arrhenius Summary
{'='*40}

CLASSIC ARRHENIUS:
  k = A × exp(-Eₐ/RT)
  No L dependence

RTM-MODIFIED:
  k = A₀ × (L/L_ref)^(-α) × exp(-Eₐ/RT)
  
  α = coherence exponent
  L = confinement length

KEY PREDICTIONS:
  1. Isothermal: log(k) vs log(L) has slope -α
  2. Different L → same Eₐ, different apparent A
  3. Enhancement at L=10nm (α=2.3): {(10e-9/L_ref)**(-2.3):.0f}×

PARAMETER RECOVERY:
  True α = {alpha_true}
  Recovered α = {alpha_recovered:.3f}
  Error = {abs(alpha_recovered - alpha_true):.3f}
  R² = {r_value**2:.4f}
"""
    
    ax.text(0.1, 0.95, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_alpha_extraction.png'), dpi=150)
    plt.close()
    
    # Save parameter comparison
    df_params = pd.DataFrame(results_params)
    df_params.to_csv(os.path.join(output_dir, 'S1_apparent_parameters.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S1: Arrhenius Classic vs RTM-Modified Kinetics
===============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

THEORETICAL FRAMEWORK
---------------------
Classic Arrhenius: k = A × exp(-Eₐ/RT)

RTM-Modified: k = A₀ × (L/L_ref)^(-α) × exp(-Eₐ_eff/RT)

where:
  L = effective confinement length (nm)
  α = coherence exponent of environment
  L_ref = reference scale (100 nm)

RTM PREDICTIONS
---------------
1. ISOTHERMAL SCALING (fixed T, vary L):
   log(k) vs log(L) has slope = -α
   Smaller L → faster reaction

2. ARRHENIUS ANALYSIS (fixed L, vary T):
   Same activation energy Eₐ
   Different apparent pre-exponential A_app ∝ L^(-α)

3. ENHANCEMENT FACTORS (at 10 nm confinement):
   α = 1.5: {(10e-9/L_ref)**(-1.5):.0f}× enhancement
   α = 2.0: {(10e-9/L_ref)**(-2.0):.0f}× enhancement
   α = 2.3: {(10e-9/L_ref)**(-2.3):.0f}× enhancement
   α = 2.5: {(10e-9/L_ref)**(-2.5):.0f}× enhancement

PARAMETER RECOVERY
------------------
True α: {alpha_true}
Recovered α: {alpha_recovered:.3f}
Error: {abs(alpha_recovered - alpha_true):.4f}
R²: {r_value**2:.4f}

EXPERIMENTAL IMPLICATIONS
-------------------------
1. Isothermal experiments at varying L reveal α
2. Different confinement platforms should give same α
   for same reaction mechanism
3. Temperature studies reveal Eₐ independently

FALSIFICATION CRITERIA
----------------------
- Slope of log(k) vs log(L) should be constant (stable α)
- Different confinement methods should agree
- No α dependence → reaction not RTM-governed
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nα recovery error: {abs(alpha_recovered - alpha_true):.4f}")
    print(f"Enhancement at 10nm (α=2.3): {(10e-9/L_ref)**(-2.3):.0f}×")
    print(f"\nOutputs: {output_dir}/")
    
    return df_params


if __name__ == "__main__":
    main()
