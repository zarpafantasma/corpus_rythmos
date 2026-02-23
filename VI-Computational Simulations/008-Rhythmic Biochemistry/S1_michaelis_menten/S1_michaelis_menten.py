#!/usr/bin/env python3
"""
S1: RTM-Modified Michaelis-Menten Kinetics
==========================================

Classical Michaelis-Menten: v = V_max * [S] / (K_m + [S])

RTM modification: k_cat (and thus V_max) scales with confinement:
    k_cat(L) = k_cat,0 * L^(-α)

where:
- L = effective confinement length (nm)
- α = coherence exponent (transport class)
- k_cat,0 = rate constant at reference scale

This simulation shows:
1. How RTM scaling affects enzyme kinetics curves
2. The effect of different α values
3. Apparent K_m changes under confinement (crowding effects)

THEORETICAL MODEL - requires experimental validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM-MODIFIED MICHAELIS-MENTEN
# =============================================================================

def michaelis_menten_classic(S, Vmax, Km):
    """Classic Michaelis-Menten equation."""
    return Vmax * S / (Km + S)


def kcat_rtm(L, kcat_0, alpha, L_ref=100):
    """
    RTM scaling for k_cat.
    
    k_cat(L) = k_cat,0 * (L/L_ref)^(-α)
    
    Smaller L → faster catalysis (if α > 0)
    """
    return kcat_0 * (L / L_ref) ** (-alpha)


def michaelis_menten_rtm(S, L, kcat_0, Km_0, E_total, alpha, L_ref=100, 
                          crowding_factor=0.0):
    """
    RTM-modified Michaelis-Menten kinetics.
    
    RTM affects:
    - k_cat scales as L^(-α)
    - K_m may increase with crowding (optional effect)
    
    Parameters:
    -----------
    S : array
        Substrate concentration
    L : float
        Confinement length (nm)
    kcat_0 : float
        k_cat at reference scale (s^-1)
    Km_0 : float
        K_m at reference scale (μM)
    E_total : float
        Total enzyme concentration (μM)
    alpha : float
        RTM coherence exponent
    L_ref : float
        Reference scale (nm)
    crowding_factor : float
        How much K_m increases with confinement (0 = no effect)
    """
    # RTM scaling of k_cat
    kcat = kcat_rtm(L, kcat_0, alpha, L_ref)
    
    # Optional: crowding effect on K_m
    # Smaller L → more crowded → higher apparent K_m
    Km = Km_0 * (1 + crowding_factor * (L_ref / L - 1))
    
    # V_max = k_cat * [E]_total
    Vmax = kcat * E_total
    
    return michaelis_menten_classic(S, Vmax, Km)


def compute_kcat_apparent(v_data, S_data, E_total):
    """
    Fit Michaelis-Menten to extract apparent k_cat and K_m.
    """
    def mm_fit(S, Vmax, Km):
        return Vmax * S / (Km + S)
    
    try:
        popt, pcov = curve_fit(mm_fit, S_data, v_data, 
                               p0=[max(v_data), np.median(S_data)],
                               bounds=([0, 0], [np.inf, np.inf]))
        Vmax_fit, Km_fit = popt
        kcat_fit = Vmax_fit / E_total
        
        # Standard errors
        perr = np.sqrt(np.diag(pcov))
        
        return {
            'kcat': kcat_fit,
            'Km': Km_fit,
            'Vmax': Vmax_fit,
            'kcat_err': perr[0] / E_total,
            'Km_err': perr[1]
        }
    except:
        return None


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    print("=" * 70)
    print("S1: RTM-Modified Michaelis-Menten Kinetics")
    print("=" * 70)
    
    output_dir = "/home/claude/012-Rhythmic_Biochemistry/S1_michaelis_menten/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Parameters
    # ===================
    
    # Reference enzyme parameters (typical values)
    kcat_0 = 100.0      # s^-1 at L_ref
    Km_0 = 50.0         # μM
    E_total = 0.1       # μM enzyme
    L_ref = 100.0       # nm (reference: bulk solution)
    
    # Substrate concentrations for kinetic curves
    S = np.logspace(-1, 3, 50)  # 0.1 to 1000 μM
    
    # Confinement scales to test
    L_values = np.array([10, 20, 50, 100, 200])  # nm
    
    # α values representing different transport classes
    alpha_values = [1.5, 2.0, 2.3, 2.5]
    
    results = []
    
    # ===================
    # Part 1: Effect of α on kinetics at fixed L
    # ===================
    print("\n1. Effect of α on kinetics at different confinements...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    
    L_demo = 20  # 20 nm confinement (nanopore)
    
    ax1 = axes1[0, 0]
    for alpha in alpha_values:
        v = michaelis_menten_rtm(S, L_demo, kcat_0, Km_0, E_total, alpha, L_ref)
        kcat_eff = kcat_rtm(L_demo, kcat_0, alpha, L_ref)
        ax1.plot(S, v, linewidth=2, label=f'α={alpha} (k_cat={kcat_eff:.1f} s⁻¹)')
    
    # Add bulk reference
    v_bulk = michaelis_menten_rtm(S, L_ref, kcat_0, Km_0, E_total, 2.0, L_ref)
    ax1.plot(S, v_bulk, 'k--', linewidth=2, label=f'Bulk (L={L_ref}nm)')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('[S] (μM)', fontsize=11)
    ax1.set_ylabel('Velocity (μM/s)', fontsize=11)
    ax1.set_title(f'Michaelis-Menten Curves at L={L_demo}nm\n(Different α values)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===================
    # Part 2: k_cat vs L for different α
    # ===================
    
    ax2 = axes1[0, 1]
    L_range = np.logspace(0.5, 2.5, 50)  # 3 to 300 nm
    
    for alpha in alpha_values:
        kcat = kcat_rtm(L_range, kcat_0, alpha, L_ref)
        ax2.plot(L_range, kcat, linewidth=2, label=f'α = {alpha}')
    
    ax2.axhline(y=kcat_0, color='gray', linestyle=':', label=f'k_cat,0 = {kcat_0} s⁻¹')
    ax2.axvline(x=L_ref, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax2.set_ylabel('k_cat (s⁻¹)', fontsize=11)
    ax2.set_title('RTM Scaling: k_cat ∝ L^(-α)\nSmaller L → Faster Catalysis', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax2.annotate('Bulk\n(L=100nm)', xy=(100, kcat_0), xytext=(150, kcat_0*2),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    # ===================
    # Part 3: Rate enhancement vs confinement
    # ===================
    
    ax3 = axes1[1, 0]
    
    for alpha in alpha_values:
        enhancement = kcat_rtm(L_range, kcat_0, alpha, L_ref) / kcat_0
        ax3.plot(L_range, enhancement, linewidth=2, label=f'α = {alpha}')
    
    ax3.axhline(y=1, color='gray', linestyle='--', label='No enhancement')
    ax3.fill_between(L_range, 1, 100, where=L_range < L_ref, alpha=0.1, color='green')
    ax3.fill_between(L_range, 0.01, 1, where=L_range > L_ref, alpha=0.1, color='red')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax3.set_ylabel('Rate Enhancement (k_cat/k_cat,0)', fontsize=11)
    ax3.set_title('Catalytic Enhancement by Confinement\n(Green = enhanced, Red = suppressed)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(0.01, 100)
    
    # ===================
    # Part 4: Simulated experimental data with noise
    # ===================
    
    ax4 = axes1[1, 1]
    
    # Generate noisy kinetic data at different L
    alpha_true = 2.2
    noise_level = 0.05  # 5% noise
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    for L, color in zip(L_values, colors):
        # True kinetic curve
        v_true = michaelis_menten_rtm(S, L, kcat_0, Km_0, E_total, alpha_true, L_ref)
        
        # Add noise
        v_noisy = v_true * (1 + noise_level * np.random.randn(len(S)))
        v_noisy = np.maximum(v_noisy, 0)
        
        ax4.scatter(S, v_noisy, s=15, alpha=0.6, color=color)
        ax4.plot(S, v_true, '-', color=color, linewidth=1.5, label=f'L={L}nm')
        
        # Fit to extract apparent k_cat
        fit_result = compute_kcat_apparent(v_noisy, S, E_total)
        if fit_result:
            results.append({
                'L': L,
                'alpha_true': alpha_true,
                'kcat_apparent': fit_result['kcat'],
                'Km_apparent': fit_result['Km']
            })
    
    ax4.set_xscale('log')
    ax4.set_xlabel('[S] (μM)', fontsize=11)
    ax4.set_ylabel('Velocity (μM/s)', fontsize=11)
    ax4.set_title(f'Simulated Experimental Data (α_true = {alpha_true})\nConfinement Effect on Kinetics', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S1_michaelis_menten.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S1_michaelis_menten.pdf'))
    plt.close()
    
    # ===================
    # Part 5: Recovery of α from simulated data
    # ===================
    
    print("\n2. Recovering α from k_cat vs L data...")
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) >= 3:
        # Fit log(k_cat) vs log(L)
        log_L = np.log(df_results['L'].values)
        log_kcat = np.log(df_results['kcat_apparent'].values)
        
        slope, intercept = np.polyfit(log_L, log_kcat, 1)
        alpha_recovered = -slope
        
        print(f"\n  True α: {alpha_true}")
        print(f"  Recovered α: {alpha_recovered:.3f}")
        print(f"  Error: {abs(alpha_recovered - alpha_true):.3f}")
        
        # Recovery plot
        fig2, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(df_results['L'], df_results['kcat_apparent'], s=80, 
                   c='blue', zorder=3, label='Fitted k_cat')
        
        L_fit = np.logspace(np.log10(L_values.min()), np.log10(L_values.max()), 50)
        kcat_fit = np.exp(intercept) * L_fit ** slope
        ax.plot(L_fit, kcat_fit, 'r--', linewidth=2, 
                label=f'Fit: α = {alpha_recovered:.2f}')
        
        kcat_true = kcat_rtm(L_fit, kcat_0, alpha_true, L_ref)
        ax.plot(L_fit, kcat_true, 'g:', linewidth=2,
                label=f'True: α = {alpha_true}')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Confinement Length L (nm)', fontsize=12)
        ax.set_ylabel('Apparent k_cat (s⁻¹)', fontsize=12)
        ax.set_title(f'α Recovery from Kinetic Data\nTrue α = {alpha_true}, Recovered α = {alpha_recovered:.2f}',
                     fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'S1_alpha_recovery.png'), dpi=150)
        plt.close()
        
        df_results['alpha_true'] = alpha_true
        df_results.to_csv(os.path.join(output_dir, 'S1_kinetic_data.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S1: RTM-Modified Michaelis-Menten Kinetics
==========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM MODIFICATION TO ENZYME KINETICS
-----------------------------------
Classic: v = V_max * [S] / (K_m + [S])

RTM adds: k_cat(L) = k_cat,0 * (L/L_ref)^(-α)

where:
- L = confinement length (nm)
- α = coherence exponent
- Higher α = stronger confinement effect

TRANSPORT CLASS PREDICTIONS
---------------------------
α ≈ 1.5: Quasi-ballistic/guided transport
α ≈ 2.0: Laplacian diffusion
α ≈ 2.2-2.5: Hierarchical/fractal transport
α > 2.5: Highly coherent (conjectural)

SIMULATION PARAMETERS
---------------------
k_cat,0 = {kcat_0} s⁻¹ (at L_ref)
K_m,0 = {Km_0} μM
[E]_total = {E_total} μM
L_ref = {L_ref} nm

RECOVERY TEST
-------------
True α = {alpha_true}
Recovered α = {alpha_recovered:.3f}
Error = {abs(alpha_recovered - alpha_true):.3f}

KEY PREDICTIONS
---------------
1. Smaller confinement (L↓) → faster catalysis (k_cat↑)
2. Effect scales as L^(-α)
3. α encodes transport class, not enzyme identity
4. Allosteric effectors should modulate α

EXPERIMENTAL VALIDATION
-----------------------
Measure k_cat at multiple confinement scales:
- Nanoporous matrices (5-200nm pores)
- Polymer crowding (PEG, dextran)
- Engineered cavities

Plot log(k_cat) vs log(L):
- Slope = -α
- Stability over 1+ decade in L = validation
"""
    
    with open(os.path.join(output_dir, 'S1_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nα recovery error: {abs(alpha_recovered - alpha_true):.3f}")
    print(f"\nOutputs saved to: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
