#!/usr/bin/env python3
"""
S2: RTM Rate Predictions in Microreactors and Nanoconfinement
=============================================================

RTM predicts rate enhancement in confined geometries:
  k(L) = k_bulk × (L/L_ref)^(-α)

This simulation applies RTM to practical reactor geometries:
1. Microfluidic channels (1-100 μm)
2. Porous catalysts (1-50 nm pores)
3. Sonochemistry (cavitation bubble size)
4. Cavity chemistry (optical/microwave cavities)

We compute:
- Expected rate enhancements
- Optimal confinement for given α
- Throughput vs selectivity tradeoffs

PRACTICAL DESIGN TOOL - requires experimental calibration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM REACTOR MODEL
# =============================================================================

def k_rtm(L, k_bulk, alpha, L_ref=100e-9):
    """
    RTM rate constant at confinement L.
    
    k(L) = k_bulk × (L/L_ref)^(-α)
    """
    return k_bulk * (L / L_ref) ** (-alpha)


def enhancement_factor(L, alpha, L_ref=100e-9):
    """Rate enhancement relative to bulk."""
    return (L / L_ref) ** (-alpha)


def residence_time(L, flow_rate, channel_length):
    """
    Residence time in a microfluidic channel.
    
    Assumes circular cross-section.
    τ_res = V / Q = π(L/2)²·channel_length / flow_rate
    """
    area = np.pi * (L / 2) ** 2
    volume = area * channel_length
    return volume / flow_rate


def conversion(k, tau_res, order=1):
    """
    Conversion for nth order reaction.
    
    First order: X = 1 - exp(-k·τ)
    Second order: X = k·τ·C0 / (1 + k·τ·C0)
    """
    if order == 1:
        return 1 - np.exp(-k * tau_res)
    elif order == 2:
        C0 = 1.0  # normalized
        return k * tau_res * C0 / (1 + k * tau_res * C0)


def throughput(L, flow_rate, channel_length, k_bulk, alpha, L_ref=100e-9):
    """
    Throughput = conversion × flow_rate
    
    Balances enhancement (small L → high k) vs. volume (small L → low flow)
    """
    k = k_rtm(L, k_bulk, alpha, L_ref)
    tau = residence_time(L, flow_rate, channel_length)
    conv = conversion(k, tau)
    
    return conv * flow_rate


# =============================================================================
# REACTOR PLATFORMS
# =============================================================================

def microfluidic_analysis(k_bulk, alpha, L_ref=100e-9):
    """
    Analyze microfluidic reactor performance.
    
    Typical dimensions: 10 μm - 1 mm channel diameter
    """
    L_range = np.logspace(-5, -3, 50)  # 10 μm to 1 mm
    
    # Fixed parameters
    channel_length = 0.1  # 10 cm
    flow_rate = 1e-9  # 1 μL/s
    
    results = []
    
    for L in L_range:
        k = k_rtm(L, k_bulk, alpha, L_ref)
        tau = residence_time(L, flow_rate, channel_length)
        conv = conversion(k, tau)
        enh = enhancement_factor(L, alpha, L_ref)
        
        results.append({
            'L_um': L * 1e6,
            'k': k,
            'tau_s': tau,
            'conversion': conv,
            'enhancement': enh
        })
    
    return pd.DataFrame(results)


def porous_catalyst_analysis(k_bulk, alpha, L_ref=100e-9):
    """
    Analyze reaction in porous catalyst.
    
    Typical pore sizes: 1-50 nm
    """
    L_range = np.logspace(-9, -7.3, 50)  # 1 nm to 50 nm
    
    results = []
    
    for L in L_range:
        k = k_rtm(L, k_bulk, alpha, L_ref)
        enh = enhancement_factor(L, alpha, L_ref)
        
        # Estimate diffusion limitation (Thiele modulus)
        D_eff = 1e-10  # m²/s (typical liquid diffusion / tortuosity)
        phi = L * np.sqrt(k / D_eff)  # Thiele modulus
        eta = np.tanh(phi) / phi if phi > 0.01 else 1.0  # Effectiveness factor
        
        results.append({
            'L_nm': L * 1e9,
            'k': k,
            'enhancement': enh,
            'thiele': phi,
            'effectiveness': eta,
            'k_effective': k * eta
        })
    
    return pd.DataFrame(results)


def sonochemistry_analysis(k_bulk, alpha, L_ref=100e-9):
    """
    Analyze sonochemical reactor (cavitation bubbles).
    
    Typical bubble sizes: 1-100 μm at collapse → 10-1000 nm effective
    """
    # Bubble collapse creates effective nanoscale confinement
    L_effective = np.logspace(-8, -6, 50)  # 10 nm to 1 μm effective
    
    results = []
    
    for L in L_effective:
        k = k_rtm(L, k_bulk, alpha, L_ref)
        enh = enhancement_factor(L, alpha, L_ref)
        
        results.append({
            'L_eff_nm': L * 1e9,
            'k': k,
            'enhancement': enh
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S2: RTM Rate Predictions in Microreactors")
    print("=" * 70)
    
    output_dir = "/home/claude/013-Rhythmic_Chemistry/S2_microreactors/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # Reference parameters
    k_bulk = 0.1  # s^-1 (typical slow reaction in bulk)
    L_ref = 100e-9  # 100 nm
    
    # ===================
    # Part 1: Enhancement across platforms
    # ===================
    
    print("\n1. Computing enhancement factors across platforms...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Microfluidic
    ax = axes1[0, 0]
    
    for alpha in [1.8, 2.0, 2.2, 2.5]:
        df = microfluidic_analysis(k_bulk, alpha, L_ref)
        ax.plot(df['L_um'], df['enhancement'], linewidth=2, label=f'α = {alpha}')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Channel Diameter (μm)', fontsize=11)
    ax.set_ylabel('Rate Enhancement', fontsize=11)
    ax.set_title('Microfluidic Channels\n(10 μm - 1 mm)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(10, 1000)
    
    # Porous catalyst
    ax = axes1[0, 1]
    
    for alpha in [1.8, 2.0, 2.2, 2.5]:
        df = porous_catalyst_analysis(k_bulk, alpha, L_ref)
        ax.plot(df['L_nm'], df['enhancement'], linewidth=2, label=f'α = {alpha}')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pore Diameter (nm)', fontsize=11)
    ax.set_ylabel('Rate Enhancement', fontsize=11)
    ax.set_title('Porous Catalysts / Zeolites\n(1 - 50 nm)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Sonochemistry
    ax = axes1[1, 0]
    
    for alpha in [1.8, 2.0, 2.2, 2.5]:
        df = sonochemistry_analysis(k_bulk, alpha, L_ref)
        ax.plot(df['L_eff_nm'], df['enhancement'], linewidth=2, label=f'α = {alpha}')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Effective Confinement at Collapse (nm)', fontsize=11)
    ax.set_ylabel('Rate Enhancement', fontsize=11)
    ax.set_title('Sonochemistry (Cavitation)\n(10 nm - 1 μm effective)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Summary comparison
    ax = axes1[1, 1]
    
    platforms = {
        'Microfluidic (100 μm)': 100e-6,
        'Microfluidic (10 μm)': 10e-6,
        'Mesoporous (10 nm)': 10e-9,
        'Microporous (2 nm)': 2e-9,
        'Cavitation (50 nm)': 50e-9
    }
    
    alpha_demo = 2.2
    enhancements = [enhancement_factor(L, alpha_demo, L_ref) for L in platforms.values()]
    
    colors = ['lightblue', 'steelblue', 'orange', 'red', 'purple']
    bars = ax.barh(list(platforms.keys()), enhancements, color=colors, alpha=0.7)
    ax.set_xlabel(f'Rate Enhancement (α = {alpha_demo})', fontsize=11)
    ax.set_xscale('log')
    ax.set_title('Enhancement by Platform', fontsize=12)
    ax.axvline(x=1, color='gray', linestyle='--')
    
    for bar, enh in zip(bars, enhancements):
        ax.text(enh * 1.1, bar.get_y() + bar.get_height()/2, 
                f'{enh:.0f}×', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_platform_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_platform_comparison.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Conversion vs confinement tradeoff
    # ===================
    
    print("\n2. Analyzing conversion vs throughput tradeoff...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Porous catalyst with diffusion limitation
    ax = axes2[0]
    
    alpha = 2.2
    df_porous = porous_catalyst_analysis(k_bulk, alpha, L_ref)
    
    ax.plot(df_porous['L_nm'], df_porous['enhancement'], 'b-', linewidth=2, 
            label='Intrinsic enhancement')
    ax.plot(df_porous['L_nm'], df_porous['effectiveness'], 'r--', linewidth=2,
            label='Effectiveness factor')
    ax.plot(df_porous['L_nm'], df_porous['enhancement'] * df_porous['effectiveness'],
            'g-', linewidth=2.5, label='Net enhancement')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pore Diameter (nm)', fontsize=11)
    ax.set_ylabel('Factor', fontsize=11)
    ax.set_title(f'Porous Catalyst: RTM Enhancement vs Diffusion Limitation\n'
                 f'(α = {alpha})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    
    # Optimal pore size
    net_enh = df_porous['enhancement'] * df_porous['effectiveness']
    idx_opt = net_enh.idxmax()
    L_opt = df_porous.loc[idx_opt, 'L_nm']
    enh_opt = net_enh.iloc[idx_opt]
    
    ax.axvline(x=L_opt, color='green', linestyle=':', alpha=0.7)
    ax.annotate(f'Optimal\nL={L_opt:.0f}nm\n{enh_opt:.0f}×', 
                xy=(L_opt, enh_opt), xytext=(L_opt*2, enh_opt*0.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))
    
    # Sensitivity to α
    ax = axes2[1]
    
    L_fixed = 10e-9  # 10 nm
    alpha_range = np.linspace(1.5, 2.8, 50)
    
    enh_intrinsic = [(L_fixed / L_ref) ** (-a) for a in alpha_range]
    
    ax.plot(alpha_range, enh_intrinsic, 'b-', linewidth=2)
    ax.fill_between(alpha_range, 1, enh_intrinsic, alpha=0.2, color='blue')
    
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_ylabel(f'Enhancement at L = {L_fixed*1e9:.0f} nm', fontsize=11)
    ax.set_title('Rate Enhancement Sensitivity to α', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axhline(y=1, color='gray', linestyle='--')
    
    # Mark typical ranges
    ax.axvspan(2.0, 2.1, alpha=0.2, color='yellow', label='Diffusive')
    ax.axvspan(2.1, 2.5, alpha=0.2, color='green', label='Hierarchical')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_optimization.png'), dpi=150)
    plt.close()
    
    # ===================
    # Part 3: Design nomogram
    # ===================
    
    print("\n3. Creating design nomogram...")
    
    fig3, ax = plt.subplots(figsize=(10, 8))
    
    L_range = np.logspace(-9, -5, 100)  # 1 nm to 10 μm
    alpha_range = np.array([1.5, 1.8, 2.0, 2.2, 2.5, 2.8])
    
    for alpha in alpha_range:
        enh = enhancement_factor(L_range, alpha, L_ref)
        ax.plot(L_range * 1e9, enh, linewidth=2, label=f'α = {alpha}')
    
    # Add horizontal lines for target enhancements
    targets = [10, 100, 1000, 10000]
    for t in targets:
        ax.axhline(y=t, color='gray', linestyle=':', alpha=0.5)
        ax.text(1.5, t*1.2, f'{t}×', fontsize=9, color='gray')
    
    # Add platform regions
    ax.axvspan(1, 10, alpha=0.1, color='red', label='Microporous')
    ax.axvspan(10, 50, alpha=0.1, color='orange', label='Mesoporous')
    ax.axvspan(1000, 10000, alpha=0.1, color='blue', label='Microfluidic')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Confinement Length L (nm)', fontsize=12)
    ax.set_ylabel('Rate Enhancement Factor', fontsize=12)
    ax.set_title('RTM Reactor Design Nomogram\nk(L) / k_bulk = (L/100nm)^(-α)', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, 10000)
    ax.set_ylim(0.1, 1e6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_design_nomogram.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_design_nomogram.pdf'))
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_porous.to_csv(os.path.join(output_dir, 'S2_porous_catalyst.csv'), index=False)
    
    # Summary table
    summary_data = []
    for name, L in platforms.items():
        for alpha in [1.8, 2.0, 2.2, 2.5]:
            enh = enhancement_factor(L, alpha, L_ref)
            summary_data.append({
                'platform': name,
                'L_nm': L * 1e9,
                'alpha': alpha,
                'enhancement': enh
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'S2_platform_summary.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S2: RTM Rate Predictions in Microreactors
==========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM REACTOR SCALING
-------------------
k(L) = k_bulk × (L/L_ref)^(-α)

Enhancement = (L/L_ref)^(-α)

PLATFORM COMPARISON (α = 2.2)
-----------------------------
"""
    
    for name, L in platforms.items():
        enh = enhancement_factor(L, 2.2, L_ref)
        summary += f"{name}: {enh:.0f}× enhancement\n"
    
    summary += f"""
OPTIMAL DESIGN (Porous Catalyst, α = {alpha})
---------------------------------------------
Pore size: {L_opt:.0f} nm
Net enhancement: {enh_opt:.0f}×
(Balances RTM enhancement vs diffusion limitation)

DESIGN GUIDELINES
-----------------
1. MICROFLUIDIC (10-1000 μm)
   - Modest enhancement (1-100×)
   - No diffusion limitation
   - Good for flow chemistry

2. MESOPOROUS (5-50 nm)
   - High enhancement (100-10000×)
   - Some diffusion effects
   - Zeolites, silica, MOFs

3. MICROPOROUS (<5 nm)
   - Very high intrinsic enhancement
   - Strong diffusion limitation
   - Net benefit depends on α

4. SONOCHEMISTRY
   - Transient confinement at collapse
   - Effective L depends on bubble dynamics
   - High α environments possible

PRACTICAL RECOMMENDATIONS
-------------------------
1. Estimate α from calibration experiments
2. Use nomogram to select L for target enhancement
3. Account for mass transfer (Thiele modulus)
4. Validate with actual rate measurements
"""
    
    with open(os.path.join(output_dir, 'S2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\nEnhancement factors (α = 2.2):")
    for name, L in platforms.items():
        enh = enhancement_factor(L, 2.2, L_ref)
        print(f"  {name}: {enh:.0f}×")
    
    print(f"\nOptimal pore size (porous catalyst): {L_opt:.0f} nm")
    print(f"\nOutputs: {output_dir}/")
    
    return df_summary


if __name__ == "__main__":
    main()
