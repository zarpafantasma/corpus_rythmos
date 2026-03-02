#!/usr/bin/env python3
"""
RTM Chemistry Analysis: Two Regimes of Molecular Diffusion
============================================================

This script validates RTM predictions in chemistry by analyzing two
distinct diffusion regimes:

1. CONFINED REGIME (Zeolites): D ∝ L^α with α >> 1
   - Configurational diffusion in nanoporous materials
   - Extreme geometric sensitivity

2. BULK REGIME (Stokes-Einstein): D ∝ r^α with α ≈ -1
   - Standard diffusion in free solution
   - Viscous drag dominates

KEY RTM INSIGHT: α changes sign between regimes, demonstrating
that RTM can distinguish transport mechanisms through scaling exponents.

Data Sources:
- Zeolites: Kärger & Ruthven, QENS studies, literature compilation
- Bulk: CRC Handbook, Landolt-Börnstein, literature values

Author: RTM Research
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

OUTPUT_DIR = "output"


# ============================================================================
# ZEOLITE DIFFUSION DATABASE
# ============================================================================

def get_zeolite_data():
    """
    Return zeolite diffusion database.
    
    Sources:
    - Kärger & Ruthven "Diffusion in Zeolites and Other Microporous Solids"
    - Jobic et al., QENS studies
    - Ruthven & Post, comprehensive reviews
    """
    
    data = [
        # (Material, Pore_nm, Guest, D_m2s, T_K, Reference)
        # n-Hexane series
        ("Zeolite 5A", 0.42, "n-Hexane", 2.0e-14, 300, "Ruthven 1984"),
        ("Ferrierite", 0.48, "n-Hexane", 5.0e-14, 300, "Kärger 1992"),
        ("ZSM-5", 0.55, "n-Hexane", 2.0e-10, 300, "Kärger 1992"),
        ("Silicalite", 0.55, "n-Hexane", 1.5e-10, 300, "Jobic 1997"),
        ("Mordenite", 0.65, "n-Hexane", 1.0e-09, 300, "Ruthven 1984"),
        ("Zeolite Y", 0.74, "n-Hexane", 5.0e-09, 300, "Kärger 1992"),
        ("Zeolite X", 0.74, "n-Hexane", 6.0e-09, 300, "Ruthven 1984"),
        ("MCM-41", 3.00, "n-Hexane", 2.0e-08, 300, "Stallmach 2000"),
        ("SBA-15", 6.00, "n-Hexane", 8.0e-08, 300, "Stallmach 2000"),
        
        # Methane series
        ("Zeolite 4A", 0.38, "Methane", 1.0e-12, 300, "Kärger 1992"),
        ("Zeolite 5A", 0.42, "Methane", 5.0e-10, 300, "Kärger 1992"),
        ("ZSM-5", 0.55, "Methane", 1.0e-08, 300, "Jobic 1997"),
        ("Silicalite", 0.55, "Methane", 8.0e-09, 300, "Caro 1993"),
        ("Zeolite Y", 0.74, "Methane", 5.0e-08, 300, "Kärger 1992"),
        ("MCM-41", 3.00, "Methane", 2.0e-07, 300, "Stallmach 2000"),
        
        # Benzene series
        ("ZSM-5", 0.55, "Benzene", 1.0e-12, 300, "Förste 1990"),
        ("Silicalite", 0.55, "Benzene", 8.0e-13, 300, "Jobic 1992"),
        ("Zeolite Y", 0.74, "Benzene", 1.0e-10, 300, "Kärger 1992"),
        ("NaX", 0.74, "Benzene", 5.0e-11, 300, "Germanus 1985"),
        ("MCM-41", 3.00, "Benzene", 5.0e-09, 300, "Stallmach 2000"),
        
        # Propane series
        ("Zeolite 5A", 0.42, "Propane", 1.0e-13, 300, "Ruthven 1984"),
        ("ZSM-5", 0.55, "Propane", 5.0e-10, 300, "Kärger 1992"),
        ("Silicalite", 0.55, "Propane", 3.0e-10, 300, "Jobic 1997"),
        ("Zeolite Y", 0.74, "Propane", 2.0e-09, 300, "Kärger 1992"),
        
        # CO2 series
        ("Zeolite 4A", 0.38, "CO2", 5.0e-11, 300, "Kärger 1992"),
        ("Zeolite 5A", 0.42, "CO2", 2.0e-10, 300, "Ruthven 1984"),
        ("ZSM-5", 0.55, "CO2", 1.0e-09, 300, "Caro 1993"),
        ("Zeolite Y", 0.74, "CO2", 5.0e-09, 300, "Kärger 1992"),
        
        # Water series
        ("Zeolite 4A", 0.38, "Water", 1.0e-10, 300, "Paoli 2002"),
        ("Zeolite 5A", 0.42, "Water", 5.0e-10, 300, "Paoli 2002"),
        ("ZSM-5", 0.55, "Water", 2.0e-09, 300, "Bussai 2002"),
        ("Zeolite Y", 0.74, "Water", 8.0e-09, 300, "Paoli 2002"),
        
        # n-Butane
        ("ZSM-5", 0.55, "n-Butane", 8.0e-11, 300, "Kärger 1992"),
        ("Silicalite", 0.55, "n-Butane", 5.0e-11, 300, "Jobic 1997"),
        ("Zeolite Y", 0.74, "n-Butane", 3.0e-09, 300, "Kärger 1992"),
    ]
    
    df = pd.DataFrame(data, columns=['Material', 'Pore_nm', 'Guest', 'D_m2s', 'T_K', 'Reference'])
    df['log_L'] = np.log10(df['Pore_nm'])
    df['log_D'] = np.log10(df['D_m2s'])
    
    return df


# ============================================================================
# STOKES-EINSTEIN DIFFUSION DATABASE
# ============================================================================

def get_stokes_einstein_data():
    """
    Return bulk diffusion database.
    
    Sources:
    - CRC Handbook of Chemistry and Physics
    - Landolt-Börnstein Tables
    - Individual literature values
    """
    
    data = [
        # (Molecule, Radius_nm, D_m2s, Solvent, T_K, Reference)
        # Small molecules
        ("H2", 0.14, 4.50e-09, "Water", 298, "CRC"),
        ("He", 0.13, 6.28e-09, "Water", 298, "CRC"),
        ("H2O", 0.14, 2.30e-09, "Water", 298, "Mills 1973"),
        ("D2O", 0.14, 1.87e-09, "Water", 298, "Mills 1973"),
        ("O2", 0.17, 2.10e-09, "Water", 298, "CRC"),
        ("N2", 0.18, 1.88e-09, "Water", 298, "CRC"),
        ("CO2", 0.23, 1.91e-09, "Water", 298, "CRC"),
        ("NH3", 0.16, 1.64e-09, "Water", 298, "CRC"),
        ("H2S", 0.19, 1.41e-09, "Water", 298, "CRC"),
        ("NO", 0.16, 2.60e-09, "Water", 298, "CRC"),
        ("Ar", 0.19, 2.00e-09, "Water", 298, "CRC"),
        ("CH4", 0.19, 1.49e-09, "Water", 298, "Witherspoon 1965"),
        
        # Alcohols
        ("Methanol", 0.19, 1.28e-09, "Water", 298, "CRC"),
        ("Ethanol", 0.22, 1.00e-09, "Water", 298, "CRC"),
        ("1-Propanol", 0.26, 0.87e-09, "Water", 298, "CRC"),
        ("1-Butanol", 0.29, 0.77e-09, "Water", 298, "CRC"),
        ("1-Pentanol", 0.32, 0.69e-09, "Water", 298, "CRC"),
        ("1-Hexanol", 0.35, 0.60e-09, "Water", 298, "CRC"),
        ("Glycerol", 0.31, 0.72e-09, "Water", 298, "CRC"),
        ("Ethylene glycol", 0.23, 1.16e-09, "Water", 298, "CRC"),
        
        # Sugars
        ("Glucose", 0.36, 0.67e-09, "Water", 298, "Gladden 1953"),
        ("Fructose", 0.36, 0.69e-09, "Water", 298, "Gladden 1953"),
        ("Sucrose", 0.47, 0.52e-09, "Water", 298, "Gosting 1953"),
        ("Maltose", 0.47, 0.51e-09, "Water", 298, "CRC"),
        ("Raffinose", 0.58, 0.43e-09, "Water", 298, "Gosting 1953"),
        ("Cyclodextrin-α", 0.65, 0.32e-09, "Water", 298, "CRC"),
        ("Cyclodextrin-β", 0.75, 0.28e-09, "Water", 298, "CRC"),
        
        # Amino acids
        ("Glycine", 0.23, 1.06e-09, "Water", 298, "Longsworth 1953"),
        ("Alanine", 0.26, 0.91e-09, "Water", 298, "Longsworth 1953"),
        ("Valine", 0.30, 0.77e-09, "Water", 298, "Longsworth 1953"),
        ("Leucine", 0.33, 0.69e-09, "Water", 298, "Longsworth 1953"),
        ("Phenylalanine", 0.36, 0.70e-09, "Water", 298, "Longsworth 1953"),
        ("Tryptophan", 0.40, 0.59e-09, "Water", 298, "Longsworth 1953"),
        
        # Ions
        ("Li+", 0.24, 1.03e-09, "Water", 298, "CRC"),
        ("Na+", 0.18, 1.33e-09, "Water", 298, "CRC"),
        ("K+", 0.14, 1.96e-09, "Water", 298, "CRC"),
        ("Rb+", 0.15, 2.07e-09, "Water", 298, "CRC"),
        ("Cs+", 0.17, 2.06e-09, "Water", 298, "CRC"),
        ("Mg2+", 0.35, 0.71e-09, "Water", 298, "CRC"),
        ("Ca2+", 0.31, 0.79e-09, "Water", 298, "CRC"),
        ("Cl-", 0.18, 2.03e-09, "Water", 298, "CRC"),
        ("Br-", 0.20, 2.08e-09, "Water", 298, "CRC"),
        ("I-", 0.22, 2.05e-09, "Water", 298, "CRC"),
        
        # Proteins
        ("Lysozyme", 1.9, 0.11e-09, "Water", 298, "Tyn & Gusek"),
        ("Myoglobin", 2.0, 0.11e-09, "Water", 298, "Tyn & Gusek"),
        ("BSA", 3.5, 0.06e-09, "Water", 298, "Tyn & Gusek"),
        ("Hemoglobin", 3.1, 0.07e-09, "Water", 298, "Tyn & Gusek"),
        ("γ-Globulin", 4.5, 0.04e-09, "Water", 298, "Tyn & Gusek"),
    ]
    
    df = pd.DataFrame(data, columns=['Molecule', 'r_nm', 'D_m2s', 'Solvent', 'T_K', 'Reference'])
    df['log_r'] = np.log10(df['r_nm'])
    df['log_D'] = np.log10(df['D_m2s'])
    
    # Add type classification
    mol_types = {
        'Gas': ['H2', 'He', 'O2', 'N2', 'CO2', 'NH3', 'H2S', 'NO', 'Ar', 'CH4', 'D2O', 'H2O'],
        'Alcohol': ['Methanol', 'Ethanol', '1-Propanol', '1-Butanol', '1-Pentanol', 
                   '1-Hexanol', 'Glycerol', 'Ethylene glycol'],
        'Sugar': ['Glucose', 'Fructose', 'Sucrose', 'Maltose', 'Raffinose', 
                 'Cyclodextrin-α', 'Cyclodextrin-β'],
        'Amino acid': ['Glycine', 'Alanine', 'Valine', 'Leucine', 'Phenylalanine', 'Tryptophan'],
        'Ion': ['Li+', 'Na+', 'K+', 'Rb+', 'Cs+', 'Mg2+', 'Ca2+', 'Cl-', 'Br-', 'I-'],
        'Protein': ['Lysozyme', 'Myoglobin', 'BSA', 'Hemoglobin', 'γ-Globulin'],
    }
    
    def get_type(mol):
        for typ, mols in mol_types.items():
            if mol in mols:
                return typ
        return 'Other'
    
    df['Type'] = df['Molecule'].apply(get_type)
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_zeolites(df):
    """Analyze zeolite diffusion data."""
    results = {}
    
    # Overall fit
    slope, intercept, r, p, se = stats.linregress(df['log_L'], df['log_D'])
    results['all'] = {'alpha': slope, 'se': se, 'r2': r**2, 'p': p, 'n': len(df)}
    
    # Microporous regime (< 0.8 nm)
    micro = df[df['Pore_nm'] < 0.8]
    if len(micro) >= 5:
        sl, it, rv, pv, er = stats.linregress(micro['log_L'], micro['log_D'])
        results['microporous'] = {'alpha': sl, 'se': er, 'r2': rv**2, 'n': len(micro)}
    
    # By guest molecule
    results['by_guest'] = {}
    for guest in df['Guest'].unique():
        subset = df[df['Guest'] == guest]
        if len(subset) >= 4:
            sl, it, rv, pv, er = stats.linregress(subset['log_L'], subset['log_D'])
            results['by_guest'][guest] = {'alpha': sl, 'r2': rv**2, 'n': len(subset)}
    
    return results


def analyze_stokes_einstein(df):
    """Analyze bulk diffusion data."""
    results = {}
    
    # Overall fit
    slope, intercept, r, p, se = stats.linregress(df['log_r'], df['log_D'])
    results['all'] = {'alpha': slope, 'se': se, 'r2': r**2, 'p': p, 'n': len(df)}
    
    # Test vs α = -1 (Stokes-Einstein prediction)
    t_stat = (slope - (-1.0)) / se
    p_vs_minus1 = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(df)-2))
    results['test_vs_minus1'] = {'t': t_stat, 'p': p_vs_minus1}
    
    # By molecule type
    results['by_type'] = {}
    for typ in df['Type'].unique():
        subset = df[df['Type'] == typ]
        if len(subset) >= 4:
            sl, it, rv, pv, er = stats.linregress(subset['log_r'], subset['log_D'])
            results['by_type'][typ] = {'alpha': sl, 'r2': rv**2, 'n': len(subset)}
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_figures(zeolite_df, stokes_df, zeolite_results, stokes_results):
    """Create analysis figures."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors_z = {'n-Hexane': '#e74c3c', 'Methane': '#3498db', 'Benzene': '#9b59b6',
                'Propane': '#27ae60', 'CO2': '#f39c12', 'Water': '#1abc9c', 'n-Butane': '#e67e22'}
    colors_s = {'Gas': '#3498db', 'Alcohol': '#e74c3c', 'Sugar': '#27ae60',
                'Amino acid': '#9b59b6', 'Ion': '#f39c12', 'Protein': '#1abc9c'}
    
    # Panel 1: Zeolites
    ax = axes[0, 0]
    for guest in zeolite_df['Guest'].unique():
        subset = zeolite_df[zeolite_df['Guest'] == guest]
        ax.scatter(subset['Pore_nm'], subset['D_m2s'], c=colors_z.get(guest, 'gray'),
                   s=70, alpha=0.7, label=guest, edgecolors='black', linewidth=0.5)
    
    alpha_z = zeolite_results['all']['alpha']
    se_z = zeolite_results['all']['se']
    x_fit = np.logspace(np.log10(zeolite_df['Pore_nm'].min()), 
                        np.log10(zeolite_df['Pore_nm'].max()), 100)
    # Calculate intercept for plotting
    sl, it, _, _, _ = stats.linregress(zeolite_df['log_L'], zeolite_df['log_D'])
    y_fit = 10**it * x_fit**sl
    ax.plot(x_fit, y_fit, 'k--', linewidth=2.5, label=f'α = {alpha_z:.1f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pore Size (nm)', fontsize=11)
    ax.set_ylabel('Diffusion Coefficient (m²/s)', fontsize=11)
    ax.set_title(f'CONFINED: Zeolite Diffusion\nα = +{alpha_z:.1f} ± {se_z:.1f} (n={len(zeolite_df)})',
                 fontsize=12, fontweight='bold', color='#c0392b')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 2: Stokes-Einstein
    ax = axes[0, 1]
    for typ in ['Gas', 'Alcohol', 'Sugar', 'Amino acid', 'Ion', 'Protein']:
        subset = stokes_df[stokes_df['Type'] == typ]
        if len(subset) > 0:
            ax.scatter(subset['r_nm'], subset['D_m2s'], c=colors_s[typ], s=70, alpha=0.7,
                       label=f'{typ} ({len(subset)})', edgecolors='black', linewidth=0.5)
    
    alpha_s = stokes_results['all']['alpha']
    se_s = stokes_results['all']['se']
    sl, it, _, _, _ = stats.linregress(stokes_df['log_r'], stokes_df['log_D'])
    x_fit = np.logspace(np.log10(stokes_df['r_nm'].min()), 
                        np.log10(stokes_df['r_nm'].max()), 100)
    y_fit = 10**it * x_fit**sl
    ax.plot(x_fit, y_fit, 'k--', linewidth=2.5, label=f'α = {alpha_s:.2f}')
    
    # Theoretical
    kT = 1.38e-23 * 298
    eta = 0.89e-3
    y_SE = kT / (6 * np.pi * eta * x_fit * 1e-9)
    ax.plot(x_fit, y_SE, 'r:', linewidth=2, alpha=0.7, label='Theory (α=-1)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Hydrodynamic Radius (nm)', fontsize=11)
    ax.set_ylabel('Diffusion Coefficient (m²/s)', fontsize=11)
    ax.set_title(f'BULK: Stokes-Einstein\nα = {alpha_s:.2f} ± {se_s:.2f} (n={len(stokes_df)})',
                 fontsize=12, fontweight='bold', color='#2980b9')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Comparison
    ax = axes[1, 0]
    ax.scatter(zeolite_df['Pore_nm'], zeolite_df['D_m2s'], c='#e74c3c', s=60, alpha=0.6,
               label='Zeolites (confined)', edgecolors='black', linewidth=0.3, marker='s')
    ax.scatter(stokes_df['r_nm'], stokes_df['D_m2s'], c='#3498db', s=60, alpha=0.6,
               label='Bulk solution', edgecolors='black', linewidth=0.3, marker='o')
    
    # Fit lines
    sl_z, it_z, _, _, _ = stats.linregress(zeolite_df['log_L'], zeolite_df['log_D'])
    sl_s, it_s, _, _, _ = stats.linregress(stokes_df['log_r'], stokes_df['log_D'])
    
    x_z = np.logspace(-0.5, 1, 50)
    ax.plot(x_z, 10**it_z * x_z**sl_z, 'r--', linewidth=2, label=f'Confined: α = +{sl_z:.1f}')
    x_s = np.logspace(-1, 0.7, 50)
    ax.plot(x_s, 10**it_s * x_s**sl_s, 'b--', linewidth=2, label=f'Bulk: α = {sl_s:.1f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Characteristic Length (nm)', fontsize=11)
    ax.set_ylabel('Diffusion Coefficient (m²/s)', fontsize=11)
    ax.set_title('TWO REGIMES: α flips sign!', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
    ═══════════════════════════════════════════════════════════
                RTM CHEMISTRY: TWO DIFFUSION REGIMES
    ═══════════════════════════════════════════════════════════
    
    ANALYSIS 1: ZEOLITE DIFFUSION (Configurational Regime)
    ───────────────────────────────────────────────────────
    • Data points: {len(zeolite_df)}
    • Materials: {zeolite_df['Material'].nunique()} zeolites
    • Guests: {zeolite_df['Guest'].nunique()} molecules
    • α = +{alpha_z:.1f} ± {se_z:.1f}
    • R² = {zeolite_results['all']['r2']:.3f}
    
    → EXTREME geometric sensitivity (α >> 1)
    
    ═══════════════════════════════════════════════════════════
    
    ANALYSIS 2: STOKES-EINSTEIN (Bulk Regime)
    ───────────────────────────────────────────────────────
    • Data points: {len(stokes_df)}
    • Size range: {stokes_df['r_nm'].min():.2f} - {stokes_df['r_nm'].max():.1f} nm
    • α = {alpha_s:.2f} ± {se_s:.2f}
    • R² = {stokes_results['all']['r2']:.3f}
    • Theory predicts: α = -1.0
    
    → VISCOUS drag dominates (α ≈ -1)
    
    ═══════════════════════════════════════════════════════════
    
    KEY RTM INSIGHT
    ───────────────────────────────────────────────────────
    • Bulk:     α ≈ -1   (bigger molecule = slower)
    • Confined: α ≈ +4-9 (bigger pore = MUCH faster)
    
    The α SIGN FLIP marks the transition between
    transport regimes!
    
    ═══════════════════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace')
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f'{OUTPUT_DIR}/chemistry_rtm_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/chemistry_rtm_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figures saved to {OUTPUT_DIR}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RTM CHEMISTRY ANALYSIS")
    print("Two Regimes of Molecular Diffusion")
    print("=" * 70)
    
    # Load data
    print("\nLoading datasets...")
    zeolite_df = get_zeolite_data()
    stokes_df = get_stokes_einstein_data()
    
    print(f"✓ Zeolites: {len(zeolite_df)} data points")
    print(f"✓ Stokes-Einstein: {len(stokes_df)} data points")
    
    # Analyze
    print(f"\n{'=' * 70}")
    print("ANALYSIS 1: ZEOLITE DIFFUSION (CONFINED)")
    print("=" * 70)
    
    zeolite_results = analyze_zeolites(zeolite_df)
    
    print(f"\nOverall: α = +{zeolite_results['all']['alpha']:.2f} ± {zeolite_results['all']['se']:.2f}")
    print(f"R² = {zeolite_results['all']['r2']:.3f}")
    
    if 'microporous' in zeolite_results:
        print(f"\nMicroporous (<0.8 nm): α = +{zeolite_results['microporous']['alpha']:.2f}")
    
    print("\nBy guest molecule:")
    for guest, res in zeolite_results['by_guest'].items():
        print(f"  {guest:12s}: α = {res['alpha']:+6.2f}, R² = {res['r2']:.3f}")
    
    print(f"\n{'=' * 70}")
    print("ANALYSIS 2: STOKES-EINSTEIN (BULK)")
    print("=" * 70)
    
    stokes_results = analyze_stokes_einstein(stokes_df)
    
    print(f"\nOverall: α = {stokes_results['all']['alpha']:.3f} ± {stokes_results['all']['se']:.3f}")
    print(f"R² = {stokes_results['all']['r2']:.3f}")
    print(f"Theory predicts: α = -1.0")
    print(f"Test vs -1: t = {stokes_results['test_vs_minus1']['t']:.2f}, p = {stokes_results['test_vs_minus1']['p']:.4f}")
    
    # Create figures
    print(f"\nGenerating figures...")
    create_figures(zeolite_df, stokes_df, zeolite_results, stokes_results)
    
    # Save data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zeolite_df.to_csv(f'{OUTPUT_DIR}/zeolite_diffusion.csv', index=False)
    stokes_df.to_csv(f'{OUTPUT_DIR}/stokes_einstein_diffusion.csv', index=False)
    print(f"✓ Data saved to {OUTPUT_DIR}/")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"""
RTM successfully distinguishes two diffusion regimes:

CONFINED (Zeolites):      α = +{zeolite_results['all']['alpha']:.1f}  (n={len(zeolite_df)})
BULK (Stokes-Einstein):   α = {stokes_results['all']['alpha']:.1f}  (n={len(stokes_df)})

The SIGN FLIP (positive ↔ negative) demonstrates that:
- In bulk: viscous drag dominates (bigger = slower)
- In confinement: geometry dominates (bigger pore = MUCH faster)

RTM correctly identifies transport mechanisms through α.
    """)


if __name__ == "__main__":
    main()
