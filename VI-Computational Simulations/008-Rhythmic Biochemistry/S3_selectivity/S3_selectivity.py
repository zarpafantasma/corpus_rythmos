#!/usr/bin/env python3
"""
S3: Substrate Selectivity Prediction by α
==========================================

RTM Prediction: If two substrates have different effective α values,
confinement can SWITCH enzyme selectivity.

Key insight:
- Substrate A: k_A ∝ L^(-α_A)
- Substrate B: k_B ∝ L^(-α_B)
- Selectivity: S = k_A/k_B ∝ L^(α_B - α_A)

If α_A > α_B:
- Large L (bulk): substrate with higher k dominates
- Small L (confined): substrate with higher α gets MORE enhanced

This allows TUNABLE SELECTIVITY via confinement geometry.

THEORETICAL MODEL - experimentally testable
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM SELECTIVITY MODEL
# =============================================================================

def k_substrate(L, k_0, alpha, L_ref=100):
    """Rate constant for a substrate at confinement L."""
    return k_0 * (L / L_ref) ** (-alpha)


def selectivity(L, k_A_0, alpha_A, k_B_0, alpha_B, L_ref=100):
    """
    Selectivity ratio S = k_A / k_B
    
    S(L) = (k_A_0 / k_B_0) * (L/L_ref)^(α_B - α_A)
    """
    k_A = k_substrate(L, k_A_0, alpha_A, L_ref)
    k_B = k_substrate(L, k_B_0, alpha_B, L_ref)
    return k_A / k_B


def crossover_length(k_A_0, alpha_A, k_B_0, alpha_B, L_ref=100):
    """
    Find L where selectivity = 1 (crossover point).
    
    k_A(L*) = k_B(L*)
    k_A_0 * (L*/L_ref)^(-α_A) = k_B_0 * (L*/L_ref)^(-α_B)
    (L*/L_ref)^(α_A - α_B) = k_A_0 / k_B_0
    L* = L_ref * (k_A_0 / k_B_0)^(1/(α_A - α_B))
    """
    if alpha_A == alpha_B:
        return np.nan  # No crossover if same α
    
    ratio = k_A_0 / k_B_0
    exponent = 1 / (alpha_A - alpha_B)
    
    L_star = L_ref * (ratio ** exponent)
    
    return L_star


def compute_rbci(alpha, alpha_ref=2.0, ciss=0.5, vibrational_coherence=0.5, 
                 variance_reduction=0.5):
    """
    Compute Rhythmic Biochemistry Coherence Index (RBCI).
    
    RBCI = (w1*α_norm + w2*CISS + w3*Vib + w4*VR) / sum(weights)
    
    Components (0-1 normalized):
    - α_norm: how far above diffusive baseline
    - CISS: spin polarization
    - Vibrational coherence: fraction of coherent modes
    - Variance reduction: under on-resonance driving
    """
    # Weights
    w = {'alpha': 0.3, 'ciss': 0.25, 'vib': 0.25, 'var': 0.2}
    
    # α normalization (0 at α=1.5, 1 at α=2.5)
    alpha_norm = np.clip((alpha - 1.5) / 1.0, 0, 1)
    
    # RBCI calculation
    rbci = (w['alpha'] * alpha_norm + 
            w['ciss'] * ciss + 
            w['vib'] * vibrational_coherence + 
            w['var'] * variance_reduction)
    
    return rbci


# =============================================================================
# SIMULATION SCENARIOS
# =============================================================================

def scenario_drug_metabolism():
    """
    Scenario: Cytochrome P450 with two drug substrates
    
    Drug A: Lipophilic, benefits from hierarchical transport (higher α)
    Drug B: Hydrophilic, more diffusive pathway (lower α)
    """
    return {
        'name': 'CYP450 Drug Metabolism',
        'enzyme': 'Cytochrome P450 3A4',
        'substrate_A': 'Lipophilic drug (e.g., midazolam)',
        'substrate_B': 'Hydrophilic drug (e.g., erythromycin)',
        'k_A_0': 80,    # s^-1 in bulk
        'k_B_0': 120,   # s^-1 in bulk (faster in bulk)
        'alpha_A': 2.4,  # Benefits more from confinement
        'alpha_B': 1.9,  # Diffusive
        'interpretation': """
In bulk (L~100nm): Drug B metabolized faster (k_B_0 > k_A_0)
Under confinement (L~20nm): Drug A catches up due to higher α
At crossover: selectivity switches

Clinical implication: Membrane-bound P450 in ER shows different
selectivity than reconstituted enzyme in solution.
"""
    }


def scenario_industrial_biocatalysis():
    """
    Scenario: Lipase selectivity between enantiomers
    
    Confinement can enhance enantioselectivity if
    the preferred enantiomer has higher α.
    """
    return {
        'name': 'Lipase Enantioselectivity',
        'enzyme': 'Candida antarctica lipase B (CALB)',
        'substrate_A': '(R)-enantiomer',
        'substrate_B': '(S)-enantiomer',
        'k_A_0': 50,    # s^-1
        'k_B_0': 45,    # s^-1 (slightly slower in bulk)
        'alpha_A': 2.3,  # Hierarchical binding
        'alpha_B': 2.1,  # Also hierarchical but less
        'interpretation': """
E-ratio (enantioselectivity) = k_R / k_S

In bulk: E = 50/45 = 1.1 (poor selectivity)
At L=20nm: E increases due to differential α
Immobilization in nanopores can enhance enantioselectivity.

Industrial application: Tune support pore size to maximize E.
"""
    }


def scenario_allosteric_regulation():
    """
    Scenario: Allosteric enzyme with substrate-competitor pair
    
    Activator binding changes the enzyme's α.
    """
    return {
        'name': 'Allosteric Regulation',
        'enzyme': 'Phosphofructokinase (PFK)',
        'substrate_A': 'Fructose-6-phosphate (substrate)',
        'substrate_B': 'ATP (inhibitor at high conc)',
        'k_A_0': 100,   # s^-1
        'k_B_0': 60,    # s^-1
        'alpha_A': 2.2,  # Normal
        'alpha_B': 2.5,  # Inhibitor mode (higher α when enzyme is tense)
        'interpretation': """
RTM predicts: allosteric activators raise enzyme's effective α
while inhibitors may lower it (or raise α for inhibitor binding).

In cellular environment (crowded, L_eff ~30nm), the balance
between substrate and inhibitor depends on confinement.

Metabolic consequence: Glycolytic flux is confinement-dependent.
"""
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Substrate Selectivity Prediction by α")
    print("=" * 70)
    
    output_dir = "/home/claude/012-Rhythmic_Biochemistry/S3_selectivity/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    L_ref = 100  # nm
    L_range = np.logspace(0.7, 2.5, 100)  # 5 to 300 nm
    
    # Get scenarios
    scenarios = [
        scenario_drug_metabolism(),
        scenario_industrial_biocatalysis(),
        scenario_allosteric_regulation()
    ]
    
    results = []
    
    # ===================
    # Main figure
    # ===================
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, scenario in enumerate(scenarios):
        print(f"\n{idx+1}. {scenario['name']}...")
        
        k_A_0 = scenario['k_A_0']
        k_B_0 = scenario['k_B_0']
        alpha_A = scenario['alpha_A']
        alpha_B = scenario['alpha_B']
        
        # Compute rate constants
        k_A = k_substrate(L_range, k_A_0, alpha_A, L_ref)
        k_B = k_substrate(L_range, k_B_0, alpha_B, L_ref)
        
        # Compute selectivity
        S = selectivity(L_range, k_A_0, alpha_A, k_B_0, alpha_B, L_ref)
        
        # Find crossover
        L_cross = crossover_length(k_A_0, alpha_A, k_B_0, alpha_B, L_ref)
        
        # Store results
        results.append({
            'scenario': scenario['name'],
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'delta_alpha': alpha_A - alpha_B,
            'k_A_0': k_A_0,
            'k_B_0': k_B_0,
            'S_bulk': k_A_0 / k_B_0,
            'S_20nm': selectivity(20, k_A_0, alpha_A, k_B_0, alpha_B, L_ref),
            'L_crossover': L_cross
        })
        
        # Plot rate constants
        ax1 = axes[0, idx]
        ax1.plot(L_range, k_A, 'b-', linewidth=2, label=f'A: α={alpha_A}')
        ax1.plot(L_range, k_B, 'r-', linewidth=2, label=f'B: α={alpha_B}')
        
        if not np.isnan(L_cross) and 5 < L_cross < 300:
            ax1.axvline(x=L_cross, color='gray', linestyle='--', alpha=0.7)
            ax1.annotate(f'Crossover\nL={L_cross:.0f}nm', xy=(L_cross, k_A_0*0.5),
                        fontsize=9, ha='center')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Confinement L (nm)', fontsize=10)
        ax1.set_ylabel('k (s⁻¹)', fontsize=10)
        ax1.set_title(f'{scenario["name"]}\nRate Constants', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot selectivity
        ax2 = axes[1, idx]
        ax2.plot(L_range, S, 'g-', linewidth=2)
        ax2.axhline(y=1, color='gray', linestyle='--', label='S=1 (no preference)')
        
        if not np.isnan(L_cross) and 5 < L_cross < 300:
            ax2.axvline(x=L_cross, color='gray', linestyle='--', alpha=0.7)
        
        # Shade regions
        ax2.fill_between(L_range, 0, S, where=S > 1, alpha=0.2, color='blue',
                         label='A preferred')
        ax2.fill_between(L_range, S, 10, where=S < 1, alpha=0.2, color='red',
                         label='B preferred')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Confinement L (nm)', fontsize=10)
        ax2.set_ylabel('Selectivity S = k_A/k_B', fontsize=10)
        ax2.set_title(f'Selectivity vs Confinement\nΔα = {alpha_A - alpha_B:.1f}', fontsize=11)
        ax2.set_ylim(0.1, 10)
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_selectivity.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_selectivity.pdf'))
    plt.close()
    
    # ===================
    # RBCI demonstration
    # ===================
    
    print("\n4. Computing RBCI across scenarios...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # RBCI components
    ax = axes2[0]
    
    alphas = np.linspace(1.5, 2.8, 50)
    rbci_base = [compute_rbci(a, ciss=0.3, vibrational_coherence=0.4, 
                              variance_reduction=0.3) for a in alphas]
    rbci_high = [compute_rbci(a, ciss=0.8, vibrational_coherence=0.7, 
                              variance_reduction=0.7) for a in alphas]
    
    ax.plot(alphas, rbci_base, 'b-', linewidth=2, label='Low coherence signals')
    ax.plot(alphas, rbci_high, 'g-', linewidth=2, label='High coherence signals')
    ax.fill_between(alphas, rbci_base, rbci_high, alpha=0.2, color='gray')
    
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_ylabel('RBCI (0-1)', fontsize=11)
    ax.set_title('Rhythmic Biochemistry Coherence Index\nvs α and Coherence Markers', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 2.8)
    ax.set_ylim(0, 1)
    
    # Selectivity tuning range
    ax = axes2[1]
    
    delta_alphas = np.linspace(-0.5, 0.5, 50)
    L_test = [10, 20, 50, 100]
    
    for L in L_test:
        S_values = (L / L_ref) ** (-delta_alphas)
        ax.plot(delta_alphas, S_values, linewidth=2, label=f'L={L}nm')
    
    ax.axhline(y=1, color='gray', linestyle='--')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Δα = α_A - α_B', fontsize=11)
    ax.set_ylabel('Selectivity Enhancement S/S_bulk', fontsize=11)
    ax.set_title('Selectivity Tuning by Confinement\n(Δα > 0: A benefits more from confinement)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_rbci_and_tuning.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'S3_selectivity_results.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S3: Substrate Selectivity Prediction by α
==========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM SELECTIVITY PREDICTION
--------------------------
If substrates A and B have different α values:

k_A(L) = k_A,0 × (L/L_ref)^(-α_A)
k_B(L) = k_B,0 × (L/L_ref)^(-α_B)

Selectivity: S = k_A/k_B ∝ L^(α_B - α_A)

KEY INSIGHT:
If Δα = α_A - α_B > 0:
  - Substrate A benefits MORE from confinement
  - Small L enhances selectivity for A
  - Crossover possible if k_B_0 > k_A_0 in bulk

SCENARIO RESULTS
----------------
"""
    
    for _, row in df_results.iterrows():
        summary += f"""
{row['scenario']}
  α_A = {row['alpha_A']}, α_B = {row['alpha_B']}
  Δα = {row['delta_alpha']:.2f}
  S_bulk = {row['S_bulk']:.2f}
  S(20nm) = {row['S_20nm']:.2f}
  Crossover L = {row['L_crossover']:.0f} nm
"""
    
    summary += f"""
APPLICATIONS
------------
1. DRUG METABOLISM
   - P450 isoforms show differential confinement effects
   - Membrane environment affects selectivity
   - Tunable by lipid composition (effective L)

2. INDUSTRIAL BIOCATALYSIS
   - Immobilization pore size affects enantioselectivity
   - Design support matrix for optimal E-ratio
   - Predict optimal confinement from α measurements

3. METABOLIC REGULATION
   - Crowded cytoplasm (L_eff ~10-30nm)
   - Allosteric regulation couples to α changes
   - Compartmentalization as selectivity tool

EXPERIMENTAL VALIDATION
-----------------------
1. Measure k_cat for A and B at multiple L values
2. Fit α_A and α_B independently
3. Predict crossover from fitted parameters
4. Test prediction at L_crossover

RBCI AS QUALITY INDEX
---------------------
RBCI aggregates:
- α normalization (30%)
- CISS spin signature (25%)
- Vibrational coherence (25%)
- Variance reduction under driving (20%)

High RBCI (>0.6): robust RTM scaling expected
Low RBCI (<0.3): deviations from RTM likely
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for _, row in df_results.iterrows():
        print(f"\n{row['scenario']}:")
        print(f"  Δα = {row['delta_alpha']:.2f}")
        print(f"  S_bulk = {row['S_bulk']:.2f} → S(20nm) = {row['S_20nm']:.2f}")
        if not np.isnan(row['L_crossover']):
            print(f"  Crossover at L = {row['L_crossover']:.0f} nm")
    
    print(f"\nOutputs: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
