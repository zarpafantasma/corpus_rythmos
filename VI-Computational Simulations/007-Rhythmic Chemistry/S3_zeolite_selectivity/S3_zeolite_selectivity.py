#!/usr/bin/env python3
"""
S3: RTM Selectivity Prediction in Zeolites and MOFs
====================================================

RTM predicts that confinement affects competing reactions differently
based on their α values:

For reactions A and B with rates:
  k_A(L) = k_A,bulk × (L/L_ref)^(-α_A)
  k_B(L) = k_B,bulk × (L/L_ref)^(-α_B)

Selectivity: S = k_A/k_B ∝ L^(α_B - α_A)

Applications:
1. Zeolite shape selectivity (pore size effects)
2. MOF catalysis (tunable cavity sizes)
3. Para vs ortho selectivity in aromatic chemistry
4. Endo vs exo in Diels-Alder reactions

THEORETICAL PREDICTIONS - experimentally testable
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM SELECTIVITY MODEL
# =============================================================================

def k_rtm(L, k_bulk, alpha, L_ref=100e-9):
    """RTM rate constant at confinement L."""
    return k_bulk * (L / L_ref) ** (-alpha)


def selectivity(L, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref=100e-9):
    """
    Selectivity ratio S = k_A / k_B
    
    S(L) = (k_A_bulk / k_B_bulk) × (L/L_ref)^(α_B - α_A)
    """
    k_A = k_rtm(L, k_A_bulk, alpha_A, L_ref)
    k_B = k_rtm(L, k_B_bulk, alpha_B, L_ref)
    return k_A / k_B


def crossover_length(k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref=100e-9):
    """Find L where selectivity = 1."""
    if alpha_A == alpha_B:
        return np.nan
    
    ratio = k_A_bulk / k_B_bulk
    L_star = L_ref * (ratio ** (1 / (alpha_A - alpha_B)))
    return L_star


# =============================================================================
# ZEOLITE/MOF DATABASE
# =============================================================================

MATERIALS = {
    # Zeolites
    'ZSM-5': {'pore_nm': 0.55, 'type': 'zeolite', 'topology': 'MFI'},
    'Mordenite': {'pore_nm': 0.70, 'type': 'zeolite', 'topology': 'MOR'},
    'Beta': {'pore_nm': 0.76, 'type': 'zeolite', 'topology': 'BEA'},
    'Y (Faujasite)': {'pore_nm': 0.74, 'type': 'zeolite', 'topology': 'FAU'},
    'MCM-41': {'pore_nm': 3.0, 'type': 'mesoporous', 'topology': 'hexagonal'},
    'SBA-15': {'pore_nm': 8.0, 'type': 'mesoporous', 'topology': 'hexagonal'},
    
    # MOFs
    'MOF-5': {'pore_nm': 1.5, 'type': 'MOF', 'topology': 'cubic'},
    'HKUST-1': {'pore_nm': 0.9, 'type': 'MOF', 'topology': 'paddle-wheel'},
    'ZIF-8': {'pore_nm': 1.16, 'type': 'MOF', 'topology': 'sodalite'},
    'UiO-66': {'pore_nm': 0.75, 'type': 'MOF', 'topology': 'fcu'},
    'MIL-101': {'pore_nm': 3.4, 'type': 'MOF', 'topology': 'MTN'},
}


# =============================================================================
# REACTION SCENARIOS
# =============================================================================

def scenario_xylene_isomerization():
    """
    Xylene isomerization in zeolites.
    
    para-xylene (linear) vs ortho/meta-xylene (bulkier)
    Para benefits more from confinement due to better fit.
    """
    return {
        'name': 'Xylene Isomerization (ZSM-5)',
        'product_A': 'para-xylene',
        'product_B': 'ortho-xylene',
        'k_A_bulk': 1.0,    # Relative
        'k_B_bulk': 1.2,    # Faster in bulk (kinetic product)
        'alpha_A': 2.4,      # Benefits from tight pores
        'alpha_B': 2.0,      # Less affected (diffusion limited)
        'interpretation': """
In bulk: ortho-xylene slightly favored kinetically.
In ZSM-5 (0.55nm): para-xylene strongly favored due to shape selectivity.
RTM explanation: para has higher α (benefits more from confinement).
Industrial use: para-xylene for PET production.
"""
    }


def scenario_diels_alder():
    """
    Diels-Alder reaction: endo vs exo selectivity.
    
    Endo: more compact transition state
    Exo: extended transition state
    """
    return {
        'name': 'Diels-Alder (Endo vs Exo)',
        'product_A': 'endo adduct',
        'product_B': 'exo adduct',
        'k_A_bulk': 0.8,    # Endo slower in bulk
        'k_B_bulk': 1.0,    # Exo kinetically favored
        'alpha_A': 2.5,      # Compact TS benefits from confinement
        'alpha_B': 2.1,      # Extended TS less affected
        'interpretation': """
In bulk: exo product often kinetically favored.
In confined cavity: endo becomes favored (compact TS).
RTM predicts selectivity inversion at calculable pore size.
Applications: MOF catalysis for stereoselective synthesis.
"""
    }


def scenario_alkane_cracking():
    """
    n-alkane vs iso-alkane cracking.
    
    Linear chains fit better in narrow pores.
    """
    return {
        'name': 'Alkane Cracking',
        'product_A': 'n-alkane cracking',
        'product_B': 'iso-alkane cracking',
        'k_A_bulk': 1.0,
        'k_B_bulk': 1.5,    # Branched slightly faster (weaker C-C)
        'alpha_A': 2.3,      # Linear fits better
        'alpha_B': 1.9,      # Branched excluded
        'interpretation': """
ZSM-5 selectively cracks linear alkanes (shape selectivity).
Branched alkanes excluded or react at pore mouths only.
RTM: different α reflects different confinement response.
"""
    }


def scenario_co2_hydrogenation():
    """
    CO2 hydrogenation: methanol vs methane.
    
    Different intermediates have different confinement responses.
    """
    return {
        'name': 'CO2 Hydrogenation',
        'product_A': 'methanol',
        'product_B': 'methane',
        'k_A_bulk': 0.5,    # Methanol slower
        'k_B_bulk': 1.0,    # Methane faster (full reduction)
        'alpha_A': 2.6,      # Benefits from controlled environment
        'alpha_B': 2.2,      # Less selective pathway
        'interpretation': """
In bulk: methane dominates (thermodynamic product).
In confined catalyst: methanol selectivity can be enhanced.
RTM: methanol pathway has higher α (coherent environment helps).
"""
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    print("=" * 70)
    print("S3: RTM Selectivity in Zeolites and MOFs")
    print("=" * 70)
    
    output_dir = "/home/claude/013-Rhythmic_Chemistry/S3_zeolite_selectivity/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    L_ref = 100e-9  # 100 nm reference
    
    # Get scenarios
    scenarios = [
        scenario_xylene_isomerization(),
        scenario_diels_alder(),
        scenario_alkane_cracking(),
        scenario_co2_hydrogenation()
    ]
    
    results = []
    
    # ===================
    # Part 1: Selectivity vs pore size
    # ===================
    
    print("\n1. Computing selectivity across pore sizes...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    L_range = np.logspace(-9.5, -7, 100)  # 0.3 to 100 nm
    
    for idx, scenario in enumerate(scenarios):
        ax = axes1[idx // 2, idx % 2]
        
        k_A_bulk = scenario['k_A_bulk']
        k_B_bulk = scenario['k_B_bulk']
        alpha_A = scenario['alpha_A']
        alpha_B = scenario['alpha_B']
        
        # Compute selectivity
        S = selectivity(L_range, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
        S_bulk = k_A_bulk / k_B_bulk
        
        ax.plot(L_range * 1e9, S, 'b-', linewidth=2.5)
        ax.axhline(y=1, color='gray', linestyle='--', label='S=1 (no preference)')
        ax.axhline(y=S_bulk, color='green', linestyle=':', 
                   label=f'Bulk selectivity: {S_bulk:.2f}')
        
        # Mark crossover
        L_cross = crossover_length(k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
        if not np.isnan(L_cross) and 0.3e-9 < L_cross < 100e-9:
            ax.axvline(x=L_cross * 1e9, color='red', linestyle=':', alpha=0.7)
            ax.annotate(f'Crossover\n{L_cross*1e9:.1f}nm', 
                        xy=(L_cross * 1e9, 1), xytext=(L_cross * 1e9 * 1.5, 1.5),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
        
        # Shade regions
        ax.fill_between(L_range * 1e9, S_bulk, S, where=S > S_bulk, 
                        alpha=0.2, color='blue', label=f'{scenario["product_A"]} enhanced')
        ax.fill_between(L_range * 1e9, S, S_bulk, where=S < S_bulk,
                        alpha=0.2, color='red', label=f'{scenario["product_B"]} enhanced')
        
        # Mark material pore sizes
        for mat_name, mat_props in MATERIALS.items():
            pore = mat_props['pore_nm']
            if 0.3 < pore < 100:
                S_at_pore = selectivity(pore * 1e-9, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
                ax.scatter([pore], [S_at_pore], s=50, zorder=5)
                if pore < 2:
                    ax.annotate(mat_name, xy=(pore, S_at_pore), 
                                xytext=(pore * 1.5, S_at_pore * 1.2),
                                fontsize=7, rotation=45)
        
        ax.set_xscale('log')
        ax.set_xlabel('Pore Diameter (nm)', fontsize=10)
        ax.set_ylabel(f'Selectivity ({scenario["product_A"]}/{scenario["product_B"]})', fontsize=10)
        ax.set_title(f'{scenario["name"]}\nΔα = {alpha_A - alpha_B:.1f}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.3, 100)
        
        # Store results
        results.append({
            'scenario': scenario['name'],
            'product_A': scenario['product_A'],
            'product_B': scenario['product_B'],
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'delta_alpha': alpha_A - alpha_B,
            'S_bulk': S_bulk,
            'S_1nm': selectivity(1e-9, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref),
            'S_10nm': selectivity(10e-9, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref),
            'L_crossover_nm': L_cross * 1e9 if not np.isnan(L_cross) else np.nan
        })
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_selectivity_scenarios.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_selectivity_scenarios.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Material comparison
    # ===================
    
    print("\n2. Comparing selectivity across materials...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use xylene scenario
    scenario = scenarios[0]
    k_A_bulk = scenario['k_A_bulk']
    k_B_bulk = scenario['k_B_bulk']
    alpha_A = scenario['alpha_A']
    alpha_B = scenario['alpha_B']
    
    ax = axes2[0]
    
    mat_names = []
    mat_selectivities = []
    mat_colors = []
    
    color_map = {'zeolite': 'blue', 'mesoporous': 'green', 'MOF': 'orange'}
    
    for mat_name, mat_props in sorted(MATERIALS.items(), key=lambda x: x[1]['pore_nm']):
        pore = mat_props['pore_nm'] * 1e-9
        S = selectivity(pore, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
        
        mat_names.append(f"{mat_name}\n({mat_props['pore_nm']:.1f}nm)")
        mat_selectivities.append(S)
        mat_colors.append(color_map[mat_props['type']])
    
    bars = ax.bar(range(len(mat_names)), mat_selectivities, color=mat_colors, alpha=0.7)
    ax.set_xticks(range(len(mat_names)))
    ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=1, color='gray', linestyle='--')
    ax.axhline(y=k_A_bulk/k_B_bulk, color='green', linestyle=':', label='Bulk')
    
    ax.set_ylabel('Selectivity (para/ortho)', fontsize=11)
    ax.set_title('Xylene Selectivity by Material\n(Blue=zeolite, Green=mesoporous, Orange=MOF)', 
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mark high selectivity
    for i, (bar, sel) in enumerate(zip(bars, mat_selectivities)):
        if sel > 2:
            ax.annotate(f'{sel:.1f}', xy=(i, sel), xytext=(i, sel + 0.5),
                        ha='center', fontsize=9)
    
    # Selectivity vs pore size scatter
    ax = axes2[1]
    
    for mat_type, color in color_map.items():
        pores = []
        sels = []
        for mat_name, mat_props in MATERIALS.items():
            if mat_props['type'] == mat_type:
                pore = mat_props['pore_nm']
                S = selectivity(pore * 1e-9, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
                pores.append(pore)
                sels.append(S)
        ax.scatter(pores, sels, s=100, c=color, label=mat_type, alpha=0.7)
    
    # Trend line
    L_fit = np.logspace(-0.5, 1, 50)
    S_fit = selectivity(L_fit * 1e-9, k_A_bulk, alpha_A, k_B_bulk, alpha_B, L_ref)
    ax.plot(L_fit, S_fit, 'k--', linewidth=2, label='RTM prediction')
    
    ax.set_xscale('log')
    ax.set_xlabel('Pore Diameter (nm)', fontsize=11)
    ax.set_ylabel('para/ortho Selectivity', fontsize=11)
    ax.set_title('Selectivity vs Pore Size\n(All materials)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=1, color='gray', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_material_comparison.png'), dpi=150)
    plt.close()
    
    # ===================
    # Part 3: Design space
    # ===================
    
    print("\n3. Creating design space map...")
    
    fig3, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D map: Δα vs L → Selectivity enhancement
    delta_alpha_range = np.linspace(-0.5, 0.8, 50)
    L_range_nm = np.logspace(-0.5, 2, 50)  # 0.3 to 100 nm
    
    DA, L_grid = np.meshgrid(delta_alpha_range, L_range_nm)
    
    # S_enhancement = S(L) / S_bulk = (L/L_ref)^(-Δα)
    S_enhancement = (L_grid * 1e-9 / L_ref) ** (-DA)
    
    # Plot
    levels = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    cs = ax.contourf(DA, L_grid, S_enhancement, levels=levels, 
                      norm=plt.matplotlib.colors.LogNorm(), cmap='RdBu_r')
    ax.contour(DA, L_grid, S_enhancement, levels=[1], colors='black', linewidths=2)
    
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Selectivity Enhancement (S/S_bulk)', fontsize=11)
    
    ax.set_xlabel('Δα = α_A - α_B', fontsize=12)
    ax.set_ylabel('Pore Diameter (nm)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('RTM Selectivity Design Space\nRed = Product A favored, Blue = Product B favored', 
                 fontsize=13)
    
    # Mark scenarios
    for scenario in scenarios:
        da = scenario['alpha_A'] - scenario['alpha_B']
        ax.annotate(scenario['name'].split('(')[0].strip(), 
                    xy=(da, 1), xytext=(da, 0.5),
                    fontsize=8, rotation=90, va='bottom',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(0.3, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_design_space.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_design_space.pdf'))
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'S3_scenario_results.csv'), index=False)
    
    # Material selectivities
    mat_data = []
    for scenario in scenarios:
        for mat_name, mat_props in MATERIALS.items():
            pore = mat_props['pore_nm'] * 1e-9
            S = selectivity(pore, scenario['k_A_bulk'], scenario['alpha_A'],
                           scenario['k_B_bulk'], scenario['alpha_B'], L_ref)
            mat_data.append({
                'scenario': scenario['name'],
                'material': mat_name,
                'pore_nm': mat_props['pore_nm'],
                'type': mat_props['type'],
                'selectivity': S
            })
    
    df_materials = pd.DataFrame(mat_data)
    df_materials.to_csv(os.path.join(output_dir, 'S3_material_selectivities.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S3: RTM Selectivity in Zeolites and MOFs
==========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM SELECTIVITY PREDICTION
--------------------------
For competing reactions A and B:
  S(L) = k_A/k_B = (k_A_bulk/k_B_bulk) × (L/L_ref)^(α_B - α_A)

If Δα = α_A - α_B > 0:
  → Smaller pores favor product A
  → Selectivity enhancement = (L/L_ref)^(-Δα)

SCENARIO RESULTS
----------------
"""
    
    for _, row in df_results.iterrows():
        summary += f"""
{row['scenario']}
  Δα = {row['delta_alpha']:.2f}
  S_bulk = {row['S_bulk']:.2f}
  S(1nm) = {row['S_1nm']:.2f}
  S(10nm) = {row['S_10nm']:.2f}
  Enhancement at 1nm: {row['S_1nm']/row['S_bulk']:.1f}×
"""
    
    summary += f"""
MATERIALS DATABASE
------------------
Zeolites:
  ZSM-5 (0.55nm), Mordenite (0.70nm), Beta (0.76nm), Y (0.74nm)
  
Mesoporous:
  MCM-41 (3.0nm), SBA-15 (8.0nm)
  
MOFs:
  ZIF-8 (1.16nm), HKUST-1 (0.9nm), UiO-66 (0.75nm), 
  MOF-5 (1.5nm), MIL-101 (3.4nm)

DESIGN PRINCIPLES
-----------------
1. Measure α for each reaction pathway independently
2. Calculate Δα = α_target - α_undesired
3. If Δα > 0: select smaller pores for higher selectivity
4. If Δα < 0: select larger pores or different mechanism
5. Crossover length: L* where selectivity = 1

EXPERIMENTAL VALIDATION
-----------------------
1. Measure selectivity in bulk (no confinement)
2. Measure in series of materials with different pore sizes
3. Plot log(S) vs log(L) - slope should equal -Δα
4. Verify with data collapse: S × L^Δα = constant
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nSelectivity enhancement at 1nm pore:")
    for _, row in df_results.iterrows():
        enh = row['S_1nm'] / row['S_bulk']
        print(f"  {row['scenario']}: {enh:.1f}×")
    
    print(f"\nOutputs: {output_dir}/")
    
    return df_results, df_materials


if __name__ == "__main__":
    main()
