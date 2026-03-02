#!/usr/bin/env python3
"""
S2: Watershed Coherence Exponent (α)
====================================

RTM-Eco predicts: τ ∝ A^α for watersheds

where:
- τ = characteristic time (nutrient residence, flow response)
- A = watershed area (km²)
- α = coherence exponent

Different watershed types should have characteristic α:
- Steep mountain: α ~ 0.3-0.4 (fast drainage)
- Forested lowland: α ~ 0.4-0.5 (buffered response)
- Wetland-dominated: α ~ 0.5-0.6 (high retention)
- Urban/degraded: α ~ 0.2-0.3 (flashy response)

This simulation:
1. Models nutrient/flow residence times across watershed scales
2. Compares α across watershed types
3. Demonstrates the Ecosystem Coherence Index (ECI)

THEORETICAL MODEL - requires hydrological validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM WATERSHED MODEL
# =============================================================================

def residence_time_rtm(A, tau_0, alpha, A_ref=100):
    """
    RTM scaling for watershed residence time.
    
    τ(A) = τ_0 × (A/A_ref)^α
    
    Parameters:
    -----------
    A : array
        Watershed area (km²)
    tau_0 : float
        Residence time at reference scale (days)
    alpha : float
        Coherence exponent
    A_ref : float
        Reference area (km²)
    """
    return tau_0 * (A / A_ref) ** alpha


def flow_response_time(A, alpha, k=0.5):
    """
    Time to peak flow after rainfall event.
    
    Scales with watershed area: T_peak ∝ A^α
    """
    # Empirical: T_peak ~ k * A^α hours
    return k * A ** alpha


# =============================================================================
# WATERSHED TYPES
# =============================================================================

WATERSHED_TYPES = {
    'Mountain Stream': {
        'alpha': 0.35,
        'tau_0': 5,          # days
        'description': 'Steep gradient, fast drainage',
        'color': 'steelblue',
        'resilience': 'moderate'
    },
    'Forested Lowland': {
        'alpha': 0.45,
        'tau_0': 15,
        'description': 'Buffered by forest, moderate retention',
        'color': 'forestgreen',
        'resilience': 'high'
    },
    'Wetland Complex': {
        'alpha': 0.55,
        'tau_0': 30,
        'description': 'High retention, slow release',
        'color': 'teal',
        'resilience': 'very high'
    },
    'Agricultural': {
        'alpha': 0.30,
        'tau_0': 8,
        'description': 'Modified drainage, moderate flashiness',
        'color': 'gold',
        'resilience': 'moderate'
    },
    'Urban/Degraded': {
        'alpha': 0.25,
        'tau_0': 3,
        'description': 'Impervious surfaces, flashy response',
        'color': 'gray',
        'resilience': 'low'
    }
}


def compute_eci(alpha, alpha_min=0.2, alpha_max=0.6):
    """
    Compute Ecosystem Coherence Index (ECI).
    
    ECI = (α - α_min) / (α_max - α_min)
    
    Normalized to [0, 1] where higher = more resilient.
    """
    eci = (alpha - alpha_min) / (alpha_max - alpha_min)
    return np.clip(eci, 0, 1)


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_watershed_data(watershed_type, n_watersheds=40, 
                            A_min=1, A_max=5000, noise_level=0.12, seed=None):
    """
    Simulate residence time data for a watershed type.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = WATERSHED_TYPES[watershed_type]
    
    # Generate log-uniform areas
    log_A = np.random.uniform(np.log10(A_min), np.log10(A_max), n_watersheds)
    A = 10 ** log_A
    
    # True residence times
    tau_true = residence_time_rtm(A, params['tau_0'], params['alpha'])
    
    # Add measurement noise
    tau_measured = tau_true * np.exp(noise_level * np.random.randn(n_watersheds))
    A_measured = A * np.exp(0.05 * np.random.randn(n_watersheds))
    
    return pd.DataFrame({
        'type': watershed_type,
        'area_km2': A_measured,
        'area_true': A,
        'tau_days': tau_measured,
        'tau_true': tau_true,
        'alpha_true': params['alpha']
    })


def estimate_alpha_with_ci(A, tau, n_bootstrap=1000):
    """
    Estimate α with bootstrap confidence intervals.
    """
    log_A = np.log(A)
    log_tau = np.log(tau)
    
    # Point estimate
    slope, intercept, r, p, se = stats.linregress(log_A, log_tau)
    
    # Bootstrap for CI
    n = len(A)
    bootstrap_slopes = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        s, _, _, _, _ = stats.linregress(log_A[idx], log_tau[idx])
        bootstrap_slopes.append(s)
    
    ci_low, ci_high = np.percentile(bootstrap_slopes, [2.5, 97.5])
    
    return {
        'alpha': slope,
        'alpha_se': np.std(bootstrap_slopes),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'r_squared': r**2,
        'intercept': intercept
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S2: Watershed Coherence Exponent (α)")
    print("=" * 70)
    
    output_dir = "/home/claude/015-Rhythmic_Ecology/S2_watershed_alpha/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    A_ref = 100  # km² reference
    
    # ===================
    # Part 1: RTM scaling across watershed types
    # ===================
    
    print("\n1. Demonstrating RTM scaling across watershed types...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: τ vs A for all types
    ax = axes1[0, 0]
    A_range = np.logspace(0, 4, 100)  # 1 to 10,000 km²
    
    for ws_type, params in WATERSHED_TYPES.items():
        tau = residence_time_rtm(A_range, params['tau_0'], params['alpha'], A_ref)
        ax.plot(A_range, tau, linewidth=2, color=params['color'],
                label=f"{ws_type} (α={params['alpha']:.2f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Watershed Area (km²)', fontsize=11)
    ax.set_ylabel('Residence Time τ (days)', fontsize=11)
    ax.set_title('RTM Prediction: τ ∝ A^α\nWatershed Residence Time Scaling', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Simulated data
    ax = axes1[0, 1]
    
    all_data = []
    for ws_type in WATERSHED_TYPES.keys():
        df = simulate_watershed_data(ws_type, n_watersheds=30, seed=42)
        all_data.append(df)
        
        params = WATERSHED_TYPES[ws_type]
        ax.scatter(df['area_km2'], df['tau_days'], s=40, alpha=0.6,
                   color=params['color'], label=ws_type)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Watershed Area (km²)', fontsize=11)
    ax.set_ylabel('Residence Time (days)', fontsize=11)
    ax.set_title('Simulated Watershed Data\n(150 watersheds across 5 types)', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: α estimation with confidence intervals
    ax = axes1[1, 0]
    
    results = []
    for ws_type in WATERSHED_TYPES.keys():
        df = simulate_watershed_data(ws_type, n_watersheds=50, seed=42)
        fit = estimate_alpha_with_ci(df['area_km2'].values, df['tau_days'].values)
        
        eci = compute_eci(fit['alpha'])
        
        results.append({
            'type': ws_type,
            'alpha_true': WATERSHED_TYPES[ws_type]['alpha'],
            'alpha_est': fit['alpha'],
            'alpha_se': fit['alpha_se'],
            'ci_low': fit['ci_low'],
            'ci_high': fit['ci_high'],
            'r_squared': fit['r_squared'],
            'eci': eci,
            'resilience': WATERSHED_TYPES[ws_type]['resilience']
        })
    
    df_results = pd.DataFrame(results)
    
    # Plot estimates with error bars
    y_pos = range(len(df_results))
    
    ax.barh(y_pos, df_results['alpha_est'], xerr=df_results['alpha_se'] * 1.96,
            color=[WATERSHED_TYPES[t]['color'] for t in df_results['type']],
            alpha=0.7, capsize=5)
    
    # Add true values
    for i, row in df_results.iterrows():
        ax.axvline(x=row['alpha_true'], color='red', linestyle=':', alpha=0.5)
        ax.scatter([row['alpha_true']], [i], color='red', s=50, zorder=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_results['type'])
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_title('α Estimation with 95% CI\n(Red dots = true values)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Ecosystem Coherence Index
    ax = axes1[1, 1]
    
    colors = [WATERSHED_TYPES[t]['color'] for t in df_results['type']]
    bars = ax.bar(range(len(df_results)), df_results['eci'], color=colors, alpha=0.7)
    
    # Add threshold lines
    ax.axhline(y=0.25, color='red', linestyle='--', label='Low resilience')
    ax.axhline(y=0.50, color='orange', linestyle='--', label='Moderate')
    ax.axhline(y=0.75, color='green', linestyle='--', label='High resilience')
    
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels([t.split()[0] for t in df_results['type']], rotation=45, ha='right')
    ax.set_ylabel('Ecosystem Coherence Index (ECI)', fontsize=11)
    ax.set_title('ECI by Watershed Type\nHigher = More Resilient', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_watershed_alpha.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_watershed_alpha.pdf'))
    plt.close()
    
    # ===================
    # Part 2: Resilience comparison
    # ===================
    
    print("\n2. Computing resilience metrics...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flow response time comparison
    ax = axes2[0]
    
    A_test = np.array([10, 100, 1000])  # km²
    
    for ws_type, params in WATERSHED_TYPES.items():
        times = residence_time_rtm(A_test, params['tau_0'], params['alpha'], A_ref)
        ax.plot(A_test, times, 'o-', linewidth=2, markersize=8,
                color=params['color'], label=ws_type)
    
    ax.set_xscale('log')
    ax.set_xlabel('Watershed Area (km²)', fontsize=11)
    ax.set_ylabel('Residence Time (days)', fontsize=11)
    ax.set_title('Residence Time at Key Scales\nWetlands retain water longest', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # α vs ECI relationship
    ax = axes2[1]
    
    alpha_range = np.linspace(0.2, 0.6, 50)
    eci_range = compute_eci(alpha_range)
    
    ax.plot(alpha_range, eci_range, 'k-', linewidth=2, label='ECI curve')
    
    for _, row in df_results.iterrows():
        ax.scatter([row['alpha_est']], [row['eci']], s=100,
                   color=WATERSHED_TYPES[row['type']]['color'],
                   label=row['type'], zorder=5)
    
    ax.set_xlabel('Coherence Exponent α', fontsize=11)
    ax.set_ylabel('Ecosystem Coherence Index (ECI)', fontsize=11)
    ax.set_title('α → ECI Mapping\nHigher α indicates greater resilience', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 0.65)
    ax.set_ylim(0, 1)
    
    # Add zones
    ax.axvspan(0.2, 0.3, alpha=0.1, color='red', label='Low resilience zone')
    ax.axvspan(0.45, 0.6, alpha=0.1, color='green', label='High resilience zone')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_resilience_comparison.png'), dpi=150)
    plt.close()
    
    # ===================
    # Save results
    # ===================
    
    df_all.to_csv(os.path.join(output_dir, 'S2_simulated_watersheds.csv'), index=False)
    df_results.to_csv(os.path.join(output_dir, 'S2_alpha_estimates.csv'), index=False)
    
    # ===================
    # Summary
    # ===================
    
    mean_error = (df_results['alpha_est'] - df_results['alpha_true']).abs().mean()
    
    summary = f"""S2: Watershed Coherence Exponent (α)
=====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM-ECO WATERSHED MODEL
-----------------------
τ(A) = τ_0 × (A/A_ref)^α

where:
  τ = residence time (days)
  A = watershed area (km²)
  α = coherence exponent

WATERSHED TYPE PARAMETERS
-------------------------
"""
    
    for ws_type, params in WATERSHED_TYPES.items():
        summary += f"{ws_type}: α = {params['alpha']:.2f}, τ_0 = {params['tau_0']} days\n"
        summary += f"  {params['description']}\n"
    
    summary += f"""
α ESTIMATION RESULTS
--------------------
"""
    
    for _, row in df_results.iterrows():
        summary += f"{row['type']}:\n"
        summary += f"  True α = {row['alpha_true']:.3f}\n"
        summary += f"  Est α = {row['alpha_est']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]\n"
        summary += f"  ECI = {row['eci']:.2f} ({row['resilience']})\n"
    
    summary += f"""
Mean α error: {mean_error:.4f}

ECOSYSTEM COHERENCE INDEX (ECI)
-------------------------------
ECI = (α - 0.2) / (0.6 - 0.2)

Interpretation:
  ECI < 0.25: Low resilience (urban, degraded)
  ECI 0.25-0.50: Moderate resilience
  ECI 0.50-0.75: High resilience (forested)
  ECI > 0.75: Very high resilience (wetlands)

MANAGEMENT IMPLICATIONS
-----------------------
1. Wetland protection: Highest α, longest retention
   → Critical for nutrient buffering

2. Urbanization impact: Lowers α dramatically
   → Flashier hydrology, less resilience

3. Forest cover: Moderates α
   → Restoration increases resilience

4. Early warning: Declining α signals degradation
"""
    
    with open(os.path.join(output_dir, 'S2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMean α recovery error: {mean_error:.4f}")
    print("\nECI by watershed type:")
    for _, row in df_results.iterrows():
        print(f"  {row['type']}: ECI = {row['eci']:.2f} ({row['resilience']})")
    print(f"\nOutputs: {output_dir}/")
    
    return df_results


if __name__ == "__main__":
    main()
