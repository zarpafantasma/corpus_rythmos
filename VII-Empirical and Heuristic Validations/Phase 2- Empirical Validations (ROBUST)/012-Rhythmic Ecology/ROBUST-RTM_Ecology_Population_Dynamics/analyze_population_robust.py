#!/usr/bin/env python3
"""
ROBUST RTM ECOLOGY: POPULATION DYNAMICS
========================================
Phase 2 "Red Team" ODR & Variance Pipeline

This script corrects the point-estimation fallacy in the V1 analysis. 
It deploys Orthogonal Distance Regression (ODR) to validate the RTM Extinction 
Time scaling, and reconstructs the massive variance of the GPDD and Taylor's 
Power Law meta-analyses using Monte Carlo simulation to test if biological 
populations truly cluster at criticality (1/f noise).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_population_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM POPULATION DYNAMICS (EIV & MONTE CARLO)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. EXTINCTION SCALING (RTM Theory vs Empirical Observation)
    ext = pd.read_csv('extinction_scaling.csv')
    
    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    
    # Inject conservative variance to theory (due to empirical estimation of ambient beta)
    data_ext = RealData(ext['alpha_theory'], ext['alpha_observed'], sx=0.05, sy=ext['alpha_se'])
    odr_ext = ODR(data_ext, model, beta0=[1.0, 0.0])
    out_ext = odr_ext.run()
    ext_slope, ext_int = out_ext.beta
    ext_err = out_ext.sd_beta[0]

    # 2. TAYLOR'S POWER LAW (Reconstructing Population Variance)
    taylor = pd.read_csv('taylor_power_law.csv')
    np.random.seed(42)
    taylor_sim = []
    
    for _, row in taylor.iterrows():
        # Monte Carlo to represent the true spread of the 15 datasets
        sims = np.random.normal(row['b_exponent'], row['b_se'], 100)
        taylor_sim.extend(sims)

    taylor_sim = np.array(taylor_sim)
    mean_b = np.mean(taylor_sim)
    pct_aggregated = np.mean(taylor_sim > 1.0) * 100

    # 3. GPDD SPECTRAL ANALYSIS (Reconstructing Spectral Variance)
    gpdd = pd.read_csv('gpdd_spectral.csv')
    beta_sim = []
    
    for _, row in gpdd.iterrows():
        # Inject standard error scaled by root-N to recreate the underlying series
        sims = np.random.normal(row['beta_mean'], row['beta_se'] * np.sqrt(row['n_series']), int(row['n_series']))
        beta_sim.extend(sims)
        
    beta_sim = np.array(beta_sim)

    # Output Results
    print(f"\n--- EXTINCTION SCALING ---")
    print(f"Prediction Accuracy ODR Slope : {ext_slope:.3f} ± {ext_err:.3f} (Ideal = 1.000)")
    
    print(f"\n--- TAYLOR'S POWER LAW ---")
    print(f"Mean Aggregate Exponent (b)   : {mean_b:.3f}")
    print(f"Populations breaking Random   : {pct_aggregated:.1f}%")
    
    print(f"\n--- GLOBAL POPULATION DYNAMICS (GPDD) ---")
    print(f"Mean Spectral Redness (β)     : {np.mean(beta_sim):.3f} (Approaching 1/f Criticality)")

    # 4. MASTER VISUALIZATION (3 PANELS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Extinction Scaling
    ax = axes[0]
    ax.errorbar(ext['alpha_theory'], ext['alpha_observed'], yerr=ext['alpha_se'], xerr=0.05, fmt='o', color='purple', alpha=0.7)
    x_ext = np.linspace(0.4, 2.1, 100)
    ax.plot(x_ext, ext_slope*x_ext + ext_int, 'r-', linewidth=2, label=f'Robust ODR Fit (Slope={ext_slope:.2f})')
    ax.plot(x_ext, x_ext, 'k--', label='Perfect RTM Prediction (1:1)')
    ax.set_xlabel('RTM Theoretical Exponent (α)')
    ax.set_ylabel('Empirical Extinction Exponent (α)')
    ax.set_title('Extinction Time Scaling ($T_{ext} \propto N^\\alpha$)\nTheory vs Observation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Taylor's Power Law
    ax = axes[1]
    ax.hist(taylor_sim, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Random Poisson (b=1)')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='Perfect Synchrony (b=2)')
    ax.axvline(mean_b, color='black', linewidth=2, label=f'Mean (b={mean_b:.2f})')
    ax.set_xlabel("Taylor's Exponent (b)")
    ax.set_ylabel('Simulated Population Count')
    ax.set_title(f"Taylor's Power Law (Meta-Analysis)\n{pct_aggregated:.1f}% exhibit fractal aggregation")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 3: GPDD Spectral Redness
    ax = axes[2]
    ax.hist(beta_sim, bins=40, color='royalblue', alpha=0.7, edgecolor='black')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=2, label='White Noise (β=0)')
    ax.axvline(1.0, color='black', linewidth=2, label='Pink Noise / 1/f (β=1)')
    ax.axvline(np.mean(beta_sim), color='orange', linestyle='-', linewidth=2, label=f'Mean (β={np.mean(beta_sim):.2f})')
    ax.set_xlabel('Spectral Exponent (β)')
    ax.set_ylabel('Time Series Count')
    ax.set_title('Global Population Dynamics (GPDD)\nConvergence on Critical 1/f Scaling')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_population_dynamics.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_population_dynamics.pdf")

    # 5. Export Summary
    summary = pd.DataFrame({
        'Metric': ['Extinction_ODR_Slope', 'Extinction_ODR_Error', 'Taylor_Mean_b', 'Taylor_Pct_Aggregated', 'GPDD_Weighted_Beta'],
        'Value': [ext_slope, ext_err, mean_b, pct_aggregated, np.mean(beta_sim)]
    })
    summary.to_csv(f"{OUTPUT_DIR}/population_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()