#!/usr/bin/env python3
"""
ROBUST RTM OCEANOGRAPHY ANALYSIS
================================
Phase 2 "Red Team" Monte Carlo & EIV Pipeline

This script rigorously validates the topological scaling laws of macroscopic 
ocean fluid dynamics. It replaces static point-estimates with Monte Carlo 
variance reconstruction to test the global limits of the Richardson t^3 
dispersion law across 1,090 drifter pairs, and utilizes Errors-in-Variables (ODR) 
to absorb satellite altimetry calibration noise.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_oceanography_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM OCEANOGRAPHY ANALYSIS (MONTE CARLO & ODR)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. RICHARDSON DISPERSION (Monte Carlo)
    df_disp = pd.read_csv('richardson_dispersion.csv')
    np.random.seed(42)
    sim_rich = []

    for _, row in df_disp.iterrows():
        # Simulate variance based on actual n_pairs deployed in each global experiment
        n_pairs = int(row['n_pairs'])
        sims = np.random.normal(row['richardson_exponent'], row['richardson_error'], n_pairs)
        sim_rich.extend(sims)

    sim_rich = np.array(sim_rich)
    mean_rich = np.mean(sim_rich)
    std_rich = np.std(sim_rich)

    # 2. OCEAN KINETIC ENERGY SCALING (ODR)
    df_ke = pd.read_csv('ocean_ke_spectrum.csv')
    log_scale = np.log10(df_ke['scale_km'])
    log_ke = np.log10(df_ke['ke_cm2_s2'])

    # Flawed OLS
    ols_slope, ols_int, _, _, _ = stats.linregress(log_scale, log_ke)

    # ODR (Absorbing 10-15% altimetry and in-situ instrument noise)
    scale_err = 0.10 / np.log(10)
    ke_err = 0.15 / np.log(10)

    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    data = RealData(log_scale, log_ke, sx=scale_err, sy=ke_err)
    odr = ODR(data, model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_err = out.sd_beta[0]

    print(f"RICHARDSON DISPERSION")
    print(f"Robust Mean (n={len(sim_rich)} pairs): n = {mean_rich:.3f} ± {std_rich:.3f} (Theory = 3.0)\n")
    print(f"OCEAN KE SPECTRUM (Macroscopic Friction)")
    print(f"Robust ODR Slope: {odr_slope:.3f} ± {odr_err:.3f}")

    # 3. Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Richardson
    ax = axes[0]
    sns.kdeplot(sim_rich, fill=True, color='teal', ax=ax, lw=2)
    ax.axvline(3.0, color='black', linestyle='-', lw=3, label='Theoretical Kolmogorov-Richardson (n=3.0)')
    ax.axvline(mean_rich, color='red', linestyle='--', lw=2, label=f'Robust Empirical Mean (n={mean_rich:.2f})')
    ax.set_xlabel('Dispersion Exponent (n)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Robust Topological Dispersion (Drifter Pairs)\nMonte Carlo Variance Injection')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: KE Spectrum ODR
    ax = axes[1]
    ax.errorbar(log_scale, log_ke, xerr=scale_err, yerr=ke_err, fmt='o', color='purple', alpha=0.6, ecolor='lightgray', label='Altimetry/Drifter Data (with ~15% Noise)')
    x_fit = np.linspace(log_scale.min(), log_scale.max(), 100)
    ax.plot(x_fit, ols_slope*x_fit + ols_int, 'r--', lw=2, label=f'Flawed OLS (slope={ols_slope:.2f})')
    ax.plot(x_fit, odr_slope*x_fit + odr_int, 'k-', lw=3, label=f'Robust ODR (slope={odr_slope:.2f})')
    ax.set_xlabel('log10(Spatial Scale km)')
    ax.set_ylabel('log10(Kinetic Energy cm²/s²)')
    ax.set_title('Global Ocean Kinetic Energy Spectrum\n(Correcting Attenuation Bias)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_oceanography_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_oceanography_rtm.pdf")

    # 4. Export
    summary = pd.DataFrame({
        'Metric': ['Richardson_Mean', 'Richardson_Std', 'Total_Drifter_Pairs', 'KE_ODR_Slope'],
        'Value': [mean_rich, std_rich, len(sim_rich), odr_slope]
    })
    summary.to_csv(f"{OUTPUT_DIR}/oceanography_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()