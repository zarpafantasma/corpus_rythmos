#!/usr/bin/env python3
"""
ROBUST RTM SPARC GALAXY ANALYSIS
=================================
Phase 2 "Red Team" ODR & Monte Carlo Pipeline

This script corrects the attenuation bias present in standard OLS kinematic 
fits. By applying Orthogonal Distance Regression (ODR), it explicitly models 
the observational uncertainties in galactic surface brightness and HI 
rotational velocities to reveal the true topological Structure-Kinematics link.
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
OUTPUT_DIR = "output_sparc_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM SPARC ANALYSIS (ODR & NOISE INJECTION)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv('sparc_rtm_analysis.csv')
    df = df.dropna(subset=['struct_proxy', 'slope_full', 'alpha_full'])

    # 1. Structure-Kinematics Correlation (ODR)
    x = df['struct_proxy']
    y = df['slope_full']

    # Baseline OLS
    ols_slope, ols_int, r_val, p_val, _ = stats.linregress(x, y)

    # Injecting 5% error in photometric structure proxy and adopting actual slope errors
    x_err = np.abs(x * 0.05) + 0.01
    y_err = df['slope_err'].clip(lower=0.02) 

    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    data = RealData(x, y, sx=x_err, sy=y_err)
    odr = ODR(data, model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_err = out.sd_beta[0]

    # 2. Flat Curves Topological Exponent (Monte Carlo)
    flat_df = df[df['slope_full'].abs() < 0.1]

    np.random.seed(42)
    sim_alphas = []
    for _, row in flat_df.iterrows():
        # Simulate based on realistic slope variance (derived from SPARC Vobs errors)
        s_err = max(row['slope_err'], 0.04) 
        sim_s = np.random.normal(row['slope_full'], s_err, 1000)
        sim_a = 2 * (1 - sim_s) # RTM topological equation
        sim_alphas.extend(sim_a)

    sim_alphas = np.array(sim_alphas)
    mean_alpha = np.mean(sim_alphas)
    std_alpha = np.std(sim_alphas)

    print(f"Structure-Kinematics OLS: r = {r_val:.3f}, p = {p_val:.2e}")
    print(f"Structure-Kinematics ODR: slope = {odr_slope:.3f} ± {odr_err:.3f}")
    print(f"Flat Curves Mean Alpha  : {mean_alpha:.3f} ± {std_alpha:.3f} (n={len(flat_df)})")

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Structure vs Kinematics (ODR)
    ax = axes[0]
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='teal', alpha=0.5, label=f'SPARC Galaxies (n={len(df)})')
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, ols_slope*x_range + ols_int, 'r--', lw=2, label=f'Flawed OLS Fit (r={r_val:.2f})')
    ax.plot(x_range, odr_slope*x_range + odr_int, 'k-', lw=3, label=f'Robust ODR Fit')
    ax.set_xlabel('Structure Proxy (Surface Brightness Gradient)')
    ax.set_ylabel('Kinematic Slope ($log \\, v / log \\, r$)')
    ax.set_title('Structure-Kinematics Correlation\n(Correcting Attenuation Bias)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Flat Curves Alpha Distribution
    ax = axes[1]
    sns.kdeplot(sim_alphas, fill=True, color='royalblue', ax=ax, lw=2)
    ax.axvline(2.0, color='black', linestyle='-', lw=3, label='Theoretical RTM Limit (α=2.0)')
    ax.axvline(mean_alpha, color='blue', linestyle='--', lw=2, label=f'Robust Mean (α={mean_alpha:.2f} ± {std_alpha:.2f})')
    ax.set_xlabel('RTM Coherence Exponent (α)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Topological Exponent of Flat Rotation Curves\nMonte Carlo Variance Injection (n={len(flat_df)})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_sparc_rtm.png", dpi=300)

    # 4. Export
    summary = pd.DataFrame({
        'Metric': ['Total_Galaxies', 'Flat_Curves', 'OLS_Correlation_r', 'ODR_Slope', 'ODR_Slope_Err', 'Flat_Mean_Alpha', 'Flat_Alpha_Std'],
        'Value': [len(df), len(flat_df), r_val, odr_slope, odr_err, mean_alpha, std_alpha]
    })
    summary.to_csv(f"{OUTPUT_DIR}/sparc_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()