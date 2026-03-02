#!/usr/bin/env python3
"""
ROBUST RTM SEISMOLOGY ANALYSIS
===============================
Phase 2 "Red Team" ODR & Inversion Noise Pipeline

This script corrects the "perfect measurement fallacy" of the V1 analysis. 
Seismic rupture lengths and durations are not observed directly; they are 
calculated via seismogram inversions. This pipeline utilizes Orthogonal 
Distance Regression (ODR) to absorb typical geophysical inversion uncertainties 
(~15% for Length, ~20% for Duration) to prove the ballistic limit survives.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_seismic_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM SEISMOLOGY ANALYSIS (ODR & INVERSION NOISE)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv('earthquake_catalog.csv')

    # Log space
    df['log_L'] = np.log10(df['L'])
    df['log_tau'] = np.log10(df['tau'])

    # 1. Flawed OLS (Baseline)
    ols_slope, ols_int, r_val, p_val, std_err = stats.linregress(df['log_L'], df['log_tau'])

    # 2. Robust ODR (Inverting seismic variance)
    # Typical uncertainty: ~15% for Length, ~20% for Duration.
    log_L_err = 0.15 / np.log(10)
    log_tau_err = 0.20 / np.log(10)

    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    data = RealData(df['log_L'], df['log_tau'], sx=log_L_err, sy=log_tau_err)
    odr = ODR(data, model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_err = out.sd_beta[0]

    print(f"\nTotal Earthquakes: {len(df)}")
    print(f"Flawed OLS Slope : α = {ols_slope:.3f}")
    print(f"Robust ODR Slope : α = {odr_slope:.3f} ± {odr_err:.3f}")

    # 3. Fault Type Analysis (ODR)
    results = []
    print("\n--- By Fault Type (ODR) ---")
    for fault_type in df['Type'].unique():
        subset = df[df['Type'] == fault_type]
        if len(subset) > 3:
            sub_data = RealData(subset['log_L'], subset['log_tau'], sx=log_L_err, sy=log_tau_err)
            sub_odr = ODR(sub_data, model, beta0=[1.0, 0.0])
            sub_out = sub_odr.run()
            results.append({
                'Type': fault_type,
                'N': len(subset),
                'ODR_Alpha': sub_out.beta[0],
                'ODR_Err': sub_out.sd_beta[0]
            })
            print(f"{fault_type:<12} (n={len(subset):<2}) : α = {sub_out.beta[0]:.3f} ± {sub_out.sd_beta[0]:.3f}")

    # 4. MASTER VISUALIZATION
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'Strike-slip': 'crimson', 'Reverse': 'royalblue', 'Normal': 'forestgreen'}

    for fault_type in df['Type'].unique():
        subset = df[df['Type'] == fault_type]
        ax.errorbar(subset['L'], subset['tau'], 
                    xerr=subset['L']*0.15, yerr=subset['tau']*0.20, 
                    fmt='o', color=colors.get(fault_type, 'gray'), alpha=0.7, 
                    ecolor='lightgray', label=f'{fault_type} (n={len(subset)})')

    x_range = np.logspace(np.log10(df['L'].min()*0.8), np.log10(df['L'].max()*1.2), 100)
    log_x = np.log10(x_range)

    ax.plot(x_range, 10**(ols_slope * log_x + ols_int), 'r--', linewidth=2, label=f'OLS Fit (α={ols_slope:.3f})')
    ax.plot(x_range, 10**(odr_slope * log_x + odr_int), 'k-', linewidth=3, label=f'Robust ODR Fit (α={odr_slope:.3f})')
    ax.plot(x_range, 10**(1.0 * log_x + odr_int), 'b:', linewidth=2, label='Perfect Ballistic (α=1.00)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rupture Length $L$ (km) [Log Scale]')
    ax.set_ylabel('Rupture Duration $\\tau$ (s) [Log Scale]')
    ax.set_title('Robust RTM Seismology: Earthquake Rupture Transport\nAbsorbing Seismic Inversion Uncertainty', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_seismic_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_seismic_rtm.pdf")

    # 5. Export
    summary = pd.DataFrame({
        'Metric': ['Total_Quakes', 'OLS_Alpha', 'ODR_Alpha', 'ODR_Error'],
        'Value': [len(df), ols_slope, odr_slope, odr_err]
    })
    summary.to_csv(f"{OUTPUT_DIR}/seismic_robust_summary.csv", index=False)
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/seismic_fault_types_odr.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()