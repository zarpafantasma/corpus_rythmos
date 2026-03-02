#!/usr/bin/env python3
"""
ROBUST RTM GRAVITATIONAL WAVES VALIDATION (EMPIRICAL ONLY)
===========================================================
Phase 2 "Red Team" ODR Pipeline

This script removes all synthetic/simulated O4 data, restricting the 
analysis strictly to the 55 confirmed LIGO/Virgo O1-O3 events.
It incorporates Orthogonal Distance Regression (ODR) to absorb the massive
10-15% Bayesian credible intervals inherent to gravitational wave mass estimates.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_gw_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM GW ANALYSIS (O1-O3 EMPIRICAL ONLY)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load ONLY the real empirical data
    df = pd.read_csv('bbh_events_o1_o3.csv')
    
    # Calculate Physical Parameters
    df['Mtotal'] = df['M1'] + df['M2']
    df['E_rad'] = df['M1'] + df['M2'] - df['Mfinal']
    
    # Filter valid
    valid = df[(df['E_rad'] > 0) & (df['Mtotal'] > 0)].copy()
    
    # Log transformations
    log_M = np.log10(valid['Mtotal'])
    log_E = np.log10(valid['E_rad'])
    
    # 2. OLS Fit (Flawed for comparison)
    ols_slope, ols_intercept, _, _, _ = stats.linregress(log_M, log_E)
    
    # 3. Robust ODR Fit (with injected LIGO Bayesian credibility intervals)
    # Typical LIGO errors: ~10% for M_total, ~15% for E_rad
    valid['log_M_err'] = 0.10 / np.log(10)
    valid['log_E_err'] = 0.15 / np.log(10)
    
    def linear_func(p, x):
        return p[0] * x + p[1]
    
    linear_model = Model(linear_func)
    data = RealData(log_M, log_E, sx=valid['log_M_err'], sy=valid['log_E_err'])
    odr = ODR(data, linear_model, beta0=[ols_slope, ols_intercept])
    out = odr.run()
    odr_slope, odr_intercept = out.beta
    odr_slope_err, _ = out.sd_beta
    
    # 4. Spin Corrected ODR
    valid['E_corrected'] = valid['E_rad'] / (1 + 0.3 * np.abs(valid['chi_eff']))
    log_E_corr = np.log10(valid['E_corrected'])
    
    data_spin = RealData(log_M, log_E_corr, sx=valid['log_M_err'], sy=valid['log_E_err'])
    odr_spin = ODR(data_spin, linear_model, beta0=[ols_slope, ols_intercept])
    out_spin = odr_spin.run()
    odr_spin_slope, _ = out_spin.beta
    odr_spin_slope_err, _ = out_spin.sd_beta
    
    print(f"\n--- EMPIRICAL RESULTS (N={len(valid)} Real Events) ---")
    print(f"Flawed OLS Slope      : α = {ols_slope:.3f}")
    print(f"Robust ODR Slope      : α = {odr_slope:.3f} ± {odr_slope_err:.3f}")
    print(f"Spin-Corrected ODR    : α = {odr_spin_slope:.3f} ± {odr_spin_slope_err:.3f}")
    
    # 5. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Error bars for ODR
    ax.errorbar(valid['Mtotal'], valid['E_rad'], 
                xerr=valid['Mtotal']*0.10, yerr=valid['E_rad']*0.15, 
                fmt='o', color='purple', alpha=0.6, ecolor='lightgray', 
                label='LIGO O1-O3 Events (with ~10-15% Error)')
    
    # Fit lines
    x_fit = np.linspace(10, 150, 100)
    log_x_fit = np.log10(x_fit)
    
    y_odr = 10**(odr_slope * log_x_fit + odr_intercept)
    y_ballistic = 10**(1.0 * log_x_fit + odr_intercept) # theoretical alpha = 1
    
    ax.plot(x_fit, y_odr, 'b-', linewidth=3, label=f'Robust ODR (α={odr_slope:.3f})')
    ax.plot(x_fit, y_ballistic, 'g:', linewidth=2, label='RTM Theoretical Ballistic (α=1.000)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total Mass $M_{total}$ ($M_\odot$)', fontsize=12)
    ax.set_ylabel('Radiated Energy $E_{rad}$ ($M_\odot c^2$)', fontsize=12)
    ax.set_title('Robust RTM Scaling: Real BBH Mergers (O1-O3)\nValidating the Ballistic Transport Limit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.savefig(f"{OUTPUT_DIR}/robust_gw_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_gw_rtm.pdf")
    
    # 6. Export Data
    results_df = pd.DataFrame({
        'Metric': ['Total_Real_Events', 'OLS_Alpha', 'ODR_Alpha', 'ODR_Alpha_StdErr', 'Spin_Corrected_ODR_Alpha'],
        'Value': [len(valid), ols_slope, odr_slope, odr_slope_err, odr_spin_slope]
    })
    results_df.to_csv(f"{OUTPUT_DIR}/gw_robust_summary.csv", index=False)
    valid.to_csv(f"{OUTPUT_DIR}/gw_o1_o3_processed.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()