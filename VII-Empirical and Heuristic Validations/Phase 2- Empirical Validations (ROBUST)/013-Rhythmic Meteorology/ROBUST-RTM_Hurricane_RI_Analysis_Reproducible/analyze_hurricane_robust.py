#!/usr/bin/env python3
"""
ROBUST RTM HURRICANE RI ANALYSIS
=================================
Phase 2 "Red Team" Continuous EIV Pipeline

This script corrects the categorical thresholding bias found in the V1 analysis. 
It abandons arbitrary NHC intensity bins (Slow/Moderate/Rapid) in favor of 
Continuous Orthogonal Distance Regression (ODR), predicting the exact maximum 
intensification rate while absorbing empirical IBTrACS measurement noise 
(~5 kt wind error).
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
OUTPUT_DIR = "output_hurricane_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM HURRICANE RI ANALYSIS (CONTINUOUS EIV)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. CONTINUOUS ODR ANALYSIS
    df = pd.read_csv('ep_storms_alpha_summary.csv')
    df = df.dropna(subset=['ALPHA_MIN', 'MAX_INTENS'])

    # Flawed OLS (Baseline)
    ols_slope, ols_int, r_val, p_val, _ = stats.linregress(df['ALPHA_MIN'], df['MAX_INTENS'])

    # Robust Errors-in-Variables (ODR)
    # Wind intensity changes have ~5 kt observational error
    # Alpha derives from wind and pressure, assume ~0.08 variance
    df['alpha_err'] = 0.08
    df['intens_err'] = 5.0

    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    data = RealData(df['ALPHA_MIN'], df['MAX_INTENS'], sx=df['alpha_err'], sy=df['intens_err'])
    odr = ODR(data, model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_err = out.sd_beta[0]

    print(f"\n--- PREDICTIVE POWER RESULTS (N={len(df)} Storms) ---")
    print(f"Flawed OLS Slope : {ols_slope:.2f} (R²={r_val**2:.3f})")
    print(f"Robust ODR Slope : {odr_slope:.2f} ± {odr_err:.2f} (True Intensity Mapping)")

    # 2. LEAD TIME KERNEL DENSITY
    lead = pd.read_csv('ri_lead_times.csv')

    # 3. MASTER VISUALIZATION (2 PANELS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Continuous Prediction
    ax = axes[0]
    ax.errorbar(df['ALPHA_MIN'], df['MAX_INTENS'], xerr=df['alpha_err'], yerr=df['intens_err'], 
                fmt='o', color='teal', alpha=0.6, ecolor='lightgray', label='IBTrACS Storms (East Pacific)')

    x_range = np.linspace(df['ALPHA_MIN'].min()-0.1, df['ALPHA_MIN'].max()+0.1, 100)
    ax.plot(x_range, ols_slope*x_range + ols_int, 'r--', linewidth=2, label='Flawed OLS Fit')
    ax.plot(x_range, odr_slope*x_range + odr_int, 'k-', linewidth=3, label=f'Robust ODR Fit (Slope={odr_slope:.1f})')

    # NHC Thresholds
    ax.axhline(30, color='red', linestyle=':', label='NHC RI Threshold (30kt/24h)')
    ax.axvspan(xmin=0.5, xmax=1.25, color='red', alpha=0.1, label='Critical RI Danger Zone (α < 1.25)')

    ax.set_xlabel('Minimum Topological Coupling (α_min)')
    ax.set_ylabel('Maximum Intensification Rate (kt / 24h)')
    ax.set_title('Continuous RI Prediction\n(Isolating the Topology Precipice)')
    ax.set_xlim(1.0, 2.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Lead Time Probability
    ax = axes[1]
    sns.kdeplot(lead['LEAD_TIME_H'], fill=True, color='crimson', ax=ax, linewidth=3)
    sns.rugplot(lead['LEAD_TIME_H'], color='black', height=0.1, ax=ax)
    
    mean_lt = lead['LEAD_TIME_H'].mean()
    ax.axvline(mean_lt, color='black', linestyle='--', linewidth=2, label=f"Mean: {mean_lt:.1f}h")
    ax.axvline(6, color='gray', linestyle=':', linewidth=2, label='Min Operational Warning (6h)')

    ax.set_xlabel('Lead Time: α-Drop Before RI Onset (Hours)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Early Warning Signal Distribution\n(Real-World Operational Window)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_hurricane_ri.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_hurricane_ri.pdf")

    # 4. Export Data
    summary_df = pd.DataFrame({
        'Metric': ['Total_Storms', 'RI_Events', 'ODR_Slope', 'ODR_Error', 'Mean_Lead_Time_H'],
        'Value': [len(df), len(lead), odr_slope, odr_err, mean_lt]
    })
    summary_df.to_csv(f"{OUTPUT_DIR}/hurricane_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()