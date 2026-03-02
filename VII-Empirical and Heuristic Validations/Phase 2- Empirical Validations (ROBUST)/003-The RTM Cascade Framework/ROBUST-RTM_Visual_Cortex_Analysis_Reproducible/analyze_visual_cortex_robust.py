#!/usr/bin/env python3
"""
ROBUST RTM VISUAL CORTEX ANALYSIS
=================================
Phase 2 "Red Team" ODR Pipeline

This script corrects the attenuation and aggregation biases from the V1 analysis. 
It applies Orthogonal Distance Regression (ODR) to properly incorporate 
measurement errors (Errors-In-Variables) in fMRI receptive fields and latencies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_visual_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM VISUAL CORTEX ANALYSIS (EIV/ODR EDITION)")
    print("=" * 60)
    
    df = pd.read_csv('visual_cortex_data.csv')
    
    # 1. Convert linear standard deviations to logarithmic standard errors
    # Formula: d(log10(x)) ≈ std(x) / (x * ln(10))
    df['log_RF_err'] = df['RF_std'] / (df['RF_deg'] * np.log(10))
    df['log_Lat_err'] = df['Latency_std'] / (df['Latency_ms'] * np.log(10))
    
    # 2. Flawed OLS (Original calculation for comparison)
    ols_slope, ols_intercept, _, _, _ = stats.linregress(df['log_RF'], df['log_Latency'])
    
    # 3. Robust Orthogonal Distance Regression (ODR)
    def linear_func(p, x):
        return p[0] * x + p[1]
        
    linear_model = Model(linear_func)
    data = RealData(df['log_RF'], df['log_Latency'], sx=df['log_RF_err'], sy=df['log_Lat_err'])
    odr = ODR(data, linear_model, beta0=[ols_slope, ols_intercept])
    out = odr.run()
    
    odr_slope, odr_intercept = out.beta
    odr_slope_err, _ = out.sd_beta
    
    # 4. Raw Population Simulation (Breaking Aggregation Bias)
    np.random.seed(42)
    raw_rf, raw_lat = [], []
    for _, row in df.iterrows():
        n_neurons = int(row['n_studies']) * 10
        sim_rf = np.random.normal(row['RF_deg'], row['RF_std'], n_neurons)
        sim_lat = np.random.normal(row['Latency_ms'], row['Latency_std'], n_neurons)
        valid = (sim_rf > 0) & (sim_lat > 0)
        raw_rf.extend(sim_rf[valid])
        raw_lat.extend(sim_lat[valid])
        
    raw_slope, raw_intercept, raw_r, _, _ = stats.linregress(np.log10(raw_rf), np.log10(raw_lat))
    
    print(f"\n--- ERRORS-IN-VARIABLES (ODR) RESULTS ---")
    print(f"Flawed OLS Slope (Ignored Noise) : α = {ols_slope:.3f}")
    print(f"True ODR Slope (Absorbed Noise)  : α = {odr_slope:.3f} ± {odr_slope_err:.3f}")
    print(f"Raw Population Subject-Level     : α = {raw_slope:.3f} (R² = {raw_r**2:.3f})")
    
    # 5. Generate Publication Graphic
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: OLS vs ODR
    ax = axes[0]
    ax.errorbar(df['log_RF'], df['log_Latency'], xerr=df['log_RF_err'], yerr=df['log_Lat_err'], 
                fmt='o', color='gray', alpha=0.5, label='Aggregated Data with Errors')
    x_line = np.linspace(df['log_RF'].min(), df['log_RF'].max(), 100)
    ax.plot(x_line, ols_slope * x_line + ols_intercept, 'r--', label=f'Flawed OLS (α={ols_slope:.3f})')
    ax.plot(x_line, odr_slope * x_line + odr_intercept, 'g-', linewidth=2, label=f'Robust ODR (α={odr_slope:.3f})')
    ax.set_xlabel('log10(Receptive Field Size)')
    ax.set_ylabel('log10(Latency)')
    ax.set_title('Attenuation Bias Correction (Errors-In-Variables)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Raw Population
    ax = axes[1]
    ax.scatter(np.log10(raw_rf), np.log10(raw_lat), alpha=0.1, color='blue', s=5, label='Raw Simulated Variance')
    ax.plot(x_line, raw_slope * x_line + raw_intercept, 'k-', linewidth=2, label=f'True Pop. Fit (α={raw_slope:.3f})')
    ax.plot(x_line, 0.5 * x_line + 1.6, 'r:', linewidth=2, label='Diffusive Limit (α=0.500)')
    ax.set_xlabel('log10(Receptive Field Size)')
    ax.set_ylabel('log10(Latency)')
    ax.set_title(f'Subject-Level Hierarchy (n={len(raw_rf)})\nDispelling Aggregation Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visual_cortex_robust.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/visual_cortex_robust.pdf")
    
    # Export Data
    results_df = pd.DataFrame({
        'Metric': ['Flawed_OLS_Alpha', 'Robust_ODR_Alpha', 'ODR_Alpha_StdErr', 'Raw_Population_Alpha', 'Raw_R_Squared'],
        'Value': [ols_slope, odr_slope, odr_slope_err, raw_slope, raw_r**2]
    })
    results_df.to_csv(f"{OUTPUT_DIR}/visual_cortex_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()