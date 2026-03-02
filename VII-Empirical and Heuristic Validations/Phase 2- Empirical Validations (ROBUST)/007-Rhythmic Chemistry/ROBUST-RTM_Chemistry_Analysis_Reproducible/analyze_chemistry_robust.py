#!/usr/bin/env python3
"""
ROBUST RTM CHEMISTRY ANALYSIS: TWO DIFFUSION REGIMES
====================================================
Phase 2 "Red Team" ODR & Guest-Normalization Pipeline

This script corrects the attenuation bias and the Simpson's Paradox found 
in the V1 analysis. It uses Guest-Normalization to isolate the pure 
topological pore-size effect in Zeolites, and Orthogonal Distance Regression (ODR) 
to properly absorb measurement errors in both hydrodynamic radii and pore sizes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_chemistry_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM CHEMISTRY ANALYSIS (GUEST-NORMALIZED ODR)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. BULK REGIME: STOKES-EINSTEIN (INVERSE TRANSPORT)
    # ---------------------------------------------------------
    s_df = pd.read_csv('stokes_einstein_diffusion.csv')
    
    # Flawed OLS
    ols_s_slope, ols_s_int, r_s, _, _ = stats.linregress(s_df['log_r'], s_df['log_D'])

    # Robust ODR (Assuming ~5% error in D and r measurements)
    s_df['log_r_err'] = 0.05 / np.log(10)
    s_df['log_D_err'] = 0.05 / np.log(10)

    def linear_func(p, x): return p[0]*x + p[1]
    se_model = Model(linear_func)
    se_data = RealData(s_df['log_r'], s_df['log_D'], sx=s_df['log_r_err'], sy=s_df['log_D_err'])
    se_odr = ODR(se_data, se_model, beta0=[ols_s_slope, ols_s_int])
    se_out = se_odr.run()
    odr_s_slope, odr_s_int = se_out.beta
    odr_s_err = se_out.sd_beta[0]

    print(f"\n--- BULK REGIME: STOKES-EINSTEIN (n={len(s_df)}) ---")
    print(f"Flawed OLS Slope : α = {ols_s_slope:.3f}")
    print(f"Robust ODR Slope : α = {odr_s_slope:.3f} ± {odr_s_err:.3f}")

    # ---------------------------------------------------------
    # 2. CONFINED REGIME: ZEOLITES (CRITICAL/RESONANT TRANSPORT)
    # ---------------------------------------------------------
    z_df = pd.read_csv('zeolite_diffusion.csv')

    # Flawed pooled OLS
    ols_z_slope, ols_z_int, r_z, _, _ = stats.linregress(z_df['log_L'], z_df['log_D'])

    # GUEST-NORMALIZED ODR (Isolating topology from guest-specific baseline)
    z_df['log_D_norm'] = z_df.groupby('Guest')['log_D'].transform(lambda x: x - x.mean())
    z_df['log_L_norm'] = z_df.groupby('Guest')['log_L'].transform(lambda x: x - x.mean())

    # Zeolites have high measurement error in QENS diffusion (often 20%) and pore sizes (~5%)
    z_df['log_L_err'] = 0.05 / np.log(10)
    z_df['log_D_err'] = 0.20 / np.log(10)

    z_data = RealData(z_df['log_L_norm'], z_df['log_D_norm'], sx=z_df['log_L_err'], sy=z_df['log_D_err'])
    z_odr = ODR(z_data, se_model, beta0=[ols_z_slope, 0.0])
    z_out = z_odr.run()
    odr_z_slope, odr_z_int = z_out.beta
    odr_z_err = z_out.sd_beta[0]

    print(f"\n--- CONFINED REGIME: ZEOLITES (n={len(z_df)}) ---")
    print(f"Flawed Pooled OLS (Confounded) : α = {ols_z_slope:.3f}")
    print(f"Guest-Normalized Robust ODR    : α = {odr_z_slope:.3f} ± {odr_z_err:.3f}")

    # ---------------------------------------------------------
    # 3. VISUALIZATION
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Stokes-Einstein
    ax = axes[0]
    ax.errorbar(s_df['log_r'], s_df['log_D'], xerr=s_df['log_r_err'], yerr=s_df['log_D_err'], 
                fmt='o', color='royalblue', alpha=0.6, ecolor='lightgray', label='Bulk Diffusion Data')
    x_s = np.linspace(s_df['log_r'].min(), s_df['log_r'].max(), 100)
    ax.plot(x_s, ols_s_slope*x_s + ols_s_int, 'r--', linewidth=2, label=f'Flawed OLS (α={ols_s_slope:.2f})')
    ax.plot(x_s, odr_s_slope*x_s + odr_s_int, 'k-', linewidth=3, label=f'Robust ODR (α={odr_s_slope:.2f})')
    ax.set_xlabel('log10(Hydrodynamic Radius)')
    ax.set_ylabel('log10(Diffusion Coefficient)')
    ax.set_title('Bulk Regime (Stokes-Einstein)\nInverse Transport Class (α < 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Zeolites
    ax = axes[1]
    for guest, group in z_df.groupby('Guest'):
        ax.scatter(group['log_L_norm'], group['log_D_norm'], s=80, alpha=0.8, label=guest, edgecolor='k')

    x_z = np.linspace(z_df['log_L_norm'].min(), z_df['log_L_norm'].max(), 100)
    # plot naive pooled slope just to show the contrast (centered at 0)
    ax.plot(x_z, ols_z_slope*x_z, 'r--', linewidth=2, label=f'Pooled OLS (α={ols_z_slope:.2f})')
    ax.plot(x_z, odr_z_slope*x_z + odr_z_int, 'g-', linewidth=3, label=f'Normalized ODR (α={odr_z_slope:.2f})')

    ax.set_xlabel('log10(Pore Size) [Guest-Normalized]')
    ax.set_ylabel('log10(Diffusion) [Guest-Normalized]')
    ax.set_title('Confined Regime (Zeolites)\nCritical/Resonant Transport Class (α >> 1)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_chemistry_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_chemistry_rtm.pdf")

    # 4. Export the data
    results_df = pd.DataFrame({
        'Regime': ['Bulk (Stokes-Einstein)', 'Confined (Zeolites)'],
        'Flawed_OLS_Alpha': [ols_s_slope, ols_z_slope],
        'Robust_ODR_Alpha': [odr_s_slope, odr_z_slope],
        'ODR_Alpha_Error': [odr_s_err, odr_z_err]
    })
    results_df.to_csv(f"{OUTPUT_DIR}/chemistry_robust_summary.csv", index=False)
    z_df.to_csv(f"{OUTPUT_DIR}/zeolite_normalized.csv", index=False)
    
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()