#!/usr/bin/env python3
"""
ROBUST RTM EPIDEMIOLOGY ANALYSIS
================================
Phase 2 "Red Team" ODR & Monte Carlo Pipeline

This script corrects the attenuation bias present in global pandemic modeling. 
Standard OLS power-law fits assume public health reporting is perfect. 
Here, we deploy Orthogonal Distance Regression (ODR) to explicitly absorb 
a 20% underreporting variance in global COVID-19 cases, and utilize 
Monte Carlo simulations to map the true probabilistic distribution of the 
super-spreader overdispersion parameter (k).
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
OUTPUT_DIR = "output_epidemiology_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM EPIDEMIOLOGY ANALYSIS (NOISE INJECTION & ODR)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. POWER-LAW CASE DISTRIBUTION (Overcoming Attenuation Bias)
    df_countries = pd.read_csv('covid_countries.csv')
    df_countries = df_countries.sort_values(by='total_cases', ascending=False).reset_index(drop=True)
    df_countries['rank'] = df_countries.index + 1

    # Limit to top 100 for reliable power-law tail
    df_top = df_countries.head(100).copy()

    log_rank = np.log10(df_top['rank'])
    log_cases = np.log10(df_top['total_cases'])

    # Flawed OLS for baseline
    ols_slope, ols_int, r_val, p_val, _ = stats.linregress(log_rank, log_cases)

    # ODR Fit (Injecting realistic ~20% underreporting noise, 5% rank noise)
    log_cases_err = 0.20 / np.log(10)
    log_rank_err = 0.05 / np.log(10)

    def linear_func(p, x): return p[0] * x + p[1]
    
    linear_model = Model(linear_func)
    data = RealData(log_rank, log_cases, sx=log_rank_err, sy=log_cases_err)
    odr = ODR(data, linear_model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_slope_err, _ = out.sd_beta

    # Rank ~ Cases^(-alpha) => log(Cases) = (-1/alpha)*log(Rank) + c => alpha = -1/slope
    ols_alpha = -1.0 / ols_slope
    odr_alpha = -1.0 / odr_slope
    odr_alpha_err = np.abs(odr_slope_err / (odr_slope**2))

    # 2. SUPER-SPREADER OVERDISPERSION (Monte Carlo)
    df_k = pd.read_csv('super_spreader_k.csv')
    covid_k_data = df_k[df_k['disease'].str.contains('COVID')].copy()

    np.random.seed(42)
    simulated_ks = []

    for _, row in covid_k_data.iterrows():
        mean_k = row['k_estimate']
        std_k = (row['k_high'] - row['k_low']) / 3.92 # 95% CI roughly 3.92 std devs
        sims = np.random.normal(mean_k, std_k, 5000)
        sims = sims[sims > 0] # k must be positive
        simulated_ks.extend(sims)

    simulated_ks = np.array(simulated_ks)
    robust_k_mean = np.mean(simulated_ks)
    robust_k_std = np.std(simulated_ks)

    print(f"POWER-LAW TAIL (N=100 Countries)")
    print(f"Flawed OLS Exponent : α = {ols_alpha:.3f}")
    print(f"Robust ODR Exponent : α = {odr_alpha:.3f} ± {odr_alpha_err:.3f}\n")
    print(f"SUPER-SPREADER K-PARAMETER (Monte Carlo)")
    print(f"Robust Mean k       : {robust_k_mean:.3f} ± {robust_k_std:.3f}")

    # 3. Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Rank-Frequency Power Law
    ax = axes[0]
    ax.errorbar(log_rank, log_cases, yerr=log_cases_err, xerr=log_rank_err, fmt='o', color='teal', alpha=0.6, ecolor='lightgray', label='Top 100 Countries (with ~20% Noise)')
    x_fit = np.linspace(log_rank.min(), log_rank.max(), 100)

    ax.plot(x_fit, ols_slope * x_fit + ols_int, 'r--', linewidth=2, label=f'Flawed OLS (α={ols_alpha:.2f})')
    ax.plot(x_fit, odr_slope * x_fit + odr_int, 'k-', linewidth=3, label=f'Robust ODR (α={odr_alpha:.2f})')

    ax.set_xlabel('log10(Rank)')
    ax.set_ylabel('log10(Total Cases)')
    ax.set_title('Robust Topological Scaling of Global Pandemic\n(Correcting Underreporting Attenuation Bias)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Super-spreader k-parameter density
    ax = axes[1]
    sns.kdeplot(simulated_ks, fill=True, color='purple', ax=ax, lw=2)
    ax.axvline(1.0, color='red', linestyle=':', lw=2, label='Poisson Random (k ≥ 1.0)')
    ax.axvline(robust_k_mean, color='black', linestyle='--', lw=3, label=f'Robust Mean (k={robust_k_mean:.2f})')
    ax.set_xlabel('Overdispersion Parameter (k)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Super-Spreader Fat-Tailed Topology\nMonte Carlo Variance Injection (k << 1.0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_epidemiology_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_epidemiology_rtm.pdf")

    # 4. Export
    summary = pd.DataFrame({
        'Metric': ['OLS_Alpha', 'ODR_Alpha', 'ODR_Alpha_Error', 'Robust_k_Mean', 'Robust_k_Std'],
        'Value': [ols_alpha, odr_alpha, odr_alpha_err, robust_k_mean, robust_k_std]
    })
    summary.to_csv(f"{OUTPUT_DIR}/epidemiology_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()