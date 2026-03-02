#!/usr/bin/env python3
"""
ROBUST RTM FINANCIAL CRASH ANALYSIS
====================================
Phase 2 "Red Team" EIV & Monte Carlo Pipeline

This script corrects the "small-n" and point-estimate fallacies of the V1 analysis. 
Financial datasets are inherently noisy. This pipeline utilizes Orthogonal 
Distance Regression (ODR) to absorb typical trading measurement variances 
(peak-to-trough bounds) and employs Monte Carlo simulation to test if the 
pre-crash DFA α-drop genuinely survives continuous market noise.
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
OUTPUT_DIR = "output_finance_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM FINANCIAL CRASH ANALYSIS (ODR & NOISE INJECTION)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv('crash_alpha_analysis.csv')

    # 1. Fixing the OLS illusion
    df['Severity'] = df['Drop_Pct'].abs()
    df['Abs_Alpha_Drop'] = df['Alpha_Drop'].abs()

    # Flawed OLS for baseline comparison
    ols_slope, ols_int, r_val, _, _ = stats.linregress(df['Severity'], df['Abs_Alpha_Drop'])

    # Robust Errors-in-Variables (ODR)
    # Injecting market boundary uncertainty (5% severity error, 0.04 alpha rolling window error)
    df['sev_err'] = 5.0
    df['alpha_err'] = 0.04

    def linear_func(p, x): return p[0]*x + p[1]
    model = Model(linear_func)
    data = RealData(df['Severity'], df['Abs_Alpha_Drop'], sx=df['sev_err'], sy=df['alpha_err'])
    odr = ODR(data, model, beta0=[ols_slope, ols_int])
    out = odr.run()
    odr_slope, odr_int = out.beta
    odr_err = out.sd_beta[0]

    # 2. Monte Carlo Simulation of Alpha Distributions
    np.random.seed(42)
    sim_baseline, sim_immediate = [], []
    for _, row in df.iterrows():
        b_sims = np.random.normal(row['Baseline_Alpha'], 0.04, 1000)
        i_sims = np.random.normal(row['Immediate_Alpha'], 0.04, 1000)
        sim_baseline.extend(b_sims)
        sim_immediate.extend(i_sims)

    t_stat, p_sim = stats.ttest_ind(sim_baseline, sim_immediate)
    cohens_d = (np.mean(sim_immediate) - np.mean(sim_baseline)) / np.sqrt((np.std(sim_immediate)**2 + np.std(sim_baseline)**2)/2)
    mean_days = (df['Lead_Time_Hours']/24.0).mean()

    print(f"\n--- PROBABILISTIC TOPOLOGY SHIFT ---")
    print(f"Robust ODR Slope (Intensity): {odr_slope:.4f} ± {odr_err:.4f}")
    print(f"Baseline Market α (Normal)  : {np.mean(sim_baseline):.3f} ± {np.std(sim_baseline):.3f}")
    print(f"Crash State α (Collapse)    : {np.mean(sim_immediate):.3f} ± {np.std(sim_immediate):.3f}")
    print(f"Effect Size (Cohen's d)     : {cohens_d:.2f}")
    print(f"Mean Early Warning Lead Time: {mean_days:.1f} days")

    # 3. MASTER VISUALIZATION (3 PANELS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: The ODR Fit
    ax = axes[0]
    for market in df['Market'].unique():
        subset = df[df['Market'] == market]
        ax.errorbar(subset['Severity'], subset['Abs_Alpha_Drop'], xerr=subset['sev_err'], yerr=subset['alpha_err'],
                    fmt='o', alpha=0.8, label=market, capsize=3)

    x_range = np.linspace(df['Severity'].min()-5, df['Severity'].max()+5, 100)
    ax.plot(x_range, ols_slope*x_range + ols_int, 'r--', linewidth=2, label=f'Flawed OLS (R²={r_val**2:.2f})')
    ax.plot(x_range, odr_slope*x_range + odr_int, 'k-', linewidth=3, label=f'Robust ODR (Slope={odr_slope:.4f})')
    ax.set_xlabel('Crash Severity (% Price Drop)')
    ax.set_ylabel('Magnitude of DFA α-Drop')
    ax.set_title('Severity vs Topological Collapse\n(Absorbing Market Boundary Noise)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Monte Carlo State Space
    ax = axes[1]
    sns.kdeplot(sim_baseline, fill=True, color='green', label='Baseline (Normal Market)', ax=ax, linewidth=2)
    sns.kdeplot(sim_immediate, fill=True, color='red', label='Immediate (During Crash)', ax=ax, linewidth=2)
    ax.axvline(0.5, color='black', linestyle=':', linewidth=2, label='Random Walk (α=0.5)')
    ax.set_xlabel('DFA Exponent (α)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Robust State Space Transition\n(Simulating Continuous Market Noise)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 3: Lead Time Density
    ax = axes[2]
    sns.histplot(df['Lead_Time_Hours']/24.0, bins=10, kde=True, color='purple', ax=ax)
    ax.axvline(mean_days, color='black', linestyle='--', linewidth=2, label=f'Mean Warning: {mean_days:.1f} days')
    ax.set_xlabel('Early Warning Signal Lead Time (Days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Operational Warning Window\n(Days before Peak Crash)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_finance_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_finance_rtm.pdf")

    # 4. Export
    summary = pd.DataFrame({
        'Metric': ['Flawed_OLS_R2', 'ODR_Slope', 'ODR_Error', 'Baseline_Mean', 'Crash_Mean', 'Cohens_d', 'Mean_Warning_Days'],
        'Value': [r_val**2, odr_slope, odr_err, np.mean(sim_baseline), np.mean(sim_immediate), cohens_d, mean_days]
    })
    summary.to_csv(f"{OUTPUT_DIR}/finance_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()