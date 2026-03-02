#!/usr/bin/env python3
"""
ROBUST RTM QUANTUM DECOHERENCE ANALYSIS
=======================================
Phase 2 "Red Team" Multivariable ODR Pipeline

This script abandons the crude "Era binning" approach. Instead, it deploys a 
multivariable Orthogonal Distance Regression (ODR) to simultaneously untangle 
chronological technology improvements (Year) from the true topological transport 
scaling (Qubits), while absorbing realistic cryogenic calibration noise (~15%).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_quantum_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM QUANTUM DECOHERENCE (MULTIVARIABLE EIV)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv('ibm_quantum_processors.csv')

    # Log space transformations
    df['log_N'] = np.log10(df['Qubits'])
    df['log_T2'] = np.log10(df['T2_us'])
    df['Year_offset'] = df['Year'] - 2017 # Baseline technology year

    # 1. Flawed Naive OLS (Confounded)
    from scipy import stats
    naive_slope, naive_int, _, _, _ = stats.linregress(df['log_N'], df['log_T2'])

    # 2. Robust Multivariable ODR 
    # Cryogenic T2 fluctuations are massive; injecting 15% calibration variance
    df['log_T2_err'] = 0.15 / np.log(10)
    df['log_N_err'] = 1e-4  # Qubits count is exact
    df['Year_err'] = 1e-4   # Year is exact

    def multi_func(B, x):
        # B[0] = alpha (pure topological qubit scaling)
        # B[1] = gamma (chronological technology improvement per year)
        # B[2] = intercept
        return B[0]*x[0] + B[1]*x[1] + B[2]

    model = Model(multi_func)
    data = RealData(np.array([df['log_N'], df['Year_offset']]), df['log_T2'], 
                    sx=np.array([df['log_N_err'], df['Year_err']]), sy=df['log_T2_err'])

    # Initial algebraic guess via statsmodels
    X = sm.add_constant(df[['log_N', 'Year_offset']])
    res = sm.OLS(df['log_T2'], X).fit()
    guess = [res.params['log_N'], res.params['Year_offset'], res.params['const']]

    # Run the physical EIV model
    odr = ODR(data, model, beta0=guess)
    out = odr.run()

    alpha_true, gamma, intercept = out.beta
    alpha_err, gamma_err, _ = out.sd_beta

    print(f"\n--- EMPIRICAL RESULTS (N={len(df)} Processors) ---")
    print(f"Flawed Naive Slope (Technology Illusion) : α = {naive_slope:.3f}")
    print(f"True Topological Scaling (Isolated)      : α = {alpha_true:.3f} ± {alpha_err:.3f}")
    print(f"Chronological Tech Improvement Rate      : γ = +{gamma:.3f} dex/year")

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: The Simpson's Illusion
    ax = axes[0]
    scatter = ax.scatter(df['log_N'], df['log_T2'], c=df['Year'], cmap='viridis', s=80, edgecolor='black', alpha=0.8)
    x_line = np.linspace(df['log_N'].min(), df['log_N'].max(), 100)
    ax.plot(x_line, naive_slope * x_line + naive_int, 'r--', linewidth=2, label=f'Naive Fit (α={naive_slope:.2f})')
    ax.set_xlabel('log10(Number of Qubits)')
    ax.set_ylabel('log10(T2 Coherence Time)')
    ax.set_title('The Illusion: Raw Scaling\n(Confounded by Manufacturing Advancements)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hardware Release Year')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: The True Topological Correction
    ax = axes[1]
    # Mathematically subtract the technology gain to isolate the pure RTM topology
    df['log_T2_isolated'] = df['log_T2'] - (gamma * df['Year_offset'])

    ax.errorbar(df['log_N'], df['log_T2_isolated'], yerr=df['log_T2_err'], fmt='o', color='purple', alpha=0.7, ecolor='lightgray', label='Tech-Normalized T2 Variance')
    ax.plot(x_line, alpha_true * x_line + intercept, 'b-', linewidth=3, label=f'True RTM Scaling (α={alpha_true:.3f})')
    ax.set_xlabel('log10(Number of Qubits)')
    ax.set_ylabel('log10(T2) [Normalized to 2017 Baseline]')
    ax.set_title('Robust RTM Isolation\nRevealing Inverse Topological Transport (α < 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_quantum_rtm.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_quantum_rtm.pdf")

    # 4. Export the isolated physics
    results_df = pd.DataFrame({
        'Metric': ['Total_Processors', 'Flawed_Naive_Alpha', 'True_Topology_Alpha', 'True_Alpha_Error', 'Tech_Growth_Factor_Gamma'],
        'Value': [len(df), naive_slope, alpha_true, alpha_err, gamma]
    })
    results_df.to_csv(f"{OUTPUT_DIR}/quantum_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()