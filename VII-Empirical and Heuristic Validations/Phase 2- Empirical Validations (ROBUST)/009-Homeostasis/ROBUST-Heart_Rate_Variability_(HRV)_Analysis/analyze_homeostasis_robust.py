#!/usr/bin/env python3
"""
ROBUST RTM HOMEOSTASIS ANALYSIS: HRV & AGING
=============================================
Phase 2 "Red Team" Multivariable Pipeline

This script abandons the visual-only boxplot approach. It deploys a 
multivariable OLS regression to mathematically isolate the gradual 
topological decay of chronological aging from the catastrophic 
topological collapse caused by pathology (Heart Failure).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_homeostasis_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM HOMEOSTASIS ANALYSIS (MULTIVARIABLE ISOLATION)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Check delimiter
        df = pd.read_csv('hrv_aging_data.txt.txt', sep='\t')
        if df.shape[1] == 1:
            df = pd.read_csv('hrv_aging_data.txt.txt', sep='\s+')
    except FileNotFoundError:
        print("Error: 'hrv_aging_data.txt.txt' not found.")
        return

    # 1. Isolate Age from Pathology (Creating dummy variable for CHF)
    df['Is_CHF'] = (df['Group'] == 'Heart_Failure').astype(int)

    # 2. Multivariable Regression: Alpha ~ Age + Pathology
    X = df[['Avg_Age', 'Is_CHF']]
    X = sm.add_constant(X)
    y = df['DFA_Alpha_Coherence']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 3. Visualization of Trajectories
    fig, ax = plt.subplots(figsize=(10, 6))

    healthy = df[df['Is_CHF'] == 0]
    chf = df[df['Is_CHF'] == 1]

    ax.scatter(healthy['Avg_Age'], healthy['DFA_Alpha_Coherence'], color='royalblue', s=100, label='Healthy Subjects', edgecolor='k', zorder=3)
    ax.scatter(chf['Avg_Age'], chf['DFA_Alpha_Coherence'], color='crimson', s=120, marker='X', label='Heart Failure (CHF)', edgecolor='k', zorder=3)

    # Plot Mathematical Trajectories
    age_range = np.linspace(20, 85, 100)
    healthy_trajectory = model.params['const'] + model.params['Avg_Age'] * age_range
    chf_trajectory = healthy_trajectory + model.params['Is_CHF']

    ax.plot(age_range, healthy_trajectory, 'b--', linewidth=2, label=f'Healthy Aging (Loss: {model.params["Avg_Age"]:.4f}/yr)')
    ax.plot(age_range, chf_trajectory, 'r-', linewidth=3, label=f'Pathological Collapse (Penalty: {model.params["Is_CHF"]:.3f})')

    # Reference lines
    ax.axhline(1.0, color='green', linestyle=':', label='Optimal Coherence (α=1.0)')
    ax.axhline(0.5, color='black', linestyle=':', label='Uncorrelated / Random (α=0.5)')

    ax.set_xlabel('Chronological Age (Years)', fontsize=12)
    ax.set_ylabel('RTM Coherence Exponent (DFA α)', fontsize=12)
    ax.set_title('Robust RTM Homeostasis: Isolating Pathological Collapse from Normal Aging', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_hrv_aging.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_hrv_aging.pdf")

    # 4. Export Data
    results = pd.DataFrame({
        'Metric': ['Baseline_Young_Alpha', 'Healthy_Aging_Decay_Per_Year', 'Pathology_Penalty_Alpha', 'R_squared', 'P_value_Age', 'P_value_CHF'],
        'Value': [model.params['const'], model.params['Avg_Age'], model.params['Is_CHF'], model.rsquared, model.pvalues['Avg_Age'], model.pvalues['Is_CHF']]
    })
    results.to_csv(f"{OUTPUT_DIR}/homeostasis_robust_summary.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()