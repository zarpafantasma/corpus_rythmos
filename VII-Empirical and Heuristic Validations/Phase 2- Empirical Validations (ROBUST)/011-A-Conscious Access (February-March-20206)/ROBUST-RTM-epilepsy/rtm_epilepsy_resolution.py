#!/usr/bin/env python3
"""
RTM Epilepsy Resolution Analysis
================================
Validates the physical predictions of RTM (Multiscale Temporal Relativity) 
on the UCI Epileptic Seizure Recognition dataset (N=11,500).

Key Operational Pivots:
1. Modality Segregation: iEEG (intracranial) cannot be directly compared to Scalp EEG.
2. Topological Collapse: Seizures are identified by a violent drop in R^2 (Holonomy), not just α.
3. Geometric Filtration: Filtering Scalp EEG for R^2 > 0.6 improves the consciousness signal.

Author: RTM Research Group
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def load_and_segregate_data(csv_path):
    """Load data and enforce strict modality separation."""
    df = pd.read_csv(csv_path)
    
    class_map = {1: 'Seizure', 2: 'Tumor Area', 3: 'Healthy Brain', 
                 4: 'Eyes Closed', 5: 'Eyes Open'}
    df['label'] = df['class'].map(class_map)
    df['modality'] = df['class'].apply(lambda x: 'Scalp EEG' if x in [4, 5] else 'Intracranial (iEEG)')
    
    return df

def cohens_d(x, y):
    """Calculate Cohen's d for effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def analyze_ieeg_collapse(df):
    """Analyze the R^2 topological collapse during seizures (iEEG only)."""
    ieeg = df[df['modality'] == 'Intracranial (iEEG)']
    seizure_r2 = ieeg[ieeg['class'] == 1]['r_squared']
    healthy_r2 = ieeg[ieeg['class'] == 3]['r_squared']
    
    d_val = cohens_d(seizure_r2, healthy_r2)
    t_val, p_val = stats.ttest_ind(seizure_r2, healthy_r2, equal_var=False)
    
    return d_val, p_val

def analyze_scalp_filtration(df):
    """Analyze how geometric filtration improves the consciousness alpha signal (Scalp EEG)."""
    scalp = df[df['modality'] == 'Scalp EEG']
    
    # 1. Unfiltered Baseline
    open_un = scalp[scalp['class'] == 5]['alpha']
    closed_un = scalp[scalp['class'] == 4]['alpha']
    d_unfiltered = cohens_d(open_un, closed_un)
    
    # 2. Topologically Filtered (R^2 > 0.6)
    scalp_f = scalp[scalp['r_squared'] > 0.6]
    open_f = scalp_f[scalp_f['class'] == 5]['alpha']
    closed_f = scalp_f[scalp_f['class'] == 4]['alpha']
    d_filtered = cohens_d(open_f, closed_f)
    t_val, p_val = stats.ttest_ind(open_f, closed_f, equal_var=False)
    
    return d_unfiltered, d_filtered, p_val

def generate_resolution_plot(df, output_path):
    """Generate the 2-panel plot demonstrating structural collapse and filtered alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: iEEG R^2 Collapse
    ieeg = df[df['modality'] == 'Intracranial (iEEG)']
    sns.boxplot(data=ieeg, x='label', y='r_squared', 
                palette={'Healthy Brain':'#2ecc71', 'Tumor Area':'#f1c40f', 'Seizure':'#e74c3c'}, 
                ax=axes[0], order=['Healthy Brain', 'Tumor Area', 'Seizure'])
    axes[0].set_title('A) Intracranial (iEEG): The Topological Collapse\nSeizures physically fracture scale-free coherence (R² drop)')
    axes[0].set_ylabel('R² (Power Law Fit Quality)')
    axes[0].set_xlabel('Clinical State')

    # Panel B: Scalp EEG Alpha (Filtered)
    scalp_filtered = df[(df['modality'] == 'Scalp EEG') & (df['r_squared'] > 0.6)]
    sns.violinplot(data=scalp_filtered, x='label', y='alpha', 
                   palette={'Eyes Closed':'#3498db', 'Eyes Open':'#9b59b6'}, 
                   ax=axes[1], order=['Eyes Closed', 'Eyes Open'], inner='quartile')
    axes[1].set_title('B) Scalp EEG: Topological Exponent (α)\nFiltered for Coherence (R² > 0.6)')
    axes[1].set_ylabel('RTM Exponent (α)')
    axes[1].set_xlabel('Consciousness State')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print("Executing RTM Epilepsy Resolution Pipeline...")
    df = load_and_segregate_data('rtm_epilepsy_real_results.csv')
    
    d_collapse, p_collapse = analyze_ieeg_collapse(df)
    d_unf, d_f, p_f = analyze_scalp_filtration(df)
    
    # Save statistics CSV
    stats_data = [
        {'Analysis': 'iEEG Topological Collapse', 'Comparison': 'Seizure vs Healthy', 'Metric': 'R_squared', 'Cohens_d': d_collapse, 'p_value': p_collapse},
        {'Analysis': 'Scalp Consciousness Gradient', 'Comparison': 'Eyes Open vs Closed (Unfiltered)', 'Metric': 'Alpha', 'Cohens_d': d_unf, 'p_value': np.nan},
        {'Analysis': 'Scalp Consciousness Gradient', 'Comparison': 'Eyes Open vs Closed (Filtered R2>0.6)', 'Metric': 'Alpha', 'Cohens_d': d_f, 'p_value': p_f}
    ]
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv('RTM_Epilepsy_Final_Stats.csv', index=False)
    
    # Generate Plot
    generate_resolution_plot(df, 'RTM_Epilepsy_Final_Resolution.png')
    print("Pipeline complete. Saved RTM_Epilepsy_Final_Stats.csv and RTM_Epilepsy_Final_Resolution.png")

if __name__ == "__main__":
    main()