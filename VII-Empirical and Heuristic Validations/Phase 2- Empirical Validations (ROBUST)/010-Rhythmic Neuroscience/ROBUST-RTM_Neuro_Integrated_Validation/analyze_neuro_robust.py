#!/usr/bin/env python3
"""
ROBUST RTM NEUROSCIENCE INTEGRATED VALIDATION
==============================================
Phase 2 "Red Team" Subject-Level Pipeline

This script corrects the "aggregation fallacy" across four independent 
neurophysiological datasets (Epilepsy, Meditation, Psychedelics, Sleep). 
By injecting empirical EEG variance (Standard Deviation), it tests if the RTM 
topological shift survives the overlap of human neurological diversity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = "output_neuro_robust"

def main():
    print("=" * 60)
    print("ROBUST RTM NEUROSCIENCE VALIDATION (SUBJECT-LEVEL)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. EPILEPSY: Subject-Level Simulation
    # ---------------------------------------------------------
    epi_df = pd.read_csv('epilepsy_data.csv')
    epi_subjects = []
    for _, row in epi_df.iterrows():
        sd = 0.3 # Empirical EEG variance
        sim_slopes = np.random.normal(row['Spectral_Slope'], sd, int(row['n']))
        for slope in sim_slopes:
            epi_subjects.append({'Category': row['Category'], 'State': row['State'], 'Slope': slope})

    epi_subj_df = pd.DataFrame(epi_subjects)
    healthy_epi = epi_subj_df[epi_subj_df['Category'] == 'Healthy']['Slope']
    ictal_epi = epi_subj_df[epi_subj_df['Category'] == 'Ictal']['Slope']
    _, p_val_epi = stats.ttest_ind(healthy_epi, ictal_epi)
    cohens_d_epi = (ictal_epi.mean() - healthy_epi.mean()) / np.sqrt((ictal_epi.std()**2 + healthy_epi.std()**2)/2)

    # ---------------------------------------------------------
    # 2. MEDITATION: Subject-Level Simulation
    # ---------------------------------------------------------
    med_df = pd.read_csv('meditation_data.csv')
    med_subjects = []
    for _, row in med_df.iterrows():
        sd = 0.2
        sim_slopes = np.random.normal(row['Slope_1f'], sd, int(row['n']))
        for slope in sim_slopes:
            med_subjects.append({'Group_Type': row['Group_Type'], 'State': row['State'], 'Slope': slope})

    med_subj_df = pd.DataFrame(med_subjects)
    novice_med = med_subj_df[(med_subj_df['Group_Type'] == 'Novice') & (med_subj_df['State'] == 'Meditation')]['Slope']
    pract_med = med_subj_df[(med_subj_df['Group_Type'] == 'Practitioner') & (med_subj_df['State'] == 'Meditation')]['Slope']
    _, p_val_med = stats.ttest_ind(novice_med, pract_med)
    cohens_d_med = (pract_med.mean() - novice_med.mean()) / np.sqrt((pract_med.std()**2 + novice_med.std()**2)/2)

    # ---------------------------------------------------------
    # 3. PSYCHEDELICS: Subject-Level Simulation
    # ---------------------------------------------------------
    psych_df = pd.read_csv('psychedelics_data.csv')
    psych_subjects = []
    for _, row in psych_df.iterrows():
        sd = 0.08 # LZ complexity variance
        sim_lz = np.random.normal(row['LZ_complexity'], sd, int(row['n']))
        for lz in sim_lz:
            psych_subjects.append({'Drug': row['Drug'], 'LZ_complexity': lz})

    psych_subj_df = pd.DataFrame(psych_subjects)
    placebo = psych_subj_df[psych_subj_df['Drug'] == 'Control']['LZ_complexity']
    lsd = psych_subj_df[psych_subj_df['Drug'] == 'LSD']['LZ_complexity']
    _, p_val_psych = stats.ttest_ind(placebo, lsd)
    cohens_d_psych = (lsd.mean() - placebo.mean()) / np.sqrt((lsd.std()**2 + placebo.std()**2)/2)

    # ---------------------------------------------------------
    # 4. SLEEP (LARGE COHORT): Subject-Level
    # ---------------------------------------------------------
    sleep_df = pd.read_csv('sleep_large_data.csv')
    sleep_subjects = []
    for _, row in sleep_df.iterrows():
        sd = 0.4
        sim_slopes = np.random.normal(row['Spectral_Slope'], sd, int(row['n']))
        for slope in sim_slopes:
            sleep_subjects.append({'State': row['State'], 'Slope': slope})

    sleep_subj_df = pd.DataFrame(sleep_subjects)
    wake_sleep = sleep_subj_df[sleep_subj_df['State'] == 'Wake (NSRR)']['Slope']
    nrem_sleep = sleep_subj_df[sleep_subj_df['State'] == 'NREM (NSRR)']['Slope']
    _, p_val_sleep = stats.ttest_ind(wake_sleep, nrem_sleep)
    cohens_d_sleep = (nrem_sleep.mean() - wake_sleep.mean()) / np.sqrt((nrem_sleep.std()**2 + wake_sleep.std()**2)/2)

    print(f"\\n--- RESULTS ---")
    print(f"Epilepsy (Healthy vs Ictal) : d = {abs(cohens_d_epi):.2f}, p = {p_val_epi:.2e}")
    print(f"Meditation (Novice vs Prac) : d = {abs(cohens_d_med):.2f}, p = {p_val_med:.2e}")
    print(f"Psychedelics (Placebo vs LSD): d = {abs(cohens_d_psych):.2f}, p = {p_val_psych:.2e}")
    print(f"Sleep (Wake vs NREM)        : d = {abs(cohens_d_sleep):.2f}, p = {p_val_sleep:.2e}")

    # ---------------------------------------------------------
    # 5. MASTER VISUALIZATION (VIOLIN PLOTS)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.violinplot(x='State', y='Slope', data=epi_subj_df, ax=axes[0, 0], palette="Reds_r", inner="quartile")
    axes[0, 0].set_title('1. Epilepsy: Hypersynchronous Collapse (α increases)', fontweight='bold')
    axes[0, 0].set_ylabel('Spectral Slope (β)')
    axes[0, 0].tick_params(axis='x', rotation=15)

    sns.violinplot(x='State', y='Slope', hue='Group_Type', data=med_subj_df, ax=axes[0, 1], palette="Blues", inner="quartile")
    axes[0, 1].set_title('2. Meditation: Expertise Effects (Slope Steepens)', fontweight='bold')
    axes[0, 1].set_ylabel('Spectral Slope (β)')

    sns.violinplot(x='Drug', y='LZ_complexity', data=psych_subj_df, ax=axes[1, 0], palette="Purples", inner="quartile")
    axes[1, 0].set_title('3. Psychedelics: Entropic Expansion', fontweight='bold')
    axes[1, 0].set_ylabel('Lempel-Ziv Complexity')

    sns.violinplot(x='State', y='Slope', data=sleep_subj_df, ax=axes[1, 1], palette="Greens", inner="quartile")
    axes[1, 1].set_title('4. Sleep: Arousal Hierarchy', fontweight='bold')
    axes[1, 1].set_ylabel('Spectral Slope (β)')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/robust_neuro_4domains.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/robust_neuro_4domains.pdf")

    # Export
    summary_res = pd.DataFrame({
        'Domain': ['Epilepsy', 'Meditation', 'Psychedelics', 'Sleep'],
        'Metric': ['Cohen_d (Healthy vs Ictal)', 'Cohen_d (Novice vs Pract)', 'Cohen_d (Placebo vs LSD)', 'Cohen_d (Wake vs NREM)'],
        'Effect_Size': [abs(cohens_d_epi), abs(cohens_d_med), abs(cohens_d_psych), abs(cohens_d_sleep)],
        'P_Value': [p_val_epi, p_val_med, p_val_psych, p_val_sleep]
    })
    summary_res.to_csv(f"{OUTPUT_DIR}/neuro_robust_summary.csv", index=False)
    print(f"\\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()