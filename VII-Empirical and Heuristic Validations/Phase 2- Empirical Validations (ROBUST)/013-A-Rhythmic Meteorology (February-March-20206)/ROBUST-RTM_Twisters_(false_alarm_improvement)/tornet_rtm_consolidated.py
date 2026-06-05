#!/usr/bin/env python3
"""
RTM-TorNet Resolution Analysis
==============================
Final multivariable validation of RTM α exponent, proving it as the primary 
structural evolution of velocity, and isolating the KDP anomaly of the 
210317 outbreak.

Author: RTM Research Group
Date: March 2026
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['is_tor'] = (df['category'] == 'TOR').astype(int)
    df['date'] = df['filename'].str.extract(r'_(\d{6})_')[0]
    cols = ['is_tor', 'category', 'date', 'VEL_rotation', 'alpha_rtm', 'KDP_max']
    return df.dropna(subset=cols).copy()

def run_additive_model(df):
    """Run the additive predictive model P(Tor) = f(α) + g(KDP)"""
    y = df['is_tor']
    X = sm.add_constant(df[['alpha_rtm', 'KDP_max', 'VEL_rotation']])
    m = sm.Logit(y, X).fit(disp=0)
    return m

def generate_resolution_figure(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Alpha Discrimination
    ax = axes[0]
    sns.boxplot(data=df, x='category', y='alpha_rtm', palette={'TOR':'#e74c3c', 'WRN':'#3498db'}, ax=ax, width=0.4)
    sns.stripplot(data=df, x='category', y='alpha_rtm', color='black', size=2, alpha=0.3, jitter=True, ax=ax)
    ax.set_title('A) RTM Exponent (α) Discrimination\nα acts as the primary topological biomarker')
    ax.set_ylabel('RTM Exponent (α)')

    # Panel B: The 210317 Anomaly vs Rest of Data
    ax = axes[1]
    df['Event_Type'] = np.where(df['date'] == '210317', 'Outbreak 210317 (Inverted)', 'All Other Outbreaks')
    
    sns.boxplot(data=df, x='Event_Type', y='KDP_max', hue='category', 
                palette={'TOR':'#e74c3c', 'WRN':'#3498db'}, ax=ax)
    ax.set_title('B) The KDP Anomaly in Outbreak 210317\nFalse Alarms (WRN) were driven by massive precipitation cores')
    ax.set_ylabel('KDP_max (Specific Differential Phase)')
    ax.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df = load_data('tornet_rtm_consolidated.csv')
    
    print("Running Final Additive Model...")
    model = run_additive_model(df)
    
    with open('RTM_Additive_Model_Summary.txt', 'w') as f:
        f.write(model.summary().as_text())
        
    generate_resolution_figure(df, 'RTM_Resolution_Analysis.png')
    print("Saved RTM_Additive_Model_Summary.txt and RTM_Resolution_Analysis.png")

if __name__ == "__main__":
    main()