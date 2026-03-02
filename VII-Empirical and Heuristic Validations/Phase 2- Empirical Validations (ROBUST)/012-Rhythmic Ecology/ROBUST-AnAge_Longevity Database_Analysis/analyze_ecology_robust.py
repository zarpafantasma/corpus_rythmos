#!usrbinenv python3

ROBUST RTM ECOLOGY ANALYSIS LONGEVITY SCALING
==============================================
Phase 2 Red Team ODR Pipeline

This script corrects the attenuation bias present in the V1 analysis. 
By applying Orthogonal Distance Regression (ODR), it explicitly models the 
massive intra-species variance in body mass (~20%) and the observational 
uncertainty of extreme-value statistics like maximum lifespan (~25%).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
import os
import warnings

warnings.filterwarnings('ignore')
OUTPUT_DIR = output_ecology_robust

def main()
    print(=  60)
    print(ROBUST RTM ECOLOGY ANALYSIS (ODR & BIOLOGICAL VARIANCE))
    print(=  60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    try
        df = pd.read_csv('anage_data.txt', sep='t', encoding='latin-1')
    except
        df = pd.read_csv('anage_data.txt', sep='t', encoding='utf-8')
    
    # Filter Valid Data
    df_clean = df.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)'])
    df_clean = df_clean[(df_clean['Body mass (g)']  0) & (df_clean['Maximum longevity (yrs)']  0)]
    
    df_clean['log_Mass'] = np.log10(df_clean['Body mass (g)'])
    df_clean['log_Longevity'] = np.log10(df_clean['Maximum longevity (yrs)'])
    
    # 2. Setup Robust ODR Parameters
    # Realistic biological variances converted to log-scale errors d(log10(x)) ≈ fractional_error  ln(10)
    log_M_err = 0.20  np.log(10) # 20% intra-species mass variance
    log_L_err = 0.25  np.log(10) # 25% observation uncertainty in max lifespans

    def linear_func(p, x) return p[0]  x + p[1]
    model = Model(linear_func)
    
    classes = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']
    colors = {'Mammalia' 'crimson', 'Aves' 'royalblue', 'Reptilia' 'forestgreen', 'Amphibia' 'darkorange'}
    
    results = []
    
    # 3. Visualization Setup
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df_clean['log_Mass'], df_clean['log_Longevity'], alpha=0.05, color='gray', label='All Species (Background)')
    
    print(fnTotal Species Analyzed {len(df_clean)})
    print(f{'Class'12} {'N'6} {'Flawed_OLS_α'15} {'Robust_ODR_α'15} {'R²'6})
    print(-  55)

    for cls in classes
        subset = df_clean[df_clean['Class'] == cls]
        if len(subset)  10
            # Baseline OLS
            ols_slope, ols_int, r, p, err = stats.linregress(subset['log_Mass'], subset['log_Longevity'])
            
            # Robust ODR
            data = RealData(subset['log_Mass'], subset['log_Longevity'], sx=log_M_err, sy=log_L_err)
            odr = ODR(data, model, beta0=[ols_slope, ols_int])
            out = odr.run()
            odr_slope, odr_int = out.beta
            odr_err = out.sd_beta[0]
            
            results.append({
                'Class' cls, 'N' len(subset), 
                'Flawed_OLS_Alpha' ols_slope, 
                'Robust_ODR_Alpha' odr_slope, 
                'ODR_Alpha_Error' odr_err, 'R_Squared' r2
            })
            
            print(f{cls12} {len(subset)6} {ols_slope15.3f} {odr_slope.3f} ± {odr_err.3f}   {r2.2f})
            
            # Plot Robust Line
            x_range = np.linspace(subset['log_Mass'].min(), subset['log_Mass'].max(), 100)
            y_pred = odr_slope  x_range + odr_int
            ax.plot(x_range, y_pred, color=colors.get(cls, 'black'), linewidth=3, 
                    label=f{cls} (ODR α={odr_slope.2f}))

    # Add theoretical Kleiber limit reference
    x_theory = np.linspace(0, 7, 100)
    y_theory = 0.25  x_theory + 0.3 # arbitrary intercept for visual reference
    ax.plot(x_theory, y_theory, 'k--', linewidth=2, label='Theoretical Optima (α=0.25)')

    ax.set_xlabel('Log10 Body Mass (g)', fontsize=12)
    ax.set_ylabel('Log10 Maximum Longevity (yrs)', fontsize=12)
    ax.set_title('Robust RTM Ecology The Biological ClocknCorrecting Attenuation Bias with ODR', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f{OUTPUT_DIR}robust_ecology_rtm.png, dpi=300)
    plt.savefig(f{OUTPUT_DIR}robust_ecology_rtm.pdf)
    
    # 4. Export
    res_df = pd.DataFrame(results)
    res_df.to_csv(f{OUTPUT_DIR}ecology_robust_summary.csv, index=False)
    print(fnFiles saved to {OUTPUT_DIR})

if __name__ == __main__
    main()