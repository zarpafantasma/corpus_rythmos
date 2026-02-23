#!/usr/bin/env python3
"""
S2: Enzyme Activity Scaling with Confinement
=============================================

RTM Estimator: α_enz = -d(log k_app) / d(log L)

This simulation validates:
1. Methodology for estimating α from confinement data
2. Robustness to noise and sample size
3. Data collapse test: k_app * L^α should collapse to constant
4. Discrimination between transport classes

Confinement methods simulated:
- Nanoporous matrices (AAM, silica)
- Polymer crowding (PEG, dextran)
- Engineered cavities

METHODOLOGY VALIDATION - requires experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# RTM ENZYME MODEL
# =============================================================================

def kapp_rtm(L, kapp_0, alpha, L_ref=100):
    """
    RTM scaling for apparent rate constant.
    k_app(L) = k_app,0 * (L/L_ref)^(-α)
    """
    return kapp_0 * (L / L_ref) ** (-alpha)


def generate_confinement_data(alpha_true, L_values, kapp_0=100, L_ref=100,
                               noise_level=0.1, n_replicates=3):
    """
    Generate synthetic k_app measurements across confinement scales.
    
    Simulates experimental variability with log-normal noise.
    """
    all_data = []
    
    for L in L_values:
        kapp_true = kapp_rtm(L, kapp_0, alpha_true, L_ref)
        
        for rep in range(n_replicates):
            # Log-normal noise (multiplicative)
            noise = np.exp(noise_level * np.random.randn())
            kapp_measured = kapp_true * noise
            
            all_data.append({
                'L': L,
                'kapp': kapp_measured,
                'kapp_true': kapp_true,
                'replicate': rep + 1
            })
    
    return pd.DataFrame(all_data)


def estimate_alpha(L_values, kapp_values, method='ols'):
    """
    Estimate α from log-log regression.
    
    Returns α, standard error, R², and p-value.
    """
    log_L = np.log(L_values)
    log_kapp = np.log(kapp_values)
    
    if method == 'ols':
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_kapp)
        
    elif method == 'theil_sen':
        # Robust estimation
        n = len(log_L)
        slopes = []
        for i in range(n):
            for j in range(i+1, n):
                if log_L[j] != log_L[i]:
                    slopes.append((log_kapp[j] - log_kapp[i]) / (log_L[j] - log_L[i]))
        slope = np.median(slopes)
        intercept = np.median(log_kapp - slope * log_L)
        
        # Compute R²
        predicted = slope * log_L + intercept
        ss_res = np.sum((log_kapp - predicted)**2)
        ss_tot = np.sum((log_kapp - np.mean(log_kapp))**2)
        r_value = np.sqrt(1 - ss_res/ss_tot) if ss_tot > 0 else 0
        
        # MAD-based std error
        slope_mad = 1.4826 * np.median(np.abs(np.array(slopes) - slope))
        std_err = slope_mad / np.sqrt(len(slopes))
        p_value = None
    
    return {
        'alpha': -slope,  # α = -slope because k_app ∝ L^(-α)
        'intercept': intercept,
        'r_squared': r_value**2,
        'std_err': std_err,
        'p_value': p_value
    }


def collapse_test(L_values, kapp_values, alpha):
    """
    Data collapse test: k_app * L^α should be constant.
    
    Returns coefficient of variation of collapsed data.
    """
    collapsed = kapp_values * (L_values ** alpha)
    cv = np.std(collapsed) / np.mean(collapsed)
    return cv, collapsed


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_noise_robustness(alpha_true=2.2, n_trials=100):
    """Test estimation accuracy across noise levels."""
    noise_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    L_values = np.array([10, 20, 40, 80, 160])
    
    results = []
    
    for noise in noise_levels:
        errors = []
        for _ in range(n_trials):
            df = generate_confinement_data(alpha_true, L_values, noise_level=noise, n_replicates=1)
            result = estimate_alpha(df['L'].values, df['kapp'].values)
            errors.append(result['alpha'] - alpha_true)
        
        results.append({
            'noise_level': noise,
            'mae': np.mean(np.abs(errors)),
            'bias': np.mean(errors),
            'std': np.std(errors)
        })
    
    return pd.DataFrame(results)


def test_sample_size(alpha_true=2.2, noise=0.10, n_trials=100):
    """Test how number of L values affects accuracy."""
    sample_sizes = [3, 4, 5, 7, 10, 15]
    
    results = []
    
    for n in sample_sizes:
        L_values = np.logspace(1, 2.2, n)  # 10 to ~160 nm
        errors = []
        
        for _ in range(n_trials):
            df = generate_confinement_data(alpha_true, L_values, noise_level=noise, n_replicates=1)
            result = estimate_alpha(df['L'].values, df['kapp'].values)
            errors.append(result['alpha'] - alpha_true)
        
        results.append({
            'n_scales': n,
            'mae': np.mean(np.abs(errors)),
            'std': np.std(errors)
        })
    
    return pd.DataFrame(results)


def test_transport_class_discrimination(n_trials=200):
    """
    Test if different transport classes can be discriminated.
    
    Classes:
    - Diffusive: α ≈ 2.0
    - Hierarchical: α ≈ 2.3
    - Guided: α ≈ 1.7
    """
    classes = {
        'diffusive': {'alpha_mean': 2.0, 'alpha_std': 0.1},
        'hierarchical': {'alpha_mean': 2.3, 'alpha_std': 0.1},
        'guided': {'alpha_mean': 1.7, 'alpha_std': 0.1}
    }
    
    L_values = np.array([10, 20, 40, 80, 160])
    
    results = []
    
    for class_name, params in classes.items():
        for _ in range(n_trials):
            alpha_true = params['alpha_mean'] + params['alpha_std'] * np.random.randn()
            alpha_true = np.clip(alpha_true, 1.0, 3.0)
            
            df = generate_confinement_data(alpha_true, L_values, noise_level=0.10)
            
            # Average replicates
            df_mean = df.groupby('L')['kapp'].mean().reset_index()
            result = estimate_alpha(df_mean['L'].values, df_mean['kapp'].values)
            
            results.append({
                'class': class_name,
                'alpha_true': alpha_true,
                'alpha_estimated': result['alpha'],
                'r_squared': result['r_squared']
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S2: Enzyme Activity Scaling with Confinement")
    print("=" * 70)
    
    output_dir = "/home/claude/012-Rhythmic_Biochemistry/S2_confinement_scaling/output"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ===================
    # Part 1: Demonstrate RTM scaling
    # ===================
    print("\n1. Demonstrating RTM confinement scaling...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Different enzyme transport classes
    enzymes = {
        'Urease (hierarchical)': {'alpha': 2.3, 'color': 'blue'},
        'LDH (diffusive)': {'alpha': 2.0, 'color': 'green'},
        'Carbonic anhydrase (guided)': {'alpha': 1.7, 'color': 'orange'}
    }
    
    L_range = np.logspace(0.7, 2.5, 50)  # 5 to 300 nm
    kapp_0 = 100  # s^-1
    
    ax1 = axes1[0, 0]
    for enzyme, params in enzymes.items():
        kapp = kapp_rtm(L_range, kapp_0, params['alpha'])
        ax1.plot(L_range, kapp, linewidth=2, color=params['color'], 
                 label=f'{enzyme}: α={params["alpha"]}')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax1.set_ylabel('k_app (s⁻¹)', fontsize=11)
    ax1.set_title('RTM Prediction: k_app ∝ L^(-α)\nDifferent Transport Classes', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    
    # ===================
    # Part 2: Simulated experimental data
    # ===================
    
    ax2 = axes1[0, 1]
    
    L_exp = np.array([10, 20, 40, 80, 160])  # Typical experimental points
    alpha_true = 2.2
    
    df_exp = generate_confinement_data(alpha_true, L_exp, noise_level=0.10, n_replicates=5)
    
    # Plot individual replicates
    ax2.scatter(df_exp['L'], df_exp['kapp'], s=40, alpha=0.5, c='blue', label='Replicates')
    
    # Plot means with error bars
    df_mean = df_exp.groupby('L')['kapp'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(df_mean['L'], df_mean['mean'], yerr=df_mean['std'], 
                 fmt='ko', markersize=8, capsize=5, label='Mean ± SD')
    
    # Fit
    result = estimate_alpha(df_mean['L'].values, df_mean['mean'].values)
    L_fit = np.logspace(np.log10(L_exp.min()), np.log10(L_exp.max()), 50)
    kapp_fit = np.exp(result['intercept']) * L_fit ** (-result['alpha'])
    ax2.plot(L_fit, kapp_fit, 'r--', linewidth=2, 
             label=f'Fit: α = {result["alpha"]:.2f} (R² = {result["r_squared"]:.3f})')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Confinement Length L (nm)', fontsize=11)
    ax2.set_ylabel('k_app (s⁻¹)', fontsize=11)
    ax2.set_title(f'Simulated Experiment (α_true = {alpha_true})\nα_recovered = {result["alpha"]:.2f}', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # ===================
    # Part 3: Data collapse test
    # ===================
    
    ax3 = axes1[1, 0]
    
    # Test collapse with correct vs incorrect α
    cv_correct, collapsed_correct = collapse_test(df_mean['L'].values, df_mean['mean'].values, 
                                                    result['alpha'])
    cv_wrong, collapsed_wrong = collapse_test(df_mean['L'].values, df_mean['mean'].values, 
                                               result['alpha'] + 0.5)
    
    ax3.bar(['Correct α\n(fitted)', 'Wrong α\n(+0.5)'], [cv_correct, cv_wrong], 
            color=['green', 'red'], alpha=0.7)
    ax3.axhline(y=0.1, color='gray', linestyle='--', label='10% CV threshold')
    ax3.set_ylabel('Coefficient of Variation', fontsize=11)
    ax3.set_title('Data Collapse Test: k_app × L^α = const?\n(Lower CV = better collapse)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Annotate
    ax3.annotate(f'CV = {cv_correct:.3f}', xy=(0, cv_correct), xytext=(0, cv_correct + 0.05),
                 ha='center', fontsize=10)
    ax3.annotate(f'CV = {cv_wrong:.3f}', xy=(1, cv_wrong), xytext=(1, cv_wrong + 0.05),
                 ha='center', fontsize=10)
    
    # ===================
    # Part 4: Multiple confinement methods
    # ===================
    
    ax4 = axes1[1, 1]
    
    methods = {
        'Nanopores (AAM)': {'L': [20, 50, 100, 200], 'offset': 1.0},
        'Silica (MCM-41)': {'L': [3, 5, 8, 12], 'offset': 1.2},
        'PEG crowding': {'L': [15, 30, 60, 120], 'offset': 0.9}
    }
    
    alpha_universal = 2.2
    
    for method, params in methods.items():
        L_vals = np.array(params['L'])
        kapp_vals = kapp_rtm(L_vals, kapp_0 * params['offset'], alpha_universal)
        kapp_noisy = kapp_vals * np.exp(0.08 * np.random.randn(len(L_vals)))
        
        ax4.scatter(L_vals, kapp_noisy, s=60, label=method)
    
    # Universal fit line
    L_all = np.logspace(0.3, 2.5, 50)
    ax4.plot(L_all, kapp_rtm(L_all, kapp_0, alpha_universal), 'k--', linewidth=2,
             label=f'Universal: α = {alpha_universal}')
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Effective Confinement L (nm)', fontsize=11)
    ax4.set_ylabel('k_app (s⁻¹)', fontsize=11)
    ax4.set_title('Multiple Confinement Methods\n(Should yield same α)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_confinement_scaling.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_confinement_scaling.pdf'))
    plt.close()
    
    # ===================
    # Validation tests
    # ===================
    
    print("\n2. Running validation tests...")
    
    df_noise = test_noise_robustness()
    df_noise.to_csv(os.path.join(output_dir, 'S2_noise_robustness.csv'), index=False)
    
    df_samples = test_sample_size()
    df_samples.to_csv(os.path.join(output_dir, 'S2_sample_size.csv'), index=False)
    
    print("3. Testing transport class discrimination...")
    df_classes = test_transport_class_discrimination()
    df_classes.to_csv(os.path.join(output_dir, 'S2_class_discrimination.csv'), index=False)
    
    # ===================
    # Validation figure
    # ===================
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Noise robustness
    ax = axes2[0, 0]
    ax.errorbar(df_noise['noise_level'], df_noise['mae'], yerr=df_noise['std'],
                marker='o', capsize=4, linewidth=2)
    ax.axhline(y=0.1, color='green', linestyle='--', label='Target (0.1)')
    ax.axhline(y=0.2, color='orange', linestyle='--', label='Acceptable (0.2)')
    ax.set_xlabel('Noise Level (log-normal σ)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error in α', fontsize=11)
    ax.set_title('Noise Robustness', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sample size
    ax = axes2[0, 1]
    ax.errorbar(df_samples['n_scales'], df_samples['mae'], yerr=df_samples['std'],
                marker='s', capsize=4, linewidth=2, color='purple')
    ax.axhline(y=0.1, color='green', linestyle='--')
    ax.set_xlabel('Number of Confinement Scales', fontsize=11)
    ax.set_ylabel('Mean Absolute Error in α', fontsize=11)
    ax.set_title('Sample Size Effect', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Class discrimination - histograms
    ax = axes2[1, 0]
    colors = {'diffusive': 'green', 'hierarchical': 'blue', 'guided': 'orange'}
    for cls in df_classes['class'].unique():
        data = df_classes[df_classes['class'] == cls]['alpha_estimated']
        ax.hist(data, bins=20, alpha=0.5, label=cls.title(), color=colors[cls])
    ax.set_xlabel('Estimated α', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Transport Class Discrimination', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Class discrimination - box plot
    ax = axes2[1, 1]
    class_order = ['guided', 'diffusive', 'hierarchical']
    data_box = [df_classes[df_classes['class'] == c]['alpha_estimated'].values for c in class_order]
    bp = ax.boxplot(data_box, labels=[c.title() for c in class_order], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['orange', 'green', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Add true means
    true_alphas = [1.7, 2.0, 2.3]
    for i, alpha in enumerate(true_alphas):
        ax.hlines(alpha, i+0.7, i+1.3, colors='red', linestyles='--', linewidth=2)
    
    ax.set_ylabel('Estimated α', fontsize=11)
    ax.set_title('α by Transport Class\n(Red dashed = true mean)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S2_validation.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S2_validation.pdf'))
    plt.close()
    
    # ===================
    # Statistics
    # ===================
    
    # Class discrimination stats
    diff_hier = df_classes[df_classes['class'] == 'diffusive']['alpha_estimated']
    hier_hier = df_classes[df_classes['class'] == 'hierarchical']['alpha_estimated']
    
    t_stat, p_val = stats.ttest_ind(diff_hier, hier_hier)
    cohens_d = (hier_hier.mean() - diff_hier.mean()) / np.sqrt((hier_hier.std()**2 + diff_hier.std()**2)/2)
    
    max_noise = df_noise[df_noise['mae'] < 0.15]['noise_level'].max()
    min_scales = df_samples[df_samples['mae'] < 0.15]['n_scales'].min()
    
    # ===================
    # Summary
    # ===================
    
    summary = f"""S2: Enzyme Activity Scaling with Confinement
=============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RTM ENZYME SCALING LAW
----------------------
k_app(L) = k_app,0 × (L/L_ref)^(-α)

Estimator: α_enz = -d(log k_app)/d(log L)

TRANSPORT CLASSES
-----------------
Guided/ballistic:    α ≈ 1.5-1.8
Laplacian diffusion: α ≈ 2.0
Hierarchical/fractal: α ≈ 2.1-2.5

VALIDATION RESULTS
------------------

1. NOISE ROBUSTNESS
   Max noise for MAE < 0.15: σ ≈ {max_noise:.2f}
   
   Noise Level    MAE
   ---------------------
"""
    for _, row in df_noise.iterrows():
        summary += f"   {row['noise_level']:.2f}          {row['mae']:.3f}\n"
    
    summary += f"""
2. SAMPLE SIZE
   Min scales for MAE < 0.15: {min_scales}
   
   N Scales    MAE
   ------------------
"""
    for _, row in df_samples.iterrows():
        summary += f"   {int(row['n_scales']):<10} {row['mae']:.3f}\n"
    
    summary += f"""
3. CLASS DISCRIMINATION
   Diffusive vs Hierarchical:
   - t-statistic: {t_stat:.2f}
   - p-value: {p_val:.2e}
   - Cohen's d: {cohens_d:.2f} (large effect)
   
   Class means (estimated):
   - Guided: {df_classes[df_classes['class']=='guided']['alpha_estimated'].mean():.2f}
   - Diffusive: {df_classes[df_classes['class']=='diffusive']['alpha_estimated'].mean():.2f}
   - Hierarchical: {df_classes[df_classes['class']=='hierarchical']['alpha_estimated'].mean():.2f}

DATA COLLAPSE TEST
------------------
CV with correct α: {cv_correct:.3f}
CV with wrong α: {cv_wrong:.3f}
Collapse ratio: {cv_wrong/cv_correct:.1f}x worse

EXPERIMENTAL RECOMMENDATIONS
----------------------------
1. Use ≥{min_scales} confinement scales spanning 1+ decade
2. Include replicates (n≥3) per scale
3. Verify with multiple confinement methods
4. Report collapse test alongside slope
"""
    
    with open(os.path.join(output_dir, 'S2_summary.txt'), 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nNoise robustness: MAE < 0.15 for σ ≤ {max_noise:.2f}")
    print(f"Sample size: Need ≥{min_scales} scales")
    print(f"Class discrimination: Cohen's d = {cohens_d:.2f}")
    print(f"Collapse test: {cv_wrong/cv_correct:.1f}x worse with wrong α")
    print(f"\nOutputs: {output_dir}/")
    
    return df_noise, df_samples, df_classes


if __name__ == "__main__":
    main()
