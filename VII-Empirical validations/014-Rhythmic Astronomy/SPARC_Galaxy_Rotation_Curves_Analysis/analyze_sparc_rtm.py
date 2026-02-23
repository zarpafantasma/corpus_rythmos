#!/usr/bin/env python3
"""
RTM ANALYSIS ON REAL SPARC DATA
================================

This is the REAL TEST of RTM predictions using observational data
from 175 galaxies in the SPARC database.

WHAT WE TEST:
1. Fit log-log slopes to observed rotation curves
2. Use surface brightness (SBdisk) as structural proxy for α
3. Test if RTM slope prediction (1 - α/2) matches observations
4. Compare with what standard physics predicts

NO MORE SYNTHETIC DATA - THIS IS REAL.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_galaxy(filepath):
    """Load a single SPARC galaxy rotation curve."""
    data = {
        'name': os.path.basename(filepath).replace('_rotmod.dat', ''),
        'distance': None,
        'Rad': [], 'Vobs': [], 'errV': [],
        'Vgas': [], 'Vdisk': [], 'Vbul': [],
        'SBdisk': [], 'SBbul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                try:
                    data['distance'] = float(line.split('=')[1].replace('Mpc', '').strip())
                except:
                    pass
            elif line.startswith('#') or not line:
                continue
            else:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        data['Rad'].append(float(parts[0]))
                        data['Vobs'].append(float(parts[1]))
                        data['errV'].append(float(parts[2]))
                        data['Vgas'].append(float(parts[3]))
                        data['Vdisk'].append(float(parts[4]))
                        data['Vbul'].append(float(parts[5]))
                        data['SBdisk'].append(float(parts[6]))
                        data['SBbul'].append(float(parts[7]))
                    except:
                        pass
    
    # Convert to arrays
    for key in ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']:
        data[key] = np.array(data[key])
    
    return data


def load_all_sparc(data_dir):
    """Load all SPARC galaxies."""
    files = glob(os.path.join(data_dir, '*_rotmod.dat'))
    galaxies = []
    
    for f in files:
        try:
            gal = load_sparc_galaxy(f)
            if len(gal['Rad']) >= 5:  # Need at least 5 points
                galaxies.append(gal)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return galaxies


# =============================================================================
# RTM ANALYSIS
# =============================================================================

def fit_rotation_slope(r, v, v_err=None, r_min=None, r_max=None):
    """
    Fit log-log slope to rotation curve.
    
    RTM predicts: slope = 1 - α/2
    So: α = 2(1 - slope)
    
    slope = 0 → α = 2 (flat curve)
    slope > 0 → α < 2 (rising curve)
    slope < 0 → α > 2 (declining curve)
    """
    # Filter by radius if specified
    mask = np.ones(len(r), dtype=bool)
    if r_min is not None:
        mask &= (r >= r_min)
    if r_max is not None:
        mask &= (r <= r_max)
    
    r_fit = r[mask]
    v_fit = v[mask]
    
    if len(r_fit) < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    # Remove zeros/negatives
    valid = (r_fit > 0) & (v_fit > 0)
    r_fit = r_fit[valid]
    v_fit = v_fit[valid]
    
    if len(r_fit) < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    log_r = np.log10(r_fit)
    log_v = np.log10(v_fit)
    
    # Robust fit (Theil-Sen)
    n = len(log_r)
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            if log_r[j] != log_r[i]:
                slopes.append((log_v[j] - log_v[i]) / (log_r[j] - log_r[i]))
    
    if len(slopes) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    slope = np.median(slopes)
    slope_err = 1.4826 * np.median(np.abs(np.array(slopes) - slope)) / np.sqrt(len(slopes))
    
    # R² 
    intercept = np.median(log_v - slope * log_r)
    residuals = log_v - (slope * log_r + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_v - np.mean(log_v))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Convert to α
    alpha = 2 * (1 - slope)
    
    return slope, slope_err, alpha, r2


def compute_structure_proxy(gal, method='sb_gradient'):
    """
    Compute structural proxy for α from surface brightness.
    
    RTM hypothesis: more structured regions → higher α
    
    Proxies:
    1. SB gradient: steep SB decline → more concentrated → higher α
    2. Mean SB: higher SB → more structured → higher α
    3. SB curvature: more curvature → more structure
    """
    r = gal['Rad']
    sb = gal['SBdisk']
    
    # Filter valid SB values
    valid = (sb > 0) & (r > 0)
    if np.sum(valid) < 3:
        return np.nan
    
    r_v = r[valid]
    sb_v = sb[valid]
    log_sb = np.log10(sb_v)
    log_r = np.log10(r_v)
    
    if method == 'sb_gradient':
        # Steeper decline → higher structure → predict higher α
        slope, _, _, _, _ = stats.linregress(log_r, log_sb)
        # Normalize: typical slope is -1 to -3
        proxy = -slope / 2  # So proxy ~ 0.5 to 1.5
        
    elif method == 'mean_sb':
        # Higher mean SB → more structure
        proxy = np.mean(log_sb) / 3  # Normalize to ~0.5-1.5 range
        
    elif method == 'concentration':
        # Ratio of inner to outer SB
        r_half = np.median(r_v)
        inner = np.mean(sb_v[r_v < r_half])
        outer = np.mean(sb_v[r_v >= r_half])
        if outer > 0:
            proxy = np.log10(inner / outer)
        else:
            proxy = np.nan
    else:
        proxy = np.nan
    
    return proxy


def analyze_galaxy_rtm(gal):
    """
    Full RTM analysis for a single galaxy.
    
    Returns dict with slopes, α estimates, and structure proxies.
    """
    r = gal['Rad']
    v = gal['Vobs']
    v_err = gal['errV']
    
    # Overall slope (full curve)
    slope_full, slope_err, alpha_full, r2_full = fit_rotation_slope(r, v)
    
    # Inner region (r < median)
    r_med = np.median(r)
    slope_inner, _, alpha_inner, r2_inner = fit_rotation_slope(r, v, r_max=r_med)
    
    # Outer region (r > median)
    slope_outer, _, alpha_outer, r2_outer = fit_rotation_slope(r, v, r_min=r_med)
    
    # Structure proxy
    struct_proxy = compute_structure_proxy(gal, method='sb_gradient')
    
    # Baryonic contribution ratio
    v_bar = np.sqrt(gal['Vgas']**2 + gal['Vdisk']**2 + gal['Vbul']**2)
    if np.mean(v) > 0:
        bar_ratio = np.mean(v_bar) / np.mean(v)
    else:
        bar_ratio = np.nan
    
    return {
        'name': gal['name'],
        'n_points': len(r),
        'r_max': r.max() if len(r) > 0 else np.nan,
        'v_max': v.max() if len(v) > 0 else np.nan,
        'slope_full': slope_full,
        'slope_err': slope_err,
        'alpha_full': alpha_full,
        'r2_full': r2_full,
        'slope_inner': slope_inner,
        'alpha_inner': alpha_inner,
        'slope_outer': slope_outer,
        'alpha_outer': alpha_outer,
        'struct_proxy': struct_proxy,
        'bar_ratio': bar_ratio
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("RTM ANALYSIS ON REAL SPARC DATA")
    print("175 Galaxies - Observational Rotation Curves")
    print("=" * 70)
    
    # Load data
    data_dir = "/home/claude/sparc_data"
    output_dir = "/home/claude/sparc_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading SPARC data...")
    galaxies = load_all_sparc(data_dir)
    print(f"Loaded {len(galaxies)} galaxies with sufficient data points")
    
    # Analyze all galaxies
    print("\nAnalyzing rotation curves...")
    results = []
    for gal in galaxies:
        res = analyze_galaxy_rtm(gal)
        results.append(res)
    
    df = pd.DataFrame(results)
    
    # Filter valid results
    df_valid = df.dropna(subset=['slope_full', 'alpha_full'])
    print(f"Valid analyses: {len(df_valid)} galaxies")
    
    # ===================
    # KEY STATISTICS
    # ===================
    print("\n" + "=" * 70)
    print("KEY RESULTS - WHAT THE DATA ACTUALLY SHOWS")
    print("=" * 70)
    
    print(f"""
    OBSERVED SLOPES (d log v / d log r):
    ------------------------------------
    Mean:   {df_valid['slope_full'].mean():.4f}
    Median: {df_valid['slope_full'].median():.4f}
    Std:    {df_valid['slope_full'].std():.4f}
    Range:  [{df_valid['slope_full'].min():.3f}, {df_valid['slope_full'].max():.3f}]
    
    DERIVED α VALUES (α = 2(1 - slope)):
    ------------------------------------
    Mean:   {df_valid['alpha_full'].mean():.3f}
    Median: {df_valid['alpha_full'].median():.3f}
    Std:    {df_valid['alpha_full'].std():.3f}
    Range:  [{df_valid['alpha_full'].min():.2f}, {df_valid['alpha_full'].max():.2f}]
    """)
    
    # RTM prediction check
    print("=" * 70)
    print("RTM PREDICTION CHECK")
    print("=" * 70)
    
    # For flat curves, RTM predicts α ≈ 2
    flat_threshold = 0.1  # |slope| < 0.1 is "flat"
    flat_curves = df_valid[np.abs(df_valid['slope_full']) < flat_threshold]
    rising_curves = df_valid[df_valid['slope_full'] > flat_threshold]
    declining_curves = df_valid[df_valid['slope_full'] < -flat_threshold]
    
    print(f"""
    CURVE CLASSIFICATION:
    ---------------------
    Flat (|slope| < {flat_threshold}):     {len(flat_curves):3d} galaxies ({100*len(flat_curves)/len(df_valid):.1f}%)
    Rising (slope > {flat_threshold}):     {len(rising_curves):3d} galaxies ({100*len(rising_curves)/len(df_valid):.1f}%)
    Declining (slope < -{flat_threshold}): {len(declining_curves):3d} galaxies ({100*len(declining_curves)/len(df_valid):.1f}%)
    
    RTM PREDICTION:
    ---------------
    Flat curves should have α ≈ 2.0
    Observed α for flat curves: {flat_curves['alpha_full'].mean():.3f} ± {flat_curves['alpha_full'].std():.3f}
    
    Rising curves should have α < 2.0
    Observed α for rising curves: {rising_curves['alpha_full'].mean():.3f} ± {rising_curves['alpha_full'].std():.3f}
    """)
    
    # Inner vs Outer comparison
    print("=" * 70)
    print("INNER vs OUTER SLOPES")
    print("=" * 70)
    
    df_inner_outer = df_valid.dropna(subset=['slope_inner', 'slope_outer'])
    
    print(f"""
    Inner regions (r < r_median):
      Mean slope: {df_inner_outer['slope_inner'].mean():.4f}
      Mean α:     {df_inner_outer['alpha_inner'].mean():.3f}
    
    Outer regions (r > r_median):
      Mean slope: {df_inner_outer['slope_outer'].mean():.4f}
      Mean α:     {df_inner_outer['alpha_outer'].mean():.3f}
    
    Difference (inner - outer):
      Δslope: {(df_inner_outer['slope_inner'] - df_inner_outer['slope_outer']).mean():.4f}
      Δα:     {(df_inner_outer['alpha_inner'] - df_inner_outer['alpha_outer']).mean():.3f}
    
    RTM PREDICTION: Inner regions (more structure) should have higher α
    """)
    
    # Structure-slope correlation
    print("=" * 70)
    print("STRUCTURE-SLOPE CORRELATION (THE KEY TEST)")
    print("=" * 70)
    
    df_struct = df_valid.dropna(subset=['struct_proxy'])
    if len(df_struct) > 10:
        r_corr, p_corr = stats.pearsonr(df_struct['struct_proxy'], df_struct['slope_full'])
        r_alpha, p_alpha = stats.pearsonr(df_struct['struct_proxy'], df_struct['alpha_full'])
        
        print(f"""
    Structure proxy (from SB gradient) vs kinematic slope:
      Pearson r = {r_corr:.4f}
      p-value   = {p_corr:.4e}
    
    Structure proxy vs derived α:
      Pearson r = {r_alpha:.4f}
      p-value   = {p_alpha:.4e}
    
    RTM PREDICTION: Structure should correlate POSITIVELY with α
                    (more structure → higher α → lower slope)
    
    RESULT: {"CONSISTENT with RTM" if r_corr < 0 and p_corr < 0.05 else "NEEDS MORE INVESTIGATION"}
        """)
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'sparc_rtm_analysis.csv'), index=False)
    
    # ===================
    # VISUALIZATION
    # ===================
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Slope histogram
    ax1 = axes[0, 0]
    ax1.hist(df_valid['slope_full'], bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='slope=0 (flat)')
    ax1.axvline(x=df_valid['slope_full'].mean(), color='green', linestyle='-', 
                linewidth=2, label=f'mean={df_valid["slope_full"].mean():.3f}')
    ax1.set_xlabel('Log-log slope (d log v / d log r)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Rotation Curve Slopes\n(175 SPARC Galaxies)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: α histogram
    ax2 = axes[0, 1]
    ax2.hist(df_valid['alpha_full'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='α=2 (RTM flat)')
    ax2.axvline(x=df_valid['alpha_full'].mean(), color='green', linestyle='-',
                linewidth=2, label=f'mean={df_valid["alpha_full"].mean():.2f}')
    ax2.set_xlabel('RTM α = 2(1 - slope)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Derived α Values', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Inner vs Outer slopes
    ax3 = axes[0, 2]
    ax3.scatter(df_inner_outer['slope_outer'], df_inner_outer['slope_inner'], 
                s=20, alpha=0.6)
    ax3.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', label='1:1 line')
    ax3.set_xlabel('Outer slope', fontsize=11)
    ax3.set_ylabel('Inner slope', fontsize=11)
    ax3.set_title('Inner vs Outer Rotation Slopes', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Structure vs Slope
    ax4 = axes[1, 0]
    valid_struct = df_struct.dropna(subset=['struct_proxy', 'slope_full'])
    ax4.scatter(valid_struct['struct_proxy'], valid_struct['slope_full'], s=20, alpha=0.6)
    if len(valid_struct) > 5:
        z = np.polyfit(valid_struct['struct_proxy'], valid_struct['slope_full'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_struct['struct_proxy'].min(), valid_struct['struct_proxy'].max(), 50)
        ax4.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Fit: slope={z[0]:.3f}')
    ax4.set_xlabel('Structure Proxy (SB gradient)', fontsize=11)
    ax4.set_ylabel('Rotation Slope', fontsize=11)
    ax4.set_title('Structure vs Kinematic Slope\n(RTM Key Prediction)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity vs slope
    ax5 = axes[1, 1]
    ax5.scatter(df_valid['v_max'], df_valid['slope_full'], s=20, alpha=0.6, c='green')
    ax5.axhline(y=0, color='red', linestyle='--')
    ax5.set_xlabel('V_max (km/s)', fontsize=11)
    ax5.set_ylabel('Rotation Slope', fontsize=11)
    ax5.set_title('Maximum Velocity vs Slope', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Example rotation curves
    ax6 = axes[1, 2]
    
    # Pick 3 representative galaxies
    examples = ['NGC2403', 'NGC3198', 'DDO154']
    colors = ['blue', 'red', 'green']
    
    for gal, color in zip(galaxies[:3], colors):
        r = gal['Rad']
        v = gal['Vobs']
        ax6.plot(r, v, 'o-', color=color, markersize=3, alpha=0.7, label=gal['name'])
    
    ax6.set_xlabel('Radius (kpc)', fontsize=11)
    ax6.set_ylabel('V_obs (km/s)', fontsize=11)
    ax6.set_title('Example SPARC Rotation Curves', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparc_rtm_analysis.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sparc_rtm_analysis.pdf'))
    plt.close()
    
    # ===================
    # FINAL VERDICT
    # ===================
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    
    mean_slope = df_valid['slope_full'].mean()
    mean_alpha = df_valid['alpha_full'].mean()
    pct_flat = 100 * len(flat_curves) / len(df_valid)
    
    print(f"""
    OBSERVATIONS:
    1. Mean slope = {mean_slope:.4f} (close to 0 = flat curves dominate)
    2. Mean α = {mean_alpha:.3f} (RTM predicts ~2 for flat curves)
    3. {pct_flat:.1f}% of galaxies have approximately flat curves
    
    RTM CONSISTENCY CHECK:
    """)
    
    # Check 1: Do most galaxies have α ≈ 2?
    alpha_near_2 = np.sum(np.abs(df_valid['alpha_full'] - 2) < 0.5) / len(df_valid) * 100
    print(f"    - Galaxies with α ∈ [1.5, 2.5]: {alpha_near_2:.1f}%")
    
    # Check 2: Is mean α close to 2?
    alpha_diff = abs(mean_alpha - 2)
    print(f"    - Mean α deviation from 2: {alpha_diff:.3f}")
    
    # Check 3: Structure correlation
    if len(df_struct) > 10:
        print(f"    - Structure-slope correlation: r={r_corr:.3f}, p={p_corr:.3e}")
    
    print(f"""
    INTERPRETATION:
    ---------------
    The observed rotation curves are {"CONSISTENT" if alpha_near_2 > 60 else "PARTIALLY CONSISTENT"} 
    with RTM's prediction that α ≈ 2 produces flat rotation curves.
    
    HOWEVER, this is also consistent with dark matter halos!
    The key discriminant is whether α correlates with baryonic structure.
    
    Structure-slope correlation: {"DETECTED" if p_corr < 0.05 else "NOT SIGNIFICANT"}
    
    MORE WORK NEEDED:
    - Better structure proxies (multi-scale entropy, Fourier modes)
    - Bin-by-bin analysis within galaxies
    - Comparison with DM halo predictions for the same correlation
    """)
    
    # Summary file
    summary = f"""SPARC RTM Analysis Summary
==========================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA
----
Galaxies analyzed: {len(df_valid)}
Total data points: {df_valid['n_points'].sum()}

SLOPE STATISTICS
----------------
Mean slope:   {mean_slope:.4f}
Median slope: {df_valid['slope_full'].median():.4f}
Std slope:    {df_valid['slope_full'].std():.4f}

ALPHA STATISTICS  
----------------
Mean α:   {mean_alpha:.3f}
Median α: {df_valid['alpha_full'].median():.3f}
Std α:    {df_valid['alpha_full'].std():.3f}

CURVE TYPES
-----------
Flat:      {len(flat_curves)} ({100*len(flat_curves)/len(df_valid):.1f}%)
Rising:    {len(rising_curves)} ({100*len(rising_curves)/len(df_valid):.1f}%)
Declining: {len(declining_curves)} ({100*len(declining_curves)/len(df_valid):.1f}%)

RTM CONSISTENCY
---------------
Galaxies with α ∈ [1.5, 2.5]: {alpha_near_2:.1f}%
Mean α deviation from 2: {alpha_diff:.3f}

STRUCTURE CORRELATION
--------------------
r = {r_corr:.4f}
p = {p_corr:.4e}
"""
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    df = main()
