#!/usr/bin/env python3
"""
S3: Baryonic Tully-Fisher Residuals and α-Correlation
======================================================

From "Rhythmic Astronomy: An RTM Slope Law for Galaxy Rotation Curves"

PURPOSE
-------
This simulation demonstrates the RTM PREDICTION for bTFR residuals:
- Standard bTFR: M_bar ∝ v_flat^4
- RTM predicts: residuals correlate with α-proxies (structure metrics)

WHAT THIS SHOWS:
- How RTM predicts bTFR residuals should behave
- The expected correlation between residuals and coherence

WHAT THIS DOES NOT PROVE:
- That real bTFR residuals correlate with structure
- That α can be measured from baryonic proxies
- This requires comparison with actual galaxy surveys

The discriminant (Section 4.2 of paper):
- DM: residuals correlate with halo parameters (concentration, spin)
- MOND: residuals correlate with acceleration scale
- RTM: residuals correlate with structural coherence

Reference: Paper Sections 4.2, 4.3
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


# =============================================================================
# BARYONIC TULLY-FISHER RELATION
# =============================================================================

def standard_btfr(M_bar, a_0=1.2e-10):
    """
    Standard bTFR: v_flat^4 ∝ M_bar
    
    Using McGaugh et al. (2000) normalization:
    v_flat = (G × M_bar × a_0)^(1/4)
    
    Parameters
    ----------
    M_bar : array
        Baryonic mass (solar masses)
    a_0 : float
        Acceleration scale (m/s²), default ~1.2e-10
    
    Returns
    -------
    v_flat : array
        Flat rotation velocity (km/s)
    """
    G = 4.3e-6  # kpc (km/s)² / M_sun
    
    # v^4 = G × M_bar × a_0
    # Convert a_0 to (km/s)² / kpc
    a_0_kpc = a_0 * 3.086e16 / 1e6  # m/s² to (km/s)²/kpc
    
    v_flat = (G * M_bar * a_0_kpc) ** 0.25
    
    return v_flat


def rtm_corrected_btfr(M_bar, alpha, alpha_ref=2.0):
    """
    RTM-modified bTFR including coherence correction.
    
    v_measured = v_btfr × (r_sample/r_ref)^(1 - α/2)
    
    At the radius where α = 2, this reduces to standard bTFR.
    At radii where α ≠ 2, there's a systematic offset.
    """
    v_btfr = standard_btfr(M_bar)
    
    # RTM correction factor
    # If sampling at r where α ≠ 2, velocity differs from flat prediction
    delta_alpha = alpha - alpha_ref
    
    # Simplified model: correction scales with delta_alpha
    # Full model would depend on sampling radius
    correction = 1 + 0.1 * delta_alpha  # ~10% per unit alpha
    
    return v_btfr * correction


# =============================================================================
# SYNTHETIC GALAXY SAMPLE
# =============================================================================

def generate_galaxy_sample(n_galaxies=200, seed=42):
    """
    Generate a synthetic galaxy sample with:
    - Baryonic masses spanning typical range
    - Structure-based α values
    - Intrinsic scatter
    
    RTM predicts: v_measured/v_btfr depends on α at measurement radius
    """
    np.random.seed(seed)
    
    # Baryonic mass distribution (log-normal, centered on 10^10 M_sun)
    log_M_bar = np.random.normal(10, 0.5, n_galaxies)
    M_bar = 10**log_M_bar
    
    # Structure metrics (proxy for α)
    # Higher structure → higher α
    # Make structure largely INDEPENDENT of mass to test the prediction
    structure_index = np.random.normal(0, 0.5, n_galaxies)
    structure_index = np.clip(structure_index, -1.5, 1.5)
    
    # Map structure to α
    # Range: α ∈ [1.7, 2.4]
    alpha = 2.05 + 0.2 * structure_index
    
    # Standard bTFR velocity (what we'd expect at α=2)
    v_btfr = standard_btfr(M_bar)
    
    # RTM effect: deviation from bTFR depends on (α - 2)
    # v_obs/v_btfr = (r_sample/r_flat)^(1 - α/2)
    # At α = 2: ratio = 1 (standard bTFR)
    # At α > 2: ratio < 1 (velocity lower than bTFR)
    # At α < 2: ratio > 1 (velocity higher than bTFR)
    
    # Assume sampling radius effect: ~5% per unit deviation from α=2
    rtm_correction = 0.08 * (alpha - 2.0)  # in log space
    
    # Log velocity with RTM correction
    log_v_btfr = np.log10(v_btfr)
    log_v_true = log_v_btfr + rtm_correction
    
    # Add observational scatter (0.03 dex typical)
    log_v_obs = log_v_true + np.random.normal(0, 0.03, n_galaxies)
    
    # Compute residuals from standard bTFR
    residuals = log_v_obs - log_v_btfr  # in dex
    
    return pd.DataFrame({
        'M_bar': M_bar,
        'log_M_bar': log_M_bar,
        'structure_index': structure_index,
        'alpha': alpha,
        'v_btfr': v_btfr,
        'log_v_btfr': log_v_btfr,
        'log_v_obs': log_v_obs,
        'v_obs': 10**log_v_obs,
        'residual_dex': residuals
    })


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_residual_correlations(df):
    """
    Analyze correlations between bTFR residuals and various parameters.
    
    RTM predicts: residuals correlate with α (structure proxy)
    DM predicts: residuals correlate with halo parameters
    """
    results = {}
    
    # Correlation with structure/α
    r_struct, p_struct = stats.pearsonr(df['structure_index'], df['residual_dex'])
    r_alpha, p_alpha = stats.pearsonr(df['alpha'], df['residual_dex'])
    
    # Correlation with mass (should be weak after standard bTFR)
    r_mass, p_mass = stats.pearsonr(df['log_M_bar'], df['residual_dex'])
    
    results['structure_corr'] = (r_struct, p_struct)
    results['alpha_corr'] = (r_alpha, p_alpha)
    results['mass_corr'] = (r_mass, p_mass)
    
    return results


def compare_model_predictions(df):
    """
    Compare RTM prediction vs null model (no structure correlation).
    """
    # RTM model: residuals ~ α
    slope_rtm, intercept_rtm, r_rtm, p_rtm, se_rtm = stats.linregress(
        df['alpha'], df['residual_dex']
    )
    
    # Null model: residuals independent of structure
    # Expected: slope ≈ 0, r² ≈ 0
    null_r2 = 0  # By definition
    
    rtm_r2 = r_rtm**2
    
    # F-test for model comparison
    n = len(df)
    f_stat = (rtm_r2 / 1) / ((1 - rtm_r2) / (n - 2))
    p_f = 1 - stats.f.cdf(f_stat, 1, n - 2)
    
    return {
        'rtm_slope': slope_rtm,
        'rtm_intercept': intercept_rtm,
        'rtm_r2': rtm_r2,
        'rtm_p': p_rtm,
        'f_stat': f_stat,
        'p_f': p_f
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(output_dir, df, correlations, model_comparison):
    """Create comprehensive visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ===================
    # Plot 1: Standard bTFR
    # ===================
    ax1 = axes[0, 0]
    
    ax1.scatter(df['log_M_bar'], np.log10(df['v_obs']), s=15, alpha=0.6, 
                c=df['alpha'], cmap='viridis')
    
    # Fit line
    M_range = np.linspace(df['log_M_bar'].min(), df['log_M_bar'].max(), 100)
    v_btfr = standard_btfr(10**M_range)
    ax1.plot(M_range, np.log10(v_btfr), 'r-', linewidth=2, label='Standard bTFR')
    
    ax1.set_xlabel('log(M_bar / M☉)', fontsize=11)
    ax1.set_ylabel('log(v_flat / km/s)', fontsize=11)
    ax1.set_title('Baryonic Tully-Fisher Relation', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    cb = plt.colorbar(ax1.collections[0], ax=ax1)
    cb.set_label('α')
    
    # ===================
    # Plot 2: Residuals vs Mass
    # ===================
    ax2 = axes[0, 1]
    
    ax2.scatter(df['log_M_bar'], df['residual_dex'], s=15, alpha=0.6)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    r, p = correlations['mass_corr']
    ax2.set_xlabel('log(M_bar / M☉)', fontsize=11)
    ax2.set_ylabel('Residual (dex)', fontsize=11)
    ax2.set_title(f'Residuals vs Mass (r = {r:.3f}, p = {p:.3f})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # ===================
    # Plot 3: Residuals vs α (RTM Prediction)
    # ===================
    ax3 = axes[0, 2]
    
    ax3.scatter(df['alpha'], df['residual_dex'], s=15, alpha=0.6, c='blue')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Fit line
    slope = model_comparison['rtm_slope']
    intercept = model_comparison['rtm_intercept']
    alpha_range = np.linspace(df['alpha'].min(), df['alpha'].max(), 50)
    ax3.plot(alpha_range, slope * alpha_range + intercept, 'r-', linewidth=2,
             label=f'Fit: slope = {slope:.3f}')
    
    r, p = correlations['alpha_corr']
    ax3.set_xlabel('α (coherence exponent)', fontsize=11)
    ax3.set_ylabel('Residual (dex)', fontsize=11)
    ax3.set_title(f'RTM Prediction: Residuals vs α\n(r = {r:.3f}, p = {p:.2e})', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ===================
    # Plot 4: Residuals vs Structure Index
    # ===================
    ax4 = axes[1, 0]
    
    ax4.scatter(df['structure_index'], df['residual_dex'], s=15, alpha=0.6, c='green')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    r, p = correlations['structure_corr']
    ax4.set_xlabel('Structure Index (proxy for α)', fontsize=11)
    ax4.set_ylabel('Residual (dex)', fontsize=11)
    ax4.set_title(f'Residuals vs Structure (r = {r:.3f}, p = {p:.2e})', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # ===================
    # Plot 5: Model comparison
    # ===================
    ax5 = axes[1, 1]
    
    models = ['Null (r²=0)', f'RTM (r²={model_comparison["rtm_r2"]:.3f})']
    r2_values = [0, model_comparison['rtm_r2']]
    colors = ['gray', 'blue']
    
    bars = ax5.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
    
    ax5.set_ylabel('R² (variance explained)', fontsize=11)
    ax5.set_title(f'Model Comparison\nF = {model_comparison["f_stat"]:.1f}, p = {model_comparison["p_f"]:.2e}', fontsize=12)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # ===================
    # Plot 6: Discriminant diagram
    # ===================
    ax6 = axes[1, 2]
    
    # Show what different models predict
    ax6.text(0.5, 0.9, 'MODEL PREDICTIONS FOR bTFR RESIDUALS', 
             transform=ax6.transAxes, fontsize=12, fontweight='bold',
             ha='center')
    
    ax6.text(0.1, 0.7, 'Dark Matter:', transform=ax6.transAxes, fontsize=11, fontweight='bold')
    ax6.text(0.15, 0.6, '• Correlate with halo concentration', transform=ax6.transAxes, fontsize=10)
    ax6.text(0.15, 0.52, '• Correlate with halo spin', transform=ax6.transAxes, fontsize=10)
    
    ax6.text(0.1, 0.4, 'MOND:', transform=ax6.transAxes, fontsize=11, fontweight='bold')
    ax6.text(0.15, 0.3, '• Correlate with acceleration scale', transform=ax6.transAxes, fontsize=10)
    
    ax6.text(0.1, 0.18, 'RTM:', transform=ax6.transAxes, fontsize=11, fontweight='bold', color='blue')
    ax6.text(0.15, 0.08, '• Correlate with structural coherence (α)', 
             transform=ax6.transAxes, fontsize=10, color='blue')
    
    ax6.text(0.5, -0.05, 'Testing requires real galaxy data!',
             transform=ax6.transAxes, fontsize=10, style='italic',
             ha='center', color='red')
    
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S3_btfr_residuals.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'S3_btfr_residuals.pdf'))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3: Baryonic Tully-Fisher Residuals and α-Correlation")
    print("From: Rhythmic Astronomy - Sections 4.2, 4.3")
    print("=" * 70)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("""
    This simulation shows what RTM PREDICTS for bTFR residuals.
    
    It does NOT prove:
    - That real galaxy residuals correlate with structure
    - That RTM is correct
    - That dark matter models are wrong
    
    This is a SYNTHETIC test showing the expected signal IF RTM is true.
    Real validation requires:
    - Actual galaxy survey data (SPARC, THINGS, etc.)
    - Independent α estimation from structural proxies
    - Comparison with DM and MOND predictions
    """)
    
    print("=" * 70)
    print("RTM PREDICTION")
    print("=" * 70)
    print("""
    Standard bTFR: M_bar ∝ v_flat^4
    
    RTM modification: v(r) ∝ r^(1 - α/2)
    
    Consequence:
    - If v is measured at radius where α ≠ 2, there's a systematic offset
    - Galaxies with higher inner α (more structure) show positive residuals
    - This correlation with structure distinguishes RTM from:
      * DM (residuals → halo parameters)
      * MOND (residuals → acceleration scale)
    """)
    
    print("=" * 70)
    print("GENERATING SYNTHETIC SAMPLE")
    print("=" * 70)
    
    # Generate sample
    df = generate_galaxy_sample(n_galaxies=200)
    
    print(f"\n    Generated {len(df)} synthetic galaxies")
    print(f"    Mass range: 10^{df['log_M_bar'].min():.1f} - 10^{df['log_M_bar'].max():.1f} M☉")
    print(f"    α range: {df['alpha'].min():.2f} - {df['alpha'].max():.2f}")
    print(f"    Residual range: {df['residual_dex'].min():.3f} to {df['residual_dex'].max():.3f} dex")
    
    # Analyze correlations
    correlations = analyze_residual_correlations(df)
    model_comparison = compare_model_predictions(df)
    
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    r_mass, p_mass = correlations['mass_corr']
    r_struct, p_struct = correlations['structure_corr']
    r_alpha, p_alpha = correlations['alpha_corr']
    
    print(f"""
    Residuals vs Mass:
      r = {r_mass:.4f}, p = {p_mass:.3f}
      Expected: weak (bTFR removes mass dependence)
    
    Residuals vs Structure Index:
      r = {r_struct:.4f}, p = {p_struct:.2e}
      RTM prediction: SIGNIFICANT correlation
    
    Residuals vs α:
      r = {r_alpha:.4f}, p = {p_alpha:.2e}
      RTM prediction: SIGNIFICANT correlation
    """)
    
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    print(f"""
    RTM model (residuals ~ α):
      Slope: {model_comparison['rtm_slope']:.4f} dex/unit-α
      R²: {model_comparison['rtm_r2']:.4f}
      
    F-test vs null model:
      F = {model_comparison['f_stat']:.1f}
      p = {model_comparison['p_f']:.2e}
    
    Interpretation:
    {'>>>> RTM model significantly better than null' if model_comparison['p_f'] < 0.05 else 'No significant improvement'}
    
    BUT REMEMBER: This is synthetic data designed to show the RTM signal!
    Real data may or may not show this correlation.
    """)
    
    # Save data
    df.to_csv(os.path.join(output_dir, 'S3_galaxy_sample.csv'), index=False)
    
    corr_df = pd.DataFrame({
        'parameter': ['mass', 'structure', 'alpha'],
        'correlation_r': [r_mass, r_struct, r_alpha],
        'p_value': [p_mass, p_struct, p_alpha]
    })
    corr_df.to_csv(os.path.join(output_dir, 'S3_correlations.csv'), index=False)
    
    # Create plots
    print("\n\nCreating plots...")
    create_plots(output_dir, df, correlations, model_comparison)
    
    # Summary
    summary = f"""S3: bTFR Residuals and α-Correlation
=====================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMPORTANT: This is a MODEL PREDICTION test, not empirical validation.

RTM PREDICTION
--------------
bTFR residuals should correlate with structural coherence (α)
because v(r) ∝ r^(1-α/2), so sampling at different α gives offsets.

SYNTHETIC RESULTS
-----------------
Sample size: {len(df)} galaxies

Correlations with bTFR residuals:
  Mass:      r = {r_mass:.4f} (weak, as expected)
  Structure: r = {r_struct:.4f} (RTM predicts: significant)
  α:         r = {r_alpha:.4f} (RTM predicts: significant)

Model comparison:
  RTM R² = {model_comparison['rtm_r2']:.4f}
  F-stat = {model_comparison['f_stat']:.1f}, p = {model_comparison['p_f']:.2e}

DISCRIMINANT
------------
DM prediction: residuals → halo parameters
MOND prediction: residuals → acceleration scale
RTM prediction: residuals → structural coherence

WHAT THIS SHOWS
---------------
✓ RTM makes a distinct, testable prediction
✓ The predicted correlation has a clear signature
✓ This could discriminate RTM from alternatives

WHAT THIS DOES NOT SHOW
-----------------------
✗ That real galaxy data shows this correlation
✗ That RTM is correct
✗ That DM/MOND predictions fail

REQUIRED FOR VALIDATION
-----------------------
1. Real bTFR data (SPARC, THINGS, LITTLE THINGS)
2. Independent α estimation from structure
3. Compare correlation strengths across models
"""
    
    with open(os.path.join(output_dir, 'S3_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
