#!/usr/bin/env python3
"""
RTM FLANK 5: THE SHAPE CONSPIRACY
===================================
A completely different approach: forget single-number correlations.
Ask whether the SHAPE of light predicts the SHAPE of the "missing" velocity.

FLANK A: THE MIRROR TEST
  Normalize V_bar(r) and V_obs(r) to unit amplitude.
  How similar are the SHAPES? Does SB structure predict this similarity?
  If baryonic geometry modulates dynamics, high-coherence galaxies
  should have V_obs shapes that "mirror" V_bar shapes more closely.

FLANK B: THE BARYON-HALO SHAPE CONSPIRACY  
  Compute V_DM(r) = sqrt(V_obs² - V_bar²).
  Correlate the SHAPE of V_DM(r) with the SHAPE of V_bar(r)
  within each galaxy. If DM is independent of baryons (ΛCDM),
  shapes should be uncorrelated. If coupled (RTM/MOND), they mirror.
  
FLANK C: THE ENTROPY ATTACK
  Compute the information entropy of the SB profile.
  "Boring" profiles (smooth exponentials) vs "complex" profiles
  (bumps, breaks, features). Does SB complexity predict DM behavior?
"""
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
from scipy.stats import f as f_dist
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('/home/claude/astro/table2.dat', sep=r'\s+', comment='#',
                 on_bad_lines='skip', skiprows=30,
                 names=['Name','Distance','Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB'])
df['Name'] = df['Name'].astype(str).str.replace(' ','').str.upper()
for c in ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

Upsilon_disk = 0.5; Upsilon_bul = 0.7

print("=" * 70)
print("RTM FLANK 5: THE SHAPE CONSPIRACY")
print("Does the SHAPE of light predict the SHAPE of dark matter?")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# FLANK A: THE MIRROR TEST
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLANK A: THE MIRROR TEST")
print("How similar are the SHAPES of V_bar(r) and V_obs(r)?")
print("Does structural coherence predict this shape similarity?")
print("=" * 70)

results = []
for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 10: continue
    
    r = sub['Rad'].values; vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values; vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values; sb = sub['SBdisk'].values
    I = 10**(-0.4 * sb)
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    vbar = np.sqrt(np.abs(vbar2))
    
    # V_DM
    vdm2 = vobs**2 - np.abs(vbar2)
    vdm = np.sqrt(np.maximum(vdm2, 0))
    
    # ─── MIRROR: Normalize shapes to [0,1] and correlate ───
    # Normalize by max to get pure shape
    vobs_norm = vobs / (np.max(vobs) + 1e-10)
    vbar_norm = vbar / (np.max(vbar) + 1e-10)
    
    # Shape similarity = Pearson correlation of normalized profiles
    mirror_r, mirror_p = stats.pearsonr(vobs_norm, vbar_norm)
    
    # Also: RMS shape difference
    shape_rms = np.sqrt(np.mean((vobs_norm - vbar_norm)**2))
    
    # ─── CONSPIRACY: V_DM shape vs V_bar shape ───
    # Normalize V_DM
    if np.max(vdm) > 5:
        vdm_norm = vdm / (np.max(vdm) + 1e-10)
        # Shape correlation between baryon and DM profiles
        conspiracy_r, conspiracy_p = stats.pearsonr(vbar_norm, vdm_norm)
        
        # Derivative conspiracy: do they curve the same way?
        log_r = np.log10(r)
        dvbar = np.gradient(vbar_norm, log_r)
        dvdm = np.gradient(vdm_norm, log_r)
        deriv_r, deriv_p = stats.pearsonr(dvbar, dvdm)
    else:
        conspiracy_r = conspiracy_p = deriv_r = deriv_p = np.nan
    
    # ─── ENTROPY: SB profile complexity ───
    # Residual from exponential fit = "bumpiness"
    log_r = np.log10(r); log_I = np.log10(np.maximum(I, 1e-30))
    ols_sb = stats.linregress(log_r, log_I)
    sb_residuals = log_I - (ols_sb.slope * log_r + ols_sb.intercept)
    sb_roughness = np.std(sb_residuals)  # how bumpy is the profile
    
    # Also compute actual entropy of SB gradient distribution
    dI = np.abs(np.gradient(log_I, log_r))
    dI_norm = dI / (np.sum(dI) + 1e-30)
    dI_norm = dI_norm[dI_norm > 0]
    sb_entropy = -np.sum(dI_norm * np.log(dI_norm + 1e-30))
    
    # ─── STRUCTURAL PARAMS ───
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = ols_sb.slope
    mu_0 = sb[0]
    
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    results.append({
        'galaxy': gal, 
        'mirror_r': mirror_r, 'shape_rms': shape_rms,
        'conspiracy_r': conspiracy_r, 'deriv_r': deriv_r,
        'sb_roughness': sb_roughness, 'sb_entropy': sb_entropy,
        'concentration': concentration, 'sb_slope': sb_slope, 'mu_0': mu_0,
        'v_flat': v_flat, 'log_mbar': log_mbar,
        'n_points': len(sub)
    })

rdf = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan).dropna(
    subset=['mirror_r','conspiracy_r','sb_roughness','log_mbar'])

print(f"\n  Galaxies analyzed: {len(rdf)}")

# ─── MIRROR RESULTS ───
print(f"\n  MIRROR (V_obs shape vs V_bar shape):")
print(f"    Mean shape correlation: {rdf['mirror_r'].mean():.3f} ± {rdf['mirror_r'].std():.3f}")
print(f"    Range: [{rdf['mirror_r'].min():.3f}, {rdf['mirror_r'].max():.3f}]")
print(f"    Galaxies with r > 0.9: {(rdf['mirror_r'] > 0.9).sum()} ({100*(rdf['mirror_r']>0.9).mean():.0f}%)")
print(f"    Galaxies with r < 0.5: {(rdf['mirror_r'] < 0.5).sum()} ({100*(rdf['mirror_r']<0.5).mean():.0f}%)")

# What predicts shape similarity?
print(f"\n  What predicts how well V_bar mirrors V_obs?")
print(f"  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_roughness', 'sb_entropy']:
    X_ctrl = np.column_stack([rdf['log_mbar'].values, rdf['v_flat'].values, np.ones(len(rdf))])
    c_y, _, _, _ = lstsq(X_ctrl, rdf['mirror_r'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, rdf[pred].values, rcond=None)
    res_y = rdf['mirror_r'].values - X_ctrl @ c_y
    res_x = rdf[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# ─── CONSPIRACY RESULTS ───
print(f"\n\n{'='*70}")
print(f"FLANK B: THE BARYON-HALO SHAPE CONSPIRACY")
print(f"{'='*70}")
rdf_c = rdf.dropna(subset=['conspiracy_r'])
print(f"\n  Galaxies with valid V_DM: {len(rdf_c)}")
print(f"\n  V_bar shape vs V_DM shape correlation:")
print(f"    Mean: {rdf_c['conspiracy_r'].mean():.3f} ± {rdf_c['conspiracy_r'].std():.3f}")
print(f"    % positive: {100*(rdf_c['conspiracy_r']>0).mean():.0f}%")

# Test: is mean conspiracy_r significantly different from 0?
t_consp, p_consp = stats.ttest_1samp(rdf_c['conspiracy_r'], 0)
print(f"    t-test vs 0: t = {t_consp:.2f}, p = {p_consp:.2e}")
print(f"    → {'V_bar and V_DM shapes ARE correlated (conspiracy!)' if p_consp < 0.05 else 'No systematic conspiracy'}")

if p_consp < 0.05:
    print(f"\n    THIS IS THE BARYON-HALO CONSPIRACY.")
    print(f"    In ΛCDM, V_DM(r) comes from an independent NFW halo.")
    print(f"    Its shape should NOT correlate with V_bar(r) shape.")
    print(f"    But it does: mean r = {rdf_c['conspiracy_r'].mean():.3f}")
    print(f"    This is one of the biggest puzzles in galaxy dynamics.")

# Does SB structure predict the STRENGTH of the conspiracy?
print(f"\n  What predicts conspiracy strength?")
print(f"  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_roughness', 'sb_entropy']:
    X_ctrl = np.column_stack([rdf_c['log_mbar'].values, rdf_c['v_flat'].values, np.ones(len(rdf_c))])
    c_y, _, _, _ = lstsq(X_ctrl, rdf_c['conspiracy_r'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, rdf_c[pred].values, rcond=None)
    res_y = rdf_c['conspiracy_r'].values - X_ctrl @ c_y
    res_x = rdf_c[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Derivative conspiracy
print(f"\n  DERIVATIVE CONSPIRACY (do V_bar and V_DM curve the same way?):")
rdf_d = rdf.dropna(subset=['deriv_r'])
print(f"    Mean deriv correlation: {rdf_d['deriv_r'].mean():.3f} ± {rdf_d['deriv_r'].std():.3f}")
t_d, p_d = stats.ttest_1samp(rdf_d['deriv_r'], 0)
print(f"    t-test vs 0: t = {t_d:.2f}, p = {p_d:.2e}")

# ─── ENTROPY RESULTS ───
print(f"\n\n{'='*70}")
print(f"FLANK C: THE ENTROPY ATTACK")
print(f"Does SB profile COMPLEXITY predict DM behavior?")
print(f"{'='*70}")

print(f"\n  SB roughness (deviation from exponential):")
print(f"    Mean: {rdf['sb_roughness'].mean():.3f} ± {rdf['sb_roughness'].std():.3f}")

# Does roughness predict mass discrepancy, baryon eff, conspiracy?
targets = {
    'mirror_r': 'Shape similarity',
    'conspiracy_r': 'Baryon-DM conspiracy', 
    'shape_rms': 'Shape RMS difference'
}

for target, label in targets.items():
    valid = rdf.dropna(subset=['sb_roughness', target, 'log_mbar', 'v_flat'])
    if len(valid) < 15: continue
    
    X_ctrl = np.column_stack([valid['log_mbar'].values, valid['v_flat'].values, np.ones(len(valid))])
    c_y, _, _, _ = lstsq(X_ctrl, valid[target].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, valid['sb_roughness'].values, rcond=None)
    res_y = valid[target].values - X_ctrl @ c_y
    res_x = valid['sb_roughness'].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {label:30s}: partial ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# Same for entropy
print(f"\n  SB entropy (gradient distribution complexity):")
for target, label in targets.items():
    valid = rdf.dropna(subset=['sb_entropy', target, 'log_mbar', 'v_flat'])
    if len(valid) < 15: continue
    
    X_ctrl = np.column_stack([valid['log_mbar'].values, valid['v_flat'].values, np.ones(len(valid))])
    c_y, _, _, _ = lstsq(X_ctrl, valid[target].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, valid['sb_entropy'].values, rcond=None)
    res_y = valid[target].values - X_ctrl @ c_y
    res_x = valid['sb_entropy'].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {label:30s}: partial ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# ═══════════════════════════════════════════════════════
# FINAL: BOOTSTRAP THE CONSPIRACY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"BOOTSTRAP: IS THE BARYON-HALO CONSPIRACY REAL?")
print(f"{'='*70}")

np.random.seed(42)
boot_means = []
for _ in range(5000):
    idx = np.random.choice(len(rdf_c), len(rdf_c), replace=True)
    boot_means.append(rdf_c['conspiracy_r'].values[idx].mean())
boot_means = np.array(boot_means)
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

print(f"  Mean conspiracy r = {rdf_c['conspiracy_r'].mean():.4f}")
print(f"  Bootstrap 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  CI excludes 0? {'YES ✓' if ci_lo > 0 or ci_hi < 0 else 'NO'}")
print(f"  % of boots > 0: {100*np.mean(np.array(boot_means) > 0):.1f}%")

# Also bootstrap the derivative conspiracy
boot_deriv = []
for _ in range(5000):
    idx = np.random.choice(len(rdf_d), len(rdf_d), replace=True)
    boot_deriv.append(rdf_d['deriv_r'].values[idx].mean())
boot_deriv = np.array(boot_deriv)
ci_d_lo, ci_d_hi = np.percentile(boot_deriv, [2.5, 97.5])

print(f"\n  Derivative conspiracy:")
print(f"  Mean deriv r = {rdf_d['deriv_r'].mean():.4f}")
print(f"  Bootstrap 95% CI = [{ci_d_lo:.4f}, {ci_d_hi:.4f}]")
print(f"  CI excludes 0? {'YES ✓' if ci_d_lo > 0 or ci_d_hi < 0 else 'NO'}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"FLANK 5 SUMMARY")
print(f"{'='*70}")
