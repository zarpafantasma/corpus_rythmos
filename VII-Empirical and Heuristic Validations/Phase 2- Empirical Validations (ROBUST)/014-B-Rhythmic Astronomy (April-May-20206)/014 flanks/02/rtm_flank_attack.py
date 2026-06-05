#!/usr/bin/env python3
"""
RTM FLANK ATTACK: RAR Scatter + Diversity Problem
===================================================
Instead of replacing DM, test if baryonic STRUCTURE predicts
the PATTERN of the mass discrepancy.

Flank A: Does local SB gradient reduce RAR scatter?
Flank B: At fixed V_flat, does SB shape predict rotation curve shape?
"""
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import json, warnings
warnings.filterwarnings('ignore')

# Load
df = pd.read_csv('/home/claude/astro/table2.dat', sep=r'\s+', comment='#',
                 on_bad_lines='skip', skiprows=30,
                 names=['Name','Distance','Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB'])
df['Name'] = df['Name'].astype(str).str.replace(' ','').str.upper()
for c in ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB','Distance']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

Upsilon_disk = 0.5
Upsilon_bul = 0.7

print("=" * 70)
print("RTM FLANK ATTACK: RAR SCATTER + DIVERSITY PROBLEM")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# FLANK A: POINT-BY-POINT RAR WITH LOCAL STRUCTURE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLANK A: DOES LOCAL SB STRUCTURE REDUCE RAR SCATTER?")
print("=" * 70)

# Compute g_obs and g_bar at each radius point
all_points = []
galaxies_used = []

for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 8: continue
    
    r = sub['Rad'].values  # kpc
    vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values
    vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values
    sb = sub['SBdisk'].values
    
    # Accelerations: g = v²/r (in consistent units, we use km²/s²/kpc)
    g_obs = vobs**2 / r
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    g_bar = np.abs(vbar2) / r
    
    # Local SB structure
    I = 10**(-0.4 * sb)
    log_r = np.log10(r)
    log_I = np.log10(I + 1e-30)
    
    # Local gradient
    if len(log_r) >= 3:
        dlogI = np.gradient(log_I, log_r)
    else:
        continue
    
    # SB concentration at this radius (ratio to mean)
    sb_local_ratio = I / np.mean(I)
    
    # Fractional radius (0=center, 1=edge)
    r_frac = (r - r.min()) / (r.max() - r.min() + 1e-10)
    
    mask = (g_obs > 0) & (g_bar > 0) & np.isfinite(dlogI)
    
    for i in range(len(r)):
        if not mask[i]: continue
        all_points.append({
            'galaxy': gal,
            'r_kpc': r[i],
            'r_frac': r_frac[i],
            'log_gobs': np.log10(g_obs[i]),
            'log_gbar': np.log10(g_bar[i]),
            'dlogI_dlogr': dlogI[i],
            'abs_gradient': abs(dlogI[i]),
            'log_sb_ratio': np.log10(sb_local_ratio[i] + 1e-10),
            'sb_mag': sb[i],
            'vobs': vobs[i]
        })
    galaxies_used.append(gal)

pdf = pd.DataFrame(all_points)
print(f"  Total radius points: {len(pdf)}")
print(f"  Galaxies: {len(galaxies_used)}")

# Standard RAR: log(g_obs) = f(log(g_bar))
# McGaugh interpolating function: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))
# with g† ≈ 1.2 × 10^-10 m/s² ≈ 3690 km²/s²/kpc
g_dagger = 3690  # km²/s²/kpc

# Compute RAR residuals
pdf['g_bar_lin'] = 10**pdf['log_gbar']
pdf['g_obs_pred_rar'] = pdf['g_bar_lin'] / (1 - np.exp(-np.sqrt(pdf['g_bar_lin'] / g_dagger)))
pdf['log_gobs_pred'] = np.log10(pdf['g_obs_pred_rar'])
pdf['rar_residual'] = pdf['log_gobs'] - pdf['log_gobs_pred']

# Clean
pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna(subset=['rar_residual', 'abs_gradient'])
pdf = pdf[np.abs(pdf['rar_residual']) < 2]  # remove extreme outliers

rar_scatter_baseline = pdf['rar_residual'].std()
print(f"\n  RAR baseline scatter (σ): {rar_scatter_baseline:.4f} dex")
print(f"  (McGaugh 2016 reports ~0.13 dex)")

# Does the local SB gradient correlate with RAR residuals?
rho_grad, p_grad = stats.spearmanr(pdf['abs_gradient'], pdf['rar_residual'])
rho_sb, p_sb = stats.spearmanr(pdf['sb_mag'], pdf['rar_residual'])
rho_rfrac, p_rfrac = stats.spearmanr(pdf['r_frac'], pdf['rar_residual'])

print(f"\n  CORRELATIONS WITH RAR RESIDUALS:")
print(f"  {'Predictor':20s} {'Spearman ρ':>12s} {'p':>12s}")
print(f"  {'-'*48}")
print(f"  {'|d(logI)/d(logr)|':20s} {rho_grad:+12.4f} {p_grad:12.2e}")
print(f"  {'SB magnitude':20s} {rho_sb:+12.4f} {p_sb:12.2e}")
print(f"  {'Fractional radius':20s} {rho_rfrac:+12.4f} {p_rfrac:12.2e}")

# Multivariate: can SB gradient REDUCE RAR scatter?
# Model 1: RAR alone → residual σ = baseline
# Model 2: RAR + gradient → does σ shrink?

y = pdf['rar_residual'].values
X = np.column_stack([pdf['abs_gradient'].values, pdf['sb_mag'].values, np.ones(len(pdf))])
c, _, _, _ = lstsq(X, y, rcond=None)
y_corr = y - X @ c
scatter_corrected = np.std(y_corr)

reduction_pct = 100 * (1 - scatter_corrected / rar_scatter_baseline)
print(f"\n  RAR scatter after structure correction:")
print(f"    Baseline σ:   {rar_scatter_baseline:.4f} dex")
print(f"    Corrected σ:  {scatter_corrected:.4f} dex")
print(f"    Reduction:    {reduction_pct:.1f}%")

# F-test for the improvement
n = len(pdf)
ss_base = np.sum(y**2)
ss_corr = np.sum(y_corr**2)
f_rar = ((ss_base - ss_corr) / 2) / (ss_corr / (n - 3))
from scipy.stats import f as f_dist
p_f_rar = 1 - f_dist.cdf(f_rar, 2, n - 3)
print(f"    F-test: F = {f_rar:.1f}, p = {p_f_rar:.2e}")

# ═══════════════════════════════════════════════════════
# FLANK A.2: WITHIN-GALAXY RAR RESIDUAL PREDICTION
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLANK A.2: WITHIN-GALAXY — Does SB gradient predict RAR")
print("           residual profile WITHIN individual galaxies?")
print("=" * 70)

within_corrs = []
for gal in galaxies_used:
    gsub = pdf[pdf['galaxy']==gal]
    if len(gsub) < 8: continue
    rho_w, p_w = stats.spearmanr(gsub['abs_gradient'], gsub['rar_residual'])
    if np.isfinite(rho_w):
        within_corrs.append({'galaxy': gal, 'rho': rho_w, 'p': p_w, 'n': len(gsub)})

wdf = pd.DataFrame(within_corrs)
print(f"  Galaxies with n≥8: {len(wdf)}")
print(f"  Mean within-galaxy ρ: {wdf['rho'].mean():.4f}")
print(f"  Median within-galaxy ρ: {wdf['rho'].median():.4f}")
print(f"  % with ρ > 0: {100*np.mean(wdf['rho'] > 0):.0f}%")
print(f"  % significant (p<0.05): {100*np.mean(wdf['p'] < 0.05):.0f}%")

t_within, p_within = stats.ttest_1samp(wdf['rho'], 0)
print(f"  t-test ρ ≠ 0: t={t_within:.2f}, p={p_within:.2e}")

# ═══════════════════════════════════════════════════════
# FLANK B: DIVERSITY PROBLEM — SHAPE PREDICTION
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK B: THE DIVERSITY PROBLEM")
print("At fixed V_flat, does SB profile SHAPE predict rotation curve SHAPE?")
print("=" * 70)

# Compute per-galaxy summary stats
gal_stats = []
for gal in galaxies_used:
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 8: continue
    
    r = sub['Rad'].values
    vobs = sub['Vobs'].values
    sb = sub['SBdisk'].values
    vgas = sub['Vgas'].values
    vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values
    
    # V_flat (median of outer 40%)
    n_outer = max(3, int(0.4 * len(r)))
    v_flat = np.median(vobs[-n_outer:])
    v_max = np.max(vobs)
    
    # RC SHAPE: inner slope (first 40%)
    n_inner = max(3, int(0.4 * len(r)))
    log_r = np.log10(r)
    log_v = np.log10(vobs)
    inner_slope = stats.linregress(log_r[:n_inner], log_v[:n_inner]).slope
    
    # RC SHAPE: rise-to-flat ratio
    v_inner = np.median(vobs[:n_inner])
    rise_ratio = v_inner / v_flat if v_flat > 0 else np.nan
    
    # SB SHAPE
    I = 10**(-0.4 * sb)
    log_I = np.log10(I + 1e-30)
    
    # SB concentration (inner 30% / outer 30%)
    n30 = max(2, int(0.3 * len(sb)))
    sb_concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    
    # SB slope (log-log)
    sb_slope = stats.linregress(log_r, log_I).slope
    
    # SB curvature
    if len(log_r) >= 5:
        sb_curv = np.polyfit(log_r, log_I, 2)[0]
    else:
        sb_curv = np.nan
    
    # Baryonic dominance in inner region
    vbar2_inner = np.abs(vgas[:n_inner])*vgas[:n_inner] + Upsilon_disk*np.abs(vdisk[:n_inner])*vdisk[:n_inner]
    vbar_inner = np.sqrt(np.abs(np.mean(vbar2_inner)))
    bar_dominance = vbar_inner / (v_inner + 1e-10)
    
    gal_stats.append({
        'galaxy': gal, 'v_flat': v_flat, 'v_max': v_max,
        'inner_slope': inner_slope, 'rise_ratio': rise_ratio,
        'sb_concentration': sb_concentration, 'sb_slope': sb_slope,
        'sb_curvature': sb_curv, 'bar_dominance': bar_dominance,
        'n_points': len(sub)
    })

gdf = pd.DataFrame(gal_stats).dropna()
print(f"  Galaxies: {len(gdf)}")

# Split into V_flat bins
gdf['vflat_bin'] = pd.qcut(gdf['v_flat'], 3, labels=['Slow','Medium','Fast'])

print(f"\n  V_flat bins:")
for b in ['Slow','Medium','Fast']:
    bsub = gdf[gdf['vflat_bin']==b]
    print(f"    {b:8s}: n={len(bsub)}, V_flat=[{bsub['v_flat'].min():.0f}, {bsub['v_flat'].max():.0f}] km/s")

# KEY TEST: Within each V_flat bin, does SB shape predict RC shape?
print(f"\n  WITHIN V_FLAT BINS: Does SB predict RC shape?")
print(f"  {'Bin':8s} {'n':>4s} {'ρ(SBconc,inner_slope)':>24s} {'p':>10s} {'ρ(SBslope,rise_ratio)':>24s} {'p':>10s}")
print("  " + "-" * 80)

diversity_results = []
for b in ['Slow','Medium','Fast']:
    bsub = gdf[gdf['vflat_bin']==b]
    if len(bsub) < 8: continue
    
    rho1, p1 = stats.spearmanr(bsub['sb_concentration'], bsub['inner_slope'])
    rho2, p2 = stats.spearmanr(bsub['sb_slope'], bsub['rise_ratio'])
    
    sig1 = '*' if p1 < 0.05 else ''
    sig2 = '*' if p2 < 0.05 else ''
    print(f"  {b:8s} {len(bsub):4d} {rho1:+20.3f}{sig1:>4s} {p1:10.4f} {rho2:+20.3f}{sig2:>4s} {p2:10.4f}")
    
    diversity_results.append({
        'bin': b, 'n': len(bsub),
        'rho_conc_slope': rho1, 'p_conc_slope': p1,
        'rho_sbslope_rise': rho2, 'p_sbslope_rise': p2
    })

# Overall (not binned)
rho_all1, p_all1 = stats.spearmanr(gdf['sb_concentration'], gdf['inner_slope'])
rho_all2, p_all2 = stats.spearmanr(gdf['sb_slope'], gdf['rise_ratio'])
print(f"  {'ALL':8s} {len(gdf):4d} {rho_all1:+20.3f}{'*' if p_all1<0.05 else '':>4s} {p_all1:10.4f} {rho_all2:+20.3f}{'*' if p_all2<0.05 else '':>4s} {p_all2:10.4f}")

# PARTIAL: control for V_flat
ols_v1 = stats.linregress(gdf['v_flat'], gdf['inner_slope'])
ols_v2 = stats.linregress(gdf['v_flat'], gdf['sb_concentration'])
res_is = gdf['inner_slope'] - (ols_v1.slope * gdf['v_flat'] + ols_v1.intercept)
res_sc = gdf['sb_concentration'] - (ols_v2.slope * gdf['v_flat'] + ols_v2.intercept)
rho_partial, p_partial = stats.spearmanr(res_sc, res_is)
print(f"\n  PARTIAL (controlling V_flat):")
print(f"    ρ(SB_conc, inner_slope | V_flat) = {rho_partial:+.4f}, p = {p_partial:.2e}")

# Multivariate: inner_slope ~ v_flat + sb_concentration + sb_slope
y = gdf['inner_slope'].values
X1 = np.column_stack([gdf['v_flat'].values, np.ones(len(gdf))])
c1, _, _, _ = lstsq(X1, y, rcond=None)
r2_vflat = 1 - np.sum((y - X1@c1)**2) / np.sum((y - y.mean())**2)

X2 = np.column_stack([gdf['v_flat'].values, gdf['sb_concentration'].values, 
                       gdf['sb_slope'].values, np.ones(len(gdf))])
c2, _, _, _ = lstsq(X2, y, rcond=None)
r2_full = 1 - np.sum((y - X2@c2)**2) / np.sum((y - y.mean())**2)

# F-test
n = len(gdf)
ss1 = np.sum((y - X1@c1)**2)
ss2 = np.sum((y - X2@c2)**2)
f_div = ((ss1 - ss2) / 2) / (ss2 / (n - 4))
p_f_div = 1 - f_dist.cdf(f_div, 2, n - 4)

print(f"\n  MULTIVARIATE (inner_slope prediction):")
print(f"    V_flat only:          R² = {r2_vflat:.4f}")
print(f"    V_flat + SB shape:    R² = {r2_full:.4f}")
print(f"    ΔR²:                  {r2_full - r2_vflat:+.4f}")
print(f"    F-test:               F = {f_div:.2f}, p = {p_f_div:.4e}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("SUMMARY OF FLANK ATTACK")
print("=" * 70)

summary = {
    "flank_A_rar_scatter": {
        "baseline_scatter_dex": round(rar_scatter_baseline, 4),
        "corrected_scatter_dex": round(scatter_corrected, 4),
        "reduction_pct": round(reduction_pct, 1),
        "f_test_p": float(f"{p_f_rar:.2e}"),
        "gradient_rho": round(rho_grad, 4),
        "gradient_p": float(f"{p_grad:.2e}")
    },
    "flank_A2_within_galaxy": {
        "n_galaxies": len(wdf),
        "mean_rho": round(wdf['rho'].mean(), 4),
        "median_rho": round(wdf['rho'].median(), 4),
        "pct_positive": round(100*np.mean(wdf['rho'] > 0), 1),
        "pct_significant": round(100*np.mean(wdf['p'] < 0.05), 1),
        "ttest_p": float(f"{p_within:.2e}")
    },
    "flank_B_diversity": {
        "partial_rho_conc_innerslope": round(rho_partial, 4),
        "partial_p": float(f"{p_partial:.2e}"),
        "r2_vflat_only": round(r2_vflat, 4),
        "r2_vflat_plus_sb": round(r2_full, 4),
        "delta_r2": round(r2_full - r2_vflat, 4),
        "f_test_p": float(f"{p_f_div:.2e}")
    }
}

with open('/home/claude/rtm_flank_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nResults saved.")
