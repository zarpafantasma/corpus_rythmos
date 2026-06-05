#!/usr/bin/env python3
"""
RTM FINAL ASSAULT: THE LOCAL COUPLING
=======================================
The unprecedented attack: POINT-BY-POINT, WITHIN-GALAXY.

At each radius in each galaxy:
  - Compute the LOCAL inferred DM density ρ_DM(r)
  - Compute the LOCAL baryonic geometry (SB gradient, gas fraction)
  - Ask: does LOCAL baryonic geometry predict LOCAL DM density,
    AFTER removing the global radial trend?

If this works: the local arrangement of visible matter predicts
the local density of "dark matter" — at the SAME radius, in the
SAME galaxy, controlling for the trivial "both vary with r" effect.

This has never been done point-by-point with structural controls.
"""
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('/home/claude/astro/table2.dat', sep=r'\s+', comment='#',
                 on_bad_lines='skip', skiprows=30,
                 names=['Name','Distance','Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB'])
df['Name'] = df['Name'].astype(str).str.replace(' ','').str.upper()
for c in ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

Upsilon_disk = 0.5; Upsilon_bul = 0.7

print("=" * 70)
print("THE FINAL ASSAULT: LOCAL BARYONIC GEOMETRY → LOCAL DM DENSITY")
print("Point-by-point. Within-galaxy. Controlling for radius.")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# PHASE 1: COMPUTE LOCAL DM DENSITY AT EACH RADIUS
# ═══════════════════════════════════════════════════════
# From V_obs(r): enclosed mass M(<r) = r * V²/G
# DM contribution: M_DM(<r) = r * (V_obs² - V_bar²) / G
# Local DM density: ρ_DM(r) ∝ d(M_DM)/d(volume) ∝ (1/r²) * d(r*V_DM²)/dr

all_points = []
galaxy_within_corrs = []

for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 12: continue
    
    r = sub['Rad'].values; vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values; vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values; sb = sub['SBdisk'].values
    I = 10**(-0.4 * sb)
    
    v2_gas = np.abs(vgas)*vgas
    v2_disk = Upsilon_disk * np.abs(vdisk)*vdisk
    v2_bul = Upsilon_bul * np.abs(vbul)*vbul
    vbar2 = v2_gas + v2_disk + v2_bul
    
    # DM enclosed mass proxy: M_DM(<r) ∝ r * V_DM²
    vdm2 = vobs**2 - np.abs(vbar2)
    m_dm_enc = r * np.maximum(vdm2, 0)  # proportional to enclosed DM mass
    
    # Local DM density: ρ_DM ∝ (1/4πr²) * dM_DM/dr
    # ≈ (1/r²) * Δ(r·V_DM²)/Δr
    dm_density = np.gradient(m_dm_enc, r) / (r**2 + 1e-10)
    dm_density = np.maximum(dm_density, 0)  # physical: density ≥ 0
    
    # Local baryonic geometry
    log_r = np.log10(r)
    log_I = np.log10(np.maximum(I, 1e-30))
    
    # G1: Local SB gradient
    dlogI = np.gradient(log_I, log_r)
    
    # G2: Local SB curvature (second derivative)
    d2logI = np.gradient(dlogI, log_r)
    
    # G3: Local gas fraction
    f_gas_local = np.abs(v2_gas) / (np.abs(vbar2) + 1e-10)
    
    # G4: Local baryon dominance = V_bar / V_obs
    bar_dom = np.sqrt(np.abs(vbar2)) / (vobs + 1e-10)
    
    # G5: Local SB "excess" — deviation from exponential at this radius
    ols_sb = stats.linregress(log_r, log_I)
    sb_excess = log_I - (ols_sb.slope * log_r + ols_sb.intercept)
    
    # Filter: need positive DM density and finite values
    mask = (dm_density > 0) & np.isfinite(dlogI) & np.isfinite(d2logI) & \
           np.isfinite(f_gas_local) & np.isfinite(sb_excess) & (r > r[1])  # skip first point
    
    if mask.sum() < 8: continue
    
    log_rho_dm = np.log10(dm_density[mask] + 1e-30)
    
    # ─── WITHIN-GALAXY: partial correlation controlling for log(r) ───
    # This is THE key test: at fixed radius, does SB geometry predict DM density?
    log_r_m = log_r[mask]
    
    # Regress DM density on radius (remove radial trend)
    ols_dm = stats.linregress(log_r_m, log_rho_dm)
    resid_dm = log_rho_dm - (ols_dm.slope * log_r_m + ols_dm.intercept)
    
    # For each geometric predictor, remove its radial trend too
    within_results = {}
    for gname, gvals in [('dlogI', dlogI[mask]), ('d2logI', d2logI[mask]),
                          ('f_gas', f_gas_local[mask]), ('sb_excess', sb_excess[mask])]:
        ols_g = stats.linregress(log_r_m, gvals)
        resid_g = gvals - (ols_g.slope * log_r_m + ols_g.intercept)
        
        rho_w, p_w = stats.spearmanr(resid_g, resid_dm)
        within_results[gname] = {'rho': rho_w, 'p': p_w}
    
    # Store per-galaxy results
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    f_gas_glob = np.sum(np.abs(v2_gas)) / (np.sum(np.abs(vbar2)) + 1e-10)
    
    galaxy_within_corrs.append({
        'galaxy': gal, 'n_points': mask.sum(),
        'rho_dlogI': within_results['dlogI']['rho'],
        'p_dlogI': within_results['dlogI']['p'],
        'rho_d2logI': within_results['d2logI']['rho'],
        'p_d2logI': within_results['d2logI']['p'],
        'rho_fgas': within_results['f_gas']['rho'],
        'p_fgas': within_results['f_gas']['p'],
        'rho_sbexcess': within_results['sb_excess']['rho'],
        'p_sbexcess': within_results['sb_excess']['p'],
        'v_flat': v_flat, 'concentration': concentration,
        'log_mbar': log_mbar, 'f_gas_global': f_gas_glob
    })
    
    # Store individual points for pooled analysis
    for i, idx in enumerate(np.where(mask)[0]):
        all_points.append({
            'galaxy': gal, 'r': r[idx], 'log_r': log_r[idx],
            'log_rho_dm': log_rho_dm[i],
            'dlogI': dlogI[idx], 'd2logI': d2logI[idx],
            'f_gas_local': f_gas_local[idx], 'sb_excess': sb_excess[idx],
            'bar_dom': bar_dom[idx]
        })

gwdf = pd.DataFrame(galaxy_within_corrs).replace([np.inf, -np.inf], np.nan).dropna(subset=['rho_dlogI'])
pdf = pd.DataFrame(all_points)

print(f"\n  Galaxies analyzed: {len(gwdf)}")
print(f"  Total radius points: {len(pdf)}")

# ═══════════════════════════════════════════════════════
# RESULTS: WITHIN-GALAXY PARTIAL CORRELATIONS
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("RESULT 1: WITHIN-GALAXY LOCAL COUPLING")
print("Does local SB geometry predict local DM density at fixed radius?")
print(f"{'='*70}")

for gname, label in [('dlogI', 'SB gradient (dlogI/dlogr)'),
                      ('d2logI', 'SB curvature (d²logI/dlogr²)'),
                      ('fgas', 'Local gas fraction'),
                      ('sbexcess', 'SB excess (deviation from exponential)')]:
    col = f'rho_{gname}'
    pcol = f'p_{gname}'
    vals = gwdf[col].dropna()
    
    t_val, p_val = stats.ttest_1samp(vals, 0)
    pct_pos = 100 * np.mean(vals > 0)
    pct_sig = 100 * np.mean(gwdf[pcol].dropna() < 0.05)
    
    print(f"\n  {label}:")
    print(f"    Mean within-galaxy partial ρ = {vals.mean():+.4f} ± {vals.std():.4f}")
    print(f"    Median = {vals.median():+.4f}")
    print(f"    t-test vs 0: t = {t_val:.2f}, p = {p_val:.2e}")
    print(f"    % positive: {pct_pos:.0f}%")
    print(f"    % individually significant: {pct_sig:.0f}%")

# ═══════════════════════════════════════════════════════
# SPLIT BY GAS FRACTION
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("RESULT 2: GAS-RICH vs GAS-POOR (within-galaxy coupling)")
print(f"{'='*70}")

median_fg = gwdf['f_gas_global'].median()
gas_rich = gwdf[gwdf['f_gas_global'] > median_fg]
gas_poor = gwdf[gwdf['f_gas_global'] <= median_fg]

for gname, label in [('dlogI', 'SB gradient'), ('fgas', 'Local gas frac'),
                      ('sbexcess', 'SB excess')]:
    col = f'rho_{gname}'
    gr_vals = gas_rich[col].dropna()
    gp_vals = gas_poor[col].dropna()
    
    t_gr, p_gr = stats.ttest_1samp(gr_vals, 0)
    t_gp, p_gp = stats.ttest_1samp(gp_vals, 0)
    u_gg, p_gg = stats.mannwhitneyu(gr_vals, gp_vals, alternative='two-sided')
    
    print(f"\n  {label}:")
    print(f"    Gas-rich  (n={len(gr_vals)}): mean ρ = {gr_vals.mean():+.4f}, "
          f"p vs 0 = {p_gr:.2e} {'✓' if p_gr < 0.05 else ''}")
    print(f"    Gas-poor  (n={len(gp_vals)}): mean ρ = {gp_vals.mean():+.4f}, "
          f"p vs 0 = {p_gp:.2e} {'✓' if p_gp < 0.05 else ''}")
    print(f"    Difference: p = {p_gg:.4f} {'★' if p_gg < 0.05 else ''}")

# ═══════════════════════════════════════════════════════
# POOLED ANALYSIS WITH GALAXY FIXED EFFECTS
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("RESULT 3: POOLED ANALYSIS — ALL POINTS, GALAXY FIXED EFFECTS")
print("Does local geometry predict local DM density across all radii?")
print(f"{'='*70}")

# Create galaxy dummies (fixed effects)
galaxies = pdf['galaxy'].unique()
gal_map = {g: i for i, g in enumerate(galaxies)}
pdf['gal_id'] = pdf['galaxy'].map(gal_map)

# Dependent variable
y = pdf['log_rho_dm'].values

# Model 1: Galaxy FE + log(r) only (baseline)
n_gal = len(galaxies)
X_fe = np.zeros((len(pdf), n_gal))
for i, g in enumerate(galaxies):
    X_fe[pdf['gal_id']==i, i] = 1.0

X1 = np.column_stack([X_fe, pdf['log_r'].values])
c1, _, _, _ = lstsq(X1, y, rcond=None)
r2_base = 1 - np.sum((y - X1@c1)**2) / np.sum((y - y.mean())**2)

# Model 2: Galaxy FE + log(r) + local SB geometry
X2 = np.column_stack([X_fe, pdf['log_r'].values,
                       pdf['dlogI'].values, pdf['sb_excess'].values,
                       pdf['f_gas_local'].values])
c2, _, _, _ = lstsq(X2, y, rcond=None)
r2_full = 1 - np.sum((y - X2@c2)**2) / np.sum((y - y.mean())**2)

n = len(pdf)
p1 = X1.shape[1]; p2 = X2.shape[1]
ss1 = np.sum((y - X1@c1)**2); ss2 = np.sum((y - X2@c2)**2)
f_inc = ((ss1-ss2)/(p2-p1)) / (ss2/(n-p2))
from scipy.stats import f as f_dist
p_f = 1 - f_dist.cdf(f_inc, p2-p1, n-p2)

print(f"\n  N points: {n}")
print(f"  N galaxies (fixed effects): {n_gal}")
print(f"\n  Galaxy FE + radius only:          R² = {r2_base:.4f}")
print(f"  Galaxy FE + radius + geometry:    R² = {r2_full:.4f}")
print(f"  ΔR²:                              {r2_full-r2_base:+.4f}")
print(f"  F-test (geometry adds value):     F = {f_inc:.1f}, p = {p_f:.2e}")

# Coefficients of the geometry terms
geo_coefs = c2[n_gal+1:]  # skip FE + log_r
geo_labels = ['dlogI', 'sb_excess', 'f_gas_local']
print(f"\n  Geometry coefficients:")
for l, c in zip(geo_labels, geo_coefs):
    print(f"    {l:18s}: {c:+.4f}")

# ═══════════════════════════════════════════════════════
# THE ULTIMATE TEST: LOCAL GAS FRACTION PREDICTS LOCAL DM
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("RESULT 4: THE ULTIMATE LOCAL COUPLING")
print("At the SAME radius in the SAME galaxy:")
print("Does local gas fraction predict local DM density?")
print(f"{'='*70}")

# Within each galaxy, at each radius, compute:
# - Residual DM density (after removing galaxy mean + radial trend)
# - Residual gas fraction (after removing galaxy mean + radial trend)
# Then correlate the residuals

# Remove galaxy means
pdf['log_rho_dm_demean'] = pdf.groupby('galaxy')['log_rho_dm'].transform(lambda x: x - x.mean())
pdf['fgas_demean'] = pdf.groupby('galaxy')['f_gas_local'].transform(lambda x: x - x.mean())
pdf['dlogI_demean'] = pdf.groupby('galaxy')['dlogI'].transform(lambda x: x - x.mean())
pdf['sbexc_demean'] = pdf.groupby('galaxy')['sb_excess'].transform(lambda x: x - x.mean())
pdf['logr_demean'] = pdf.groupby('galaxy')['log_r'].transform(lambda x: x - x.mean())

# Remove radial trend within each galaxy
pdf['dm_resid'] = pdf.groupby('galaxy').apply(
    lambda g: pd.Series(
        g['log_rho_dm_demean'].values - stats.linregress(g['logr_demean'].values, g['log_rho_dm_demean'].values).slope * g['logr_demean'].values,
        index=g.index
    )
).values

pdf['fgas_resid'] = pdf.groupby('galaxy').apply(
    lambda g: pd.Series(
        g['fgas_demean'].values - stats.linregress(g['logr_demean'].values, g['fgas_demean'].values).slope * g['logr_demean'].values,
        index=g.index
    )
).values

pdf['dlogI_resid'] = pdf.groupby('galaxy').apply(
    lambda g: pd.Series(
        g['dlogI_demean'].values - stats.linregress(g['logr_demean'].values, g['dlogI_demean'].values).slope * g['logr_demean'].values,
        index=g.index
    )
).values

# Clean
clean = pdf.dropna(subset=['dm_resid','fgas_resid','dlogI_resid'])
clean = clean[np.isfinite(clean['dm_resid']) & np.isfinite(clean['fgas_resid'])]

rho_fgas, p_fgas = stats.spearmanr(clean['fgas_resid'], clean['dm_resid'])
rho_dlogI, p_dlogI = stats.spearmanr(clean['dlogI_resid'], clean['dm_resid'])

print(f"\n  After removing galaxy means AND radial trends:")
print(f"  N points: {len(clean)}")
print(f"\n  LOCAL gas fraction → LOCAL DM density:")
print(f"    Spearman ρ = {rho_fgas:+.4f}, p = {p_fgas:.2e}")
print(f"\n  LOCAL SB gradient → LOCAL DM density:")
print(f"    Spearman ρ = {rho_dlogI:+.4f}, p = {p_dlogI:.2e}")

# Bootstrap
np.random.seed(42)
boot_fgas = []
boot_dlogI = []
# Cluster bootstrap by galaxy to preserve within-galaxy structure
gal_list = clean['galaxy'].unique()
for _ in range(3000):
    boot_gals = np.random.choice(gal_list, len(gal_list), replace=True)
    boot_data = pd.concat([clean[clean['galaxy']==g] for g in boot_gals])
    r1, _ = stats.spearmanr(boot_data['fgas_resid'], boot_data['dm_resid'])
    r2, _ = stats.spearmanr(boot_data['dlogI_resid'], boot_data['dm_resid'])
    boot_fgas.append(r1)
    boot_dlogI.append(r2)

boot_fgas = np.array(boot_fgas)
boot_dlogI = np.array(boot_dlogI)

ci_fg = np.percentile(boot_fgas, [2.5, 97.5])
ci_dl = np.percentile(boot_dlogI, [2.5, 97.5])

print(f"\n  CLUSTER BOOTSTRAP (3000 iterations, by galaxy):")
print(f"    f_gas → DM: mean = {np.mean(boot_fgas):+.4f}, 95% CI = [{ci_fg[0]:+.4f}, {ci_fg[1]:+.4f}]")
print(f"    CI excludes 0? {'YES ✓' if ci_fg[0] > 0 or ci_fg[1] < 0 else 'NO'}")
print(f"\n    dlogI → DM: mean = {np.mean(boot_dlogI):+.4f}, 95% CI = [{ci_dl[0]:+.4f}, {ci_dl[1]:+.4f}]")
print(f"    CI excludes 0? {'YES ✓' if ci_dl[0] > 0 or ci_dl[1] < 0 else 'NO'}")

# ═══════════════════════════════════════════════════════
# RESULT 5: THE SMOKING GUN — WITHIN-GALAXY, POINT-BY-POINT
# What fraction of individual galaxies show this local coupling?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("RESULT 5: PER-GALAXY LOCAL f_gas → LOCAL ρ_DM COUPLING")
print(f"{'='*70}")

per_gal_local = []
for gal in gal_list:
    gsub = clean[clean['galaxy']==gal]
    if len(gsub) < 8: continue
    rho_l, p_l = stats.spearmanr(gsub['fgas_resid'], gsub['dm_resid'])
    if np.isfinite(rho_l):
        per_gal_local.append({'galaxy': gal, 'rho': rho_l, 'p': p_l, 'n': len(gsub)})

pgl = pd.DataFrame(per_gal_local)
print(f"\n  Galaxies with n≥8: {len(pgl)}")
print(f"  Mean local ρ(f_gas, ρ_DM | radius): {pgl['rho'].mean():+.4f} ± {pgl['rho'].std():.4f}")
print(f"  Median: {pgl['rho'].median():+.4f}")
print(f"  % positive: {100*np.mean(pgl['rho']>0):.0f}%")
print(f"  % individually significant: {100*np.mean(pgl['p']<0.05):.0f}%")
t_pgl, p_pgl = stats.ttest_1samp(pgl['rho'], 0)
print(f"  t-test vs 0: t = {t_pgl:.2f}, p = {p_pgl:.2e}")

# Does this local coupling depend on global gas fraction?
median_fg = gwdf['f_gas_global'].median()
pgl_merged = pgl.merge(gwdf[['galaxy','f_gas_global']], on='galaxy')
gr = pgl_merged[pgl_merged['f_gas_global'] > median_fg]
gp = pgl_merged[pgl_merged['f_gas_global'] <= median_fg]

if len(gr) >= 5 and len(gp) >= 5:
    print(f"\n  LOCAL coupling split by GLOBAL gas fraction:")
    print(f"    Gas-rich  (n={len(gr)}): mean local ρ = {gr['rho'].mean():+.4f}, "
          f"p vs 0 = {stats.ttest_1samp(gr['rho'], 0)[1]:.2e}")
    print(f"    Gas-poor  (n={len(gp)}): mean local ρ = {gp['rho'].mean():+.4f}, "
          f"p vs 0 = {stats.ttest_1samp(gp['rho'], 0)[1]:.2e}")

# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
