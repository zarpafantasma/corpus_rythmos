#!/usr/bin/env python3
"""
RTM FLANK 4: THREE MORE ANGLES
================================
A: Per-galaxy RAR offset — do galaxies sit systematically above/below RAR?
B: DM fraction profile SHAPE — does SB predict HOW DM fraction grows?
C: The baryon-DM conspiracy radius — where baryon contribution peaks
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
g_dagger = 3690  # km²/s²/kpc (≈ 1.2e-10 m/s²)

print("=" * 70)
print("RTM FLANK 4: THREE MORE ANGLES OF ATTACK")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# FLANK A: PER-GALAXY RAR OFFSET
# The RAR scatter is ~0.13 dex per POINT.
# But each GALAXY has a mean offset from RAR.
# Does this systematic galaxy-level offset correlate with SB?
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLANK A: PER-GALAXY SYSTEMATIC RAR OFFSET")
print("Each galaxy sits slightly above or below the universal RAR.")
print("Is this offset predicted by baryonic structure?")
print("=" * 70)

gal_rar = []
for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 8: continue
    
    r = sub['Rad'].values; vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values; vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values; sb = sub['SBdisk'].values; I = 10**(-0.4*sb)
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    g_obs = vobs**2 / r
    g_bar = np.abs(vbar2) / r
    
    mask = (g_bar > 10) & (g_obs > 10)
    if mask.sum() < 5: continue
    
    # McGaugh RAR prediction
    g_rar_pred = g_bar[mask] / (1 - np.exp(-np.sqrt(g_bar[mask] / g_dagger)))
    
    # Per-point residual
    log_resid = np.log10(g_obs[mask]) - np.log10(g_rar_pred)
    log_resid = log_resid[np.isfinite(log_resid)]
    if len(log_resid) < 3: continue
    
    # Galaxy-level RAR offset (mean residual)
    rar_offset = np.median(log_resid)
    rar_scatter = np.std(log_resid)  # internal scatter
    
    # SB structure
    log_r = np.log10(r); log_I = np.log10(np.maximum(I, 1e-30))
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    mu_0 = sb[0]  # central SB
    
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    # SB Sérsic-like curvature
    if len(log_r) >= 5:
        sb_curv = np.polyfit(log_r, log_I, 2)[0]
    else:
        sb_curv = np.nan
    
    # Effective radius (half-light)
    cumI = np.cumsum(I * r)
    total_light = cumI[-1]
    r_eff_idx = np.searchsorted(cumI, total_light/2)
    r_eff = r[min(r_eff_idx, len(r)-1)]
    
    gal_rar.append({
        'galaxy': gal, 'rar_offset': rar_offset, 'rar_scatter': rar_scatter,
        'concentration': concentration, 'sb_slope': sb_slope, 'mu_0': mu_0,
        'sb_curv': sb_curv, 'r_eff': r_eff, 'log_r_eff': np.log10(r_eff),
        'v_flat': v_flat, 'log_mbar': log_mbar, 'n_points': mask.sum()
    })

ga = pd.DataFrame(gal_rar).replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Galaxies with RAR offset: {len(ga)}")
print(f"  RAR offset range: {ga['rar_offset'].min():.3f} to {ga['rar_offset'].max():.3f} dex")
print(f"  RAR offset std: {ga['rar_offset'].std():.3f} dex")

# Zero-order
print(f"\n  ZERO-ORDER correlations with RAR offset:")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_curv', 'log_r_eff', 'log_mbar', 'v_flat']:
    valid = ga.dropna(subset=[pred])
    rho, p = stats.spearmanr(valid[pred], valid['rar_offset'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL controlling for mass
print(f"\n  PARTIAL (controlling log_mbar):")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_curv', 'log_r_eff']:
    valid = ga.dropna(subset=[pred, 'rar_offset', 'log_mbar'])
    o1 = stats.linregress(valid['log_mbar'], valid['rar_offset'])
    o2 = stats.linregress(valid['log_mbar'], valid[pred])
    res_y = valid['rar_offset'].values - (o1.slope * valid['log_mbar'].values + o1.intercept)
    res_x = valid[pred].values - (o2.slope * valid['log_mbar'].values + o2.intercept)
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL controlling for mass AND v_flat
print(f"\n  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_curv', 'log_r_eff']:
    valid = ga.dropna(subset=[pred, 'rar_offset', 'log_mbar', 'v_flat'])
    X_ctrl = np.column_stack([valid['log_mbar'].values, valid['v_flat'].values, np.ones(len(valid))])
    c_y, _, _, _ = lstsq(X_ctrl, valid['rar_offset'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, valid[pred].values, rcond=None)
    res_y = valid['rar_offset'].values - X_ctrl @ c_y
    res_x = valid[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Multivariate: how much scatter can we explain?
y = ga['rar_offset'].values
X1 = np.column_stack([ga['log_mbar'].values, ga['v_flat'].values, np.ones(len(ga))])
c1, _, _, _ = lstsq(X1, y, rcond=None)
r2_mass = 1 - np.sum((y-X1@c1)**2)/np.sum((y-y.mean())**2)

X2 = np.column_stack([ga['log_mbar'].values, ga['v_flat'].values,
                       ga['concentration'].values, ga['mu_0'].values, np.ones(len(ga))])
c2, _, _, _ = lstsq(X2, y, rcond=None)
r2_full = 1 - np.sum((y-X2@c2)**2)/np.sum((y-y.mean())**2)

n = len(ga)
ss1 = np.sum((y-X1@c1)**2); ss2 = np.sum((y-X2@c2)**2)
f_inc = ((ss1-ss2)/2)/(ss2/(n-5))
p_f = 1 - f_dist.cdf(f_inc, 2, n-5)

print(f"\n  MULTIVARIATE:")
print(f"    Mass + V_flat only:        R² = {r2_mass:.4f}")
print(f"    Mass + V_flat + SB struct: R² = {r2_full:.4f}")
print(f"    ΔR²:                       {r2_full-r2_mass:+.4f}")
print(f"    F-test:                    p = {p_f:.2e}")

# ═══════════════════════════════════════════════════════
# FLANK B: DM FRACTION GROWTH RATE
# How quickly does f_DM = V_DM²/V_obs² grow with radius?
# This "steepness of DM takeover" should relate to SB structure
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK B: DM FRACTION GROWTH RATE")
print("How fast does DM take over? Does SB predict the steepness?")
print("=" * 70)

gal_fdm = []
for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 10: continue
    
    r = sub['Rad'].values; vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values; vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values; sb = sub['SBdisk'].values; I = 10**(-0.4*sb)
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    
    # DM fraction: f_DM(r) = 1 - V_bar²/V_obs²
    f_dm = 1 - np.abs(vbar2) / (vobs**2 + 1e-10)
    f_dm = np.clip(f_dm, 0, 1)
    
    # Growth rate: slope of f_DM vs log(r)
    log_r = np.log10(r)
    mask = np.isfinite(f_dm) & np.isfinite(log_r) & (f_dm > 0.01) & (f_dm < 0.99)
    if mask.sum() < 5: continue
    
    growth = stats.linregress(log_r[mask], f_dm[mask])
    dm_growth_rate = growth.slope  # higher = faster DM takeover
    dm_growth_r2 = growth.rvalue**2
    
    # Where does f_DM cross 0.5?
    f_dm_cross = np.nan
    for i in range(len(f_dm)-1):
        if f_dm[i] < 0.5 and f_dm[i+1] >= 0.5:
            frac = (0.5 - f_dm[i]) / (f_dm[i+1] - f_dm[i] + 1e-10)
            f_dm_cross = r[i] + frac * (r[i+1] - r[i])
            break
    
    # SB structure
    log_I = np.log10(np.maximum(I, 1e-30))
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    mu_0 = sb[0]
    
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    gal_fdm.append({
        'galaxy': gal, 'dm_growth_rate': dm_growth_rate,
        'dm_growth_r2': dm_growth_r2,
        'f_dm_cross_r': f_dm_cross,
        'log_f_dm_cross': np.log10(f_dm_cross) if f_dm_cross and f_dm_cross > 0 else np.nan,
        'concentration': concentration, 'sb_slope': sb_slope, 'mu_0': mu_0,
        'v_flat': v_flat, 'log_mbar': log_mbar
    })

gb = pd.DataFrame(gal_fdm).replace([np.inf, -np.inf], np.nan).dropna(subset=['dm_growth_rate','concentration','log_mbar'])
print(f"\n  Galaxies: {len(gb)}")

# Zero-order
print(f"\n  ZERO-ORDER with DM growth rate:")
for pred in ['concentration', 'sb_slope', 'mu_0', 'log_mbar', 'v_flat']:
    rho, p = stats.spearmanr(gb[pred], gb['dm_growth_rate'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL
print(f"\n  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0']:
    X_ctrl = np.column_stack([gb['log_mbar'].values, gb['v_flat'].values, np.ones(len(gb))])
    c_y, _, _, _ = lstsq(X_ctrl, gb['dm_growth_rate'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, gb[pred].values, rcond=None)
    res_y = gb['dm_growth_rate'].values - X_ctrl @ c_y
    res_x = gb[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Also check f_DM crossing radius
gb_cross = gb.dropna(subset=['log_f_dm_cross'])
if len(gb_cross) >= 15:
    print(f"\n  DM 50% CROSSING RADIUS (n={len(gb_cross)}):")
    for pred in ['concentration', 'sb_slope', 'mu_0']:
        o1 = stats.linregress(gb_cross['log_mbar'], gb_cross['log_f_dm_cross'])
        o2 = stats.linregress(gb_cross['log_mbar'], gb_cross[pred])
        res_y = gb_cross['log_f_dm_cross'].values - (o1.slope * gb_cross['log_mbar'].values + o1.intercept)
        res_x = gb_cross[pred].values - (o2.slope * gb_cross['log_mbar'].values + o2.intercept)
        rho, p = stats.spearmanr(res_x, res_y)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# ═══════════════════════════════════════════════════════
# FLANK C: THE BARYON-DM CONSPIRACY
# At what radius does V_bar peak? Does SB predict the 
# ratio V_bar_peak / V_flat (how much baryons contribute at max)?
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK C: THE BARYON-DM CONSPIRACY")
print("How much do baryons contribute at their peak? (V_bar_max / V_flat)")
print("Does SB structure predict this 'baryon effectiveness'?")
print("=" * 70)

gal_consp = []
for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 8: continue
    
    r = sub['Rad'].values; vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values; vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values; sb = sub['SBdisk'].values; I = 10**(-0.4*sb)
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    vbar = np.sqrt(np.abs(vbar2))
    
    # Baryon effectiveness
    vbar_max = np.max(vbar)
    r_bar_peak = r[np.argmax(vbar)]
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    
    baryon_effectiveness = vbar_max / v_flat if v_flat > 5 else np.nan
    
    # Where does V_bar peak relative to R_last?
    r_bar_frac = r_bar_peak / r[-1]
    
    # SB structure
    log_r = np.log10(r); log_I = np.log10(np.maximum(I, 1e-30))
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    mu_0 = sb[0]
    
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    if np.isfinite(baryon_effectiveness) and np.isfinite(concentration):
        gal_consp.append({
            'galaxy': gal, 'baryon_eff': baryon_effectiveness,
            'r_bar_frac': r_bar_frac, 'r_bar_peak': r_bar_peak,
            'concentration': concentration, 'sb_slope': sb_slope, 'mu_0': mu_0,
            'v_flat': v_flat, 'log_mbar': log_mbar, 'vbar_max': vbar_max
        })

gc = pd.DataFrame(gal_consp).replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Galaxies: {len(gc)}")
print(f"  Baryon effectiveness range: {gc['baryon_eff'].min():.2f} — {gc['baryon_eff'].max():.2f}")
print(f"  (1.0 = baryons fully explain V_flat; <1 = DM needed)")

# Zero-order
print(f"\n  ZERO-ORDER with baryon effectiveness:")
for pred in ['concentration', 'sb_slope', 'mu_0', 'log_mbar', 'v_flat']:
    rho, p = stats.spearmanr(gc[pred], gc['baryon_eff'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL
print(f"\n  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0']:
    X_ctrl = np.column_stack([gc['log_mbar'].values, gc['v_flat'].values, np.ones(len(gc))])
    c_y, _, _, _ = lstsq(X_ctrl, gc['baryon_eff'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, gc[pred].values, rcond=None)
    res_y = gc['baryon_eff'].values - X_ctrl @ c_y
    res_x = gc[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Bootstrap the strongest partial
best_pred = 'mu_0'  # will check which is strongest
for pred in ['concentration', 'sb_slope', 'mu_0']:
    X_ctrl = np.column_stack([gc['log_mbar'].values, gc['v_flat'].values, np.ones(len(gc))])
    c_y, _, _, _ = lstsq(X_ctrl, gc['baryon_eff'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, gc[pred].values, rcond=None)
    res_y = gc['baryon_eff'].values - X_ctrl @ c_y
    res_x = gc[pred].values - X_ctrl @ c_x
    rho_test, _ = stats.spearmanr(res_x, res_y)
    
    if abs(rho_test) > 0.2:
        np.random.seed(42)
        boot = []
        for _ in range(3000):
            idx = np.random.choice(len(gc), len(gc), replace=True)
            cy, _, _, _ = lstsq(X_ctrl[idx], gc['baryon_eff'].values[idx], rcond=None)
            cx, _, _, _ = lstsq(X_ctrl[idx], gc[pred].values[idx], rcond=None)
            ry = gc['baryon_eff'].values[idx] - X_ctrl[idx] @ cy
            rx = gc[pred].values[idx] - X_ctrl[idx] @ cx
            r_b, _ = stats.spearmanr(rx, ry)
            boot.append(r_b)
        boot = np.array(boot)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n  BOOTSTRAP for {pred}:")
        print(f"    Mean ρ = {np.mean(boot):+.3f}, 95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        print(f"    CI excludes 0? {'YES ✓' if (ci_lo > 0 or ci_hi < 0) else 'NO'}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK 4 SUMMARY")
print("=" * 70)
