#!/usr/bin/env python3
"""
RTM FLANK 6: GAS vs STARS — THE GEOMETRIC DECOMPOSITION
=========================================================
SPARC gives Vgas and Vdisk SEPARATELY. Gas and stars have fundamentally
different geometries: gas extends further, is diffuse, pressure-supported.
Stars are in a thin concentrated disk.

If GEOMETRY matters (RTM), then:
- Gas fraction should modulate the baryon-DM conspiracy
- Gas-dominated galaxies should couple differently to DM than star-dominated
- The relative geometry (gas extended vs stars concentrated) should matter

Also: THE BREAK SHARPNESS — how abruptly does the baryon→DM transition happen?
Sharp breaks (MOND-like) vs gradual transitions (NFW-like)?
Does SB structure predict this?
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
print("RTM FLANK 6: GAS vs STARS + BREAK SHARPNESS")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# COMPUTE PER-GALAXY GAS/STELLAR DECOMPOSITION
# ═══════════════════════════════════════════════════════
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

    # Separate baryonic components
    v2_gas = np.abs(vgas) * vgas
    v2_disk = Upsilon_disk * np.abs(vdisk) * vdisk
    v2_bul = Upsilon_bul * np.abs(vbul) * vbul
    vbar2 = v2_gas + v2_disk + v2_bul
    vbar = np.sqrt(np.abs(vbar2))

    # Gas fraction at each radius: f_gas(r) = V²_gas / V²_bar
    f_gas_local = np.abs(v2_gas) / (np.abs(vbar2) + 1e-10)

    # Global gas fraction (integrated)
    total_gas = np.sum(np.abs(v2_gas))
    total_bar = np.sum(np.abs(vbar2))
    f_gas_global = total_gas / (total_bar + 1e-10)

    # Gas dominance radius: where does gas first exceed stars?
    gas_dominates = np.abs(v2_gas) > (np.abs(v2_disk) + np.abs(v2_bul))
    r_gas_dom = np.nan
    for i in range(len(r)):
        if gas_dominates[i]:
            r_gas_dom = r[i]
            break
    r_gas_frac = r_gas_dom / r[-1] if not np.isnan(r_gas_dom) else np.nan

    # Geometric contrast: how different are gas and stellar distributions?
    # Gas extent relative to stellar extent
    v_gas_abs = np.sqrt(np.abs(v2_gas))
    v_disk_abs = np.sqrt(np.abs(v2_disk))
    if np.max(v_gas_abs) > 1 and np.max(v_disk_abs) > 1:
        # Half-mass radius of gas vs stars
        cum_gas = np.cumsum(np.abs(v2_gas) * r)
        cum_disk = np.cumsum(np.abs(v2_disk) * r)
        r_half_gas = r[np.searchsorted(cum_gas, cum_gas[-1]/2)] if cum_gas[-1] > 0 else np.nan
        r_half_disk = r[np.searchsorted(cum_disk, cum_disk[-1]/2)] if cum_disk[-1] > 0 else np.nan
        geometric_contrast = np.log10(r_half_gas / (r_half_disk + 1e-10)) if r_half_disk > 0 else np.nan
    else:
        geometric_contrast = np.nan

    # ─── BREAK SHARPNESS ───
    # How abruptly does f_bar = V²_bar/V²_obs transition?
    f_bar = np.abs(vbar2) / (vobs**2 + 1e-10)
    f_bar = np.clip(f_bar, 0, 2)

    # Break sharpness = max |d(f_bar)/d(log r)|
    log_r = np.log10(r)
    df_bar = np.gradient(f_bar, log_r)
    break_sharpness = np.max(np.abs(df_bar))

    # Break radius = where f_bar drops fastest
    break_idx = np.argmax(np.abs(df_bar))
    r_break = r[break_idx]
    r_break_frac = r_break / r[-1]

    # ─── V_DM for conspiracy ───
    vdm2 = vobs**2 - np.abs(vbar2)
    vdm = np.sqrt(np.maximum(vdm2, 0))
    
    # Conspiracy
    if np.max(vbar) > 5 and np.max(vdm) > 5:
        vbar_norm = vbar / np.max(vbar)
        vdm_norm = vdm / np.max(vdm)
        conspiracy_r, _ = stats.pearsonr(vbar_norm, vdm_norm)
    else:
        conspiracy_r = np.nan

    # ─── SB structure ───
    log_I = np.log10(np.maximum(I, 1e-30))
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    mu_0 = sb[0]
    sb_residuals = log_I - (sb_slope * log_r + stats.linregress(log_r, log_I).intercept)
    sb_roughness = np.std(sb_residuals)

    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    m_bar = np.trapezoid(I * r, r)
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan

    # Baryon effectiveness
    baryon_eff = np.max(vbar) / (v_flat + 1e-10)

    results.append({
        'galaxy': gal,
        'f_gas_global': f_gas_global,
        'r_gas_frac': r_gas_frac,
        'geometric_contrast': geometric_contrast,
        'break_sharpness': break_sharpness,
        'r_break_frac': r_break_frac,
        'conspiracy_r': conspiracy_r,
        'baryon_eff': baryon_eff,
        'concentration': concentration, 'sb_slope': sb_slope,
        'mu_0': mu_0, 'sb_roughness': sb_roughness,
        'v_flat': v_flat, 'log_mbar': log_mbar
    })

rdf = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan)

# ═══════════════════════════════════════════════════════
# FLANK A: GAS FRACTION MODULATES EVERYTHING?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FLANK A: DOES GAS FRACTION MODULATE THE BARYON-DM COUPLING?")
print("='*70")

valid = rdf.dropna(subset=['f_gas_global','conspiracy_r','baryon_eff','log_mbar','v_flat'])
print(f"\n  Galaxies: {len(valid)}")
print(f"  Gas fraction range: {valid['f_gas_global'].min():.2f} — {valid['f_gas_global'].max():.2f}")
print(f"  Median gas fraction: {valid['f_gas_global'].median():.2f}")

# Split: gas-rich vs gas-poor
median_fgas = valid['f_gas_global'].median()
gas_rich = valid[valid['f_gas_global'] > median_fgas]
gas_poor = valid[valid['f_gas_global'] <= median_fgas]

print(f"\n  Gas-rich (>{median_fgas:.2f}): n={len(gas_rich)}, mean conspiracy r = {gas_rich['conspiracy_r'].mean():.3f}")
print(f"  Gas-poor (<={median_fgas:.2f}): n={len(gas_poor)}, mean conspiracy r = {gas_poor['conspiracy_r'].mean():.3f}")
u_gf, p_gf = stats.mannwhitneyu(gas_rich['conspiracy_r'].dropna(), gas_poor['conspiracy_r'].dropna())
print(f"  Mann-Whitney: p = {p_gf:.4f}")

# Partial: gas fraction vs conspiracy, controlling mass + V_flat
print(f"\n  PARTIAL correlations of GAS FRACTION with dynamics (controlling M + V_flat):")
for target, label in [('conspiracy_r', 'Conspiracy strength'),
                       ('baryon_eff', 'Baryon effectiveness'),
                       ('break_sharpness', 'Break sharpness')]:
    v = valid.dropna(subset=[target])
    X_ctrl = np.column_stack([v['log_mbar'].values, v['v_flat'].values, np.ones(len(v))])
    c_y, _, _, _ = lstsq(X_ctrl, v[target].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, v['f_gas_global'].values, rcond=None)
    res_y = v[target].values - X_ctrl @ c_y
    res_x = v['f_gas_global'].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    f_gas → {label:25s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Does gas fraction modulate the SB→dynamics link?
print(f"\n  INTERACTION: Does gas fraction change how SB predicts dynamics?")
for group, label in [(gas_rich, 'Gas-rich'), (gas_poor, 'Gas-poor')]:
    g = group.dropna(subset=['concentration','baryon_eff','log_mbar','v_flat'])
    if len(g) < 10: continue
    X_ctrl = np.column_stack([g['log_mbar'].values, g['v_flat'].values, np.ones(len(g))])
    c_y, _, _, _ = lstsq(X_ctrl, g['baryon_eff'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, g['concentration'].values, rcond=None)
    res_y = g['baryon_eff'].values - X_ctrl @ c_y
    res_x = g['concentration'].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '*' if p < 0.05 else 'ns'
    print(f"    {label:10s} (n={len(g):3d}): ρ(conc→baryon_eff|M,V) = {rho:+.3f}, p={p:.4f} {sig}")

# ═══════════════════════════════════════════════════════
# FLANK B: GEOMETRIC CONTRAST (gas extent vs stellar extent)
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK B: GEOMETRIC CONTRAST (Gas vs Stellar Extent)")
print("Does the relative geometry of gas and stars predict DM coupling?")
print(f"{'='*70}")

valid_gc = rdf.dropna(subset=['geometric_contrast','conspiracy_r','log_mbar','v_flat'])
print(f"\n  Galaxies with geometric contrast: {len(valid_gc)}")
print(f"  Contrast range: {valid_gc['geometric_contrast'].min():.2f} — {valid_gc['geometric_contrast'].max():.2f}")
print(f"  (positive = gas more extended; negative = stars more extended)")

# Partial
for target, label in [('conspiracy_r', 'Conspiracy'), ('baryon_eff', 'Baryon eff'),
                       ('break_sharpness', 'Break sharpness')]:
    v = valid_gc.dropna(subset=[target])
    if len(v) < 15: continue
    X_ctrl = np.column_stack([v['log_mbar'].values, v['v_flat'].values, np.ones(len(v))])
    c_y, _, _, _ = lstsq(X_ctrl, v[target].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, v['geometric_contrast'].values, rcond=None)
    res_y = v[target].values - X_ctrl @ c_y
    res_x = v['geometric_contrast'].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    contrast → {label:20s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# ═══════════════════════════════════════════════════════
# FLANK C: BREAK SHARPNESS — MOND vs NFW SIGNATURE
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK C: BREAK SHARPNESS — How Abrupt Is the DM Transition?")
print("Sharp = MOND-like. Gradual = NFW-like.")
print("Does SB structure predict the transition style?")
print(f"{'='*70}")

valid_bs = rdf.dropna(subset=['break_sharpness','log_mbar','v_flat','concentration'])
print(f"\n  Galaxies: {len(valid_bs)}")
print(f"  Break sharpness range: {valid_bs['break_sharpness'].min():.2f} — {valid_bs['break_sharpness'].max():.2f}")

# Partial correlations
print(f"\n  PARTIAL (controlling M + V_flat):")
for pred in ['concentration', 'sb_slope', 'mu_0', 'sb_roughness', 'f_gas_global']:
    v = valid_bs.dropna(subset=[pred])
    if len(v) < 15: continue
    X_ctrl = np.column_stack([v['log_mbar'].values, v['v_flat'].values, np.ones(len(v))])
    c_y, _, _, _ = lstsq(X_ctrl, v['break_sharpness'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, v[pred].values, rcond=None)
    res_y = v['break_sharpness'].values - X_ctrl @ c_y
    res_x = v[pred].values - X_ctrl @ c_x
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Multivariate: can we predict break sharpness?
v = valid_bs.dropna(subset=['f_gas_global','sb_roughness'])
y = v['break_sharpness'].values
X1 = np.column_stack([v['log_mbar'].values, v['v_flat'].values, np.ones(len(v))])
c1, _, _, _ = lstsq(X1, y, rcond=None)
r2_mass = 1 - np.sum((y-X1@c1)**2)/np.sum((y-y.mean())**2)

X2 = np.column_stack([v['log_mbar'].values, v['v_flat'].values,
                       v['concentration'].values, v['mu_0'].values,
                       v['f_gas_global'].values, np.ones(len(v))])
c2, _, _, _ = lstsq(X2, y, rcond=None)
r2_full = 1 - np.sum((y-X2@c2)**2)/np.sum((y-y.mean())**2)

n = len(v)
ss1 = np.sum((y-X1@c1)**2); ss2 = np.sum((y-X2@c2)**2)
f_inc = ((ss1-ss2)/3)/(ss2/(n-6))
p_f = 1 - f_dist.cdf(f_inc, 3, n-6)

print(f"\n  MULTIVARIATE (break sharpness prediction):")
print(f"    M + V_flat only:           R² = {r2_mass:.4f}")
print(f"    M + V + struct + f_gas:    R² = {r2_full:.4f}")
print(f"    ΔR²:                       {r2_full-r2_mass:+.4f}")
print(f"    F-test:                    p = {p_f:.2e}")

# ═══════════════════════════════════════════════════════
# FLANK D: THE COMBINED MODEL
# Can gas fraction + SB structure together predict conspiracy?
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FLANK D: COMBINED MODEL — Gas + Structure → Conspiracy")
print(f"{'='*70}")

v = rdf.dropna(subset=['conspiracy_r','f_gas_global','concentration','mu_0','log_mbar','v_flat','sb_roughness'])
y = v['conspiracy_r'].values

X1 = np.column_stack([v['log_mbar'].values, v['v_flat'].values, np.ones(len(v))])
c1, _, _, _ = lstsq(X1, y, rcond=None)
r2_base = 1 - np.sum((y-X1@c1)**2)/np.sum((y-y.mean())**2)

X2 = np.column_stack([v['log_mbar'].values, v['v_flat'].values,
                       v['concentration'].values, v['mu_0'].values,
                       v['f_gas_global'].values, v['sb_roughness'].values,
                       np.ones(len(v))])
c2, _, _, _ = lstsq(X2, y, rcond=None)
r2_full = 1 - np.sum((y-X2@c2)**2)/np.sum((y-y.mean())**2)

n = len(v)
ss1 = np.sum((y-X1@c1)**2); ss2 = np.sum((y-X2@c2)**2)
f_inc = ((ss1-ss2)/4)/(ss2/(n-7))
p_f = 1 - f_dist.cdf(f_inc, 4, n-7)

print(f"\n  N = {n}")
print(f"  M + V_flat only:                         R² = {r2_base:.4f}")
print(f"  M + V + conc + μ₀ + f_gas + roughness:   R² = {r2_full:.4f}")
print(f"  ΔR²:                                     {r2_full-r2_base:+.4f}")
print(f"  F-test:                                  p = {p_f:.2e}")

print(f"\n  Coefficients:")
labels = ['log_mbar','v_flat','concentration','mu_0','f_gas','roughness','intercept']
for l, c in zip(labels, c2):
    print(f"    {l:18s}: {c:+.4f}")

# Bootstrap R²
np.random.seed(42)
boot_r2 = []
for _ in range(3000):
    idx = np.random.choice(n, n, replace=True)
    yb = y[idx]
    X2b = X2[idx]
    X1b = X1[idx]
    c1b, _, _, _ = lstsq(X1b, yb, rcond=None)
    c2b, _, _, _ = lstsq(X2b, yb, rcond=None)
    r2_1 = 1 - np.sum((yb-X1b@c1b)**2)/np.sum((yb-yb.mean())**2)
    r2_2 = 1 - np.sum((yb-X2b@c2b)**2)/np.sum((yb-yb.mean())**2)
    boot_r2.append(r2_2 - r2_1)

boot_r2 = np.array(boot_r2)
ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])
print(f"\n  Bootstrap ΔR² (3000 iterations):")
print(f"    Mean ΔR² = {np.mean(boot_r2):.4f}")
print(f"    95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"    CI excludes 0? {'YES ✓' if ci_lo > 0 else 'NO'}")

# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
