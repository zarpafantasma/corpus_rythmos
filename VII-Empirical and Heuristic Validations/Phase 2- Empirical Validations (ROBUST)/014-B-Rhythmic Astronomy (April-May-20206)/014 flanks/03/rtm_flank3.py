#!/usr/bin/env python3
"""
RTM FLANK 3: THE UNEXPECTED ANGLES
====================================
Three attacks the front door doesn't see:

FLANK A: "Where does DM kick in?"
  → Can SB structure predict the RADIUS where mass discrepancy begins?
  → If yes: light tells you WHERE dark matter starts dominating.

FLANK B: "Cusp-Core from light alone"
  → From V_obs and V_bar, derive the inferred DM density slope.
  → Can SB structure predict whether a galaxy has a cusp or a core?
  → This is the biggest unsolved problem in ΛCDM small-scale structure.

FLANK C: "The acceleration coupling"  
  → At which acceleration does each galaxy transition to DM-dominated?
  → Does SB structure predict deviations from the universal g† = 1.2e-10?
"""
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
from scipy.stats import f as f_dist
import json, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/claude/astro/table2.dat', sep=r'\s+', comment='#',
                 on_bad_lines='skip', skiprows=30,
                 names=['Name','Distance','Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB'])
df['Name'] = df['Name'].astype(str).str.replace(' ','').str.upper()
for c in ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','errSB','Distance']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

Upsilon_disk = 0.5; Upsilon_bul = 0.7

print("=" * 70)
print("RTM FLANK 3: THREE UNEXPECTED ANGLES")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# PRE-COMPUTATION: Per-galaxy profiles
# ═══════════════════════════════════════════════════════
galaxy_data = {}

for gal in df['Name'].unique():
    if gal == 'NAN' or len(gal) < 2: continue
    sub = df[df['Name']==gal].copy()
    sub = sub.dropna(subset=['Rad','Vobs','Vgas','Vdisk','SBdisk'])
    sub = sub[(sub['Rad'] > 0) & (sub['Vobs'] > 0)]
    if len(sub) < 10: continue
    
    r = sub['Rad'].values
    vobs = sub['Vobs'].values
    vgas = sub['Vgas'].values
    vdisk = sub['Vdisk'].values
    vbul = sub['Vbul'].values
    sb = sub['SBdisk'].values
    errv = sub['errV'].values
    
    vbar2 = np.abs(vgas)*vgas + Upsilon_disk*np.abs(vdisk)*vdisk + Upsilon_bul*np.abs(vbul)*vbul
    vbar = np.sqrt(np.abs(vbar2))
    
    # DM velocity: V_DM² = V_obs² - V_bar²
    vdm2 = vobs**2 - np.abs(vbar2)
    vdm = np.sqrt(np.maximum(vdm2, 0))
    
    # Mass discrepancy profile
    D = np.where(np.abs(vbar2) > 100, vobs**2 / np.abs(vbar2), np.nan)
    
    # Accelerations (km²/s²/kpc)
    g_obs = vobs**2 / r
    g_bar = np.abs(vbar2) / r
    
    # SB intensity
    I = 10**(-0.4 * sb)
    log_r = np.log10(r)
    log_I = np.log10(np.maximum(I, 1e-30))
    
    galaxy_data[gal] = {
        'r': r, 'vobs': vobs, 'vbar': vbar, 'vdm': vdm, 'vdm2': vdm2,
        'D': D, 'g_obs': g_obs, 'g_bar': g_bar,
        'I': I, 'sb': sb, 'log_r': log_r, 'log_I': log_I, 'errv': errv
    }

print(f"  Galaxies loaded: {len(galaxy_data)}")

# ═══════════════════════════════════════════════════════
# FLANK A: WHERE DOES DARK MATTER KICK IN?
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLANK A: CAN LIGHT PREDICT WHERE DARK MATTER BEGINS?")
print("Find r_transition where D(r) first exceeds 1.5")
print("Test if SB structure predicts r_transition at fixed mass")
print("=" * 70)

flank_a = []
for gal, gd in galaxy_data.items():
    D = gd['D']
    r = gd['r']
    I = gd['I']
    log_r = gd['log_r']
    log_I = gd['log_I']
    
    # Find transition radius where D first exceeds 1.5
    valid_D = ~np.isnan(D)
    if valid_D.sum() < 5: continue
    
    r_trans = np.nan
    for i in range(len(D)):
        if valid_D[i] and D[i] > 1.5:
            r_trans = r[i]
            break
    
    if np.isnan(r_trans): continue
    
    # Normalize: fraction of total extent
    r_trans_frac = r_trans / r.max()
    
    # SB structure parameters
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    
    # SB scale length (exponential fit: I = I0 * exp(-r/h))
    valid_I = I > 0
    if valid_I.sum() >= 3:
        ols_exp = stats.linregress(r[valid_I], np.log(I[valid_I]))
        h_scale = -1.0 / ols_exp.slope if ols_exp.slope < -0.01 else np.nan
    else:
        h_scale = np.nan
    
    # r_trans in units of scale length
    r_trans_h = r_trans / h_scale if h_scale and h_scale > 0 else np.nan
    
    # Total "mass" proxy
    m_bar = np.trapezoid(I * r, r) if len(r) > 1 else np.nan
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    # V_flat
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(gd['vobs'][-n_out:])
    
    flank_a.append({
        'galaxy': gal, 'r_trans': r_trans, 'r_trans_frac': r_trans_frac,
        'r_trans_h': r_trans_h, 'h_scale': h_scale,
        'concentration': concentration, 'sb_slope': sb_slope,
        'log_mbar': log_mbar, 'v_flat': v_flat,
        'log_r_trans': np.log10(r_trans)
    })

fa = pd.DataFrame(flank_a).replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Galaxies with transition: {len(fa)}")
print(f"  r_trans range: {fa['r_trans'].min():.1f} — {fa['r_trans'].max():.1f} kpc")
print(f"  r_trans/h range: {fa['r_trans_h'].min():.1f} — {fa['r_trans_h'].max():.1f} scale lengths")

# Zero-order
print(f"\n  ZERO-ORDER correlations with log(r_trans):")
for pred in ['concentration', 'sb_slope', 'h_scale', 'log_mbar', 'v_flat']:
    valid = fa.dropna(subset=[pred])
    rho, p = stats.spearmanr(valid[pred], valid['log_r_trans'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL: controlling for mass
print(f"\n  PARTIAL correlations (controlling log_mbar):")
for pred in ['concentration', 'sb_slope', 'h_scale']:
    valid = fa.dropna(subset=[pred, 'log_r_trans', 'log_mbar'])
    o1 = stats.linregress(valid['log_mbar'], valid['log_r_trans'])
    o2 = stats.linregress(valid['log_mbar'], valid[pred])
    res_y = valid['log_r_trans'].values - (o1.slope * valid['log_mbar'].values + o1.intercept)
    res_x = valid[pred].values - (o2.slope * valid['log_mbar'].values + o2.intercept)
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# r_trans in scale lengths vs concentration
rho_rh, p_rh = stats.spearmanr(fa['concentration'], fa['r_trans_h'])
print(f"\n  r_trans/h vs concentration: ρ = {rho_rh:+.3f}, p = {p_rh:.2e}")

# ═══════════════════════════════════════════════════════
# FLANK B: CUSP vs CORE FROM LIGHT ALONE
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK B: CAN LIGHT PREDICT CUSP vs CORE?")
print("Derive inner DM density slope from V_DM(r)")
print("Test if SB structure predicts the DM inner slope")
print("=" * 70)

flank_b = []
for gal, gd in galaxy_data.items():
    r = gd['r']
    vdm = gd['vdm']
    vdm2 = gd['vdm2']
    I = gd['I']
    log_r = gd['log_r']
    log_I = gd['log_I']
    vobs = gd['vobs']
    
    # Inner region (first 40%)
    n_inner = max(4, int(0.4 * len(r)))
    
    # DM "rotation curve" inner slope
    # For NFW: V_DM rises as r^0.5 (cusp) at small r
    # For cored: V_DM rises as r^1.0 at small r
    vdm_inner = vdm[:n_inner]
    r_inner = r[:n_inner]
    
    # Need positive V_DM
    mask = vdm_inner > 5  # minimum 5 km/s to be meaningful
    if mask.sum() < 4: continue
    
    log_vdm = np.log10(vdm_inner[mask])
    log_r_in = np.log10(r_inner[mask])
    
    dm_inner_slope = stats.linregress(log_r_in, log_vdm).slope
    
    # This slope relates to DM density:
    # V_DM ∝ r^β → ρ_DM ∝ r^(2β-2)
    # Cusp (NFW): β ≈ 0.5 → ρ ∝ r^-1
    # Core: β ≈ 1.0 → ρ ∝ r^0 (constant density)
    dm_density_slope = 2 * dm_inner_slope - 2
    
    # SB structure
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    
    # Inner SB slope specifically
    sb_inner_slope = stats.linregress(log_r[:n_inner], log_I[:n_inner]).slope
    
    # V_flat
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    
    # Mass proxy
    m_bar = np.trapezoid(I * r, r) if len(r) > 1 else np.nan
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    # Baryon dominance in inner region
    vbar_inner = np.median(gd['vbar'][:n_inner])
    bar_dominance = vbar_inner / (np.median(vobs[:n_inner]) + 1e-10)
    
    if np.isfinite(dm_inner_slope) and np.isfinite(concentration):
        flank_b.append({
            'galaxy': gal, 'dm_inner_slope': dm_inner_slope,
            'dm_density_slope': dm_density_slope,
            'is_core': dm_inner_slope > 0.75,  # closer to linear rise
            'is_cusp': dm_inner_slope < 0.5,    # closer to NFW
            'concentration': concentration, 'sb_slope': sb_slope,
            'sb_inner_slope': sb_inner_slope,
            'v_flat': v_flat, 'log_mbar': log_mbar,
            'bar_dominance': bar_dominance
        })

fb = pd.DataFrame(flank_b).replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Galaxies with inner DM slope: {len(fb)}")
print(f"  DM inner slope range: {fb['dm_inner_slope'].min():.2f} — {fb['dm_inner_slope'].max():.2f}")
print(f"  Cuspy (β<0.5): {fb['is_cusp'].sum()} ({100*fb['is_cusp'].mean():.0f}%)")
print(f"  Cored (β>0.75): {fb['is_core'].sum()} ({100*fb['is_core'].mean():.0f}%)")
print(f"  Intermediate: {len(fb) - fb['is_cusp'].sum() - fb['is_core'].sum()}")

# Zero-order
print(f"\n  ZERO-ORDER correlations with DM inner slope:")
for pred in ['concentration', 'sb_slope', 'sb_inner_slope', 'v_flat', 'log_mbar', 'bar_dominance']:
    rho, p = stats.spearmanr(fb[pred], fb['dm_inner_slope'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL: controlling for mass and v_flat
print(f"\n  PARTIAL (controlling log_mbar + v_flat):")
for pred in ['concentration', 'sb_slope', 'sb_inner_slope', 'bar_dominance']:
    X_ctrl = np.column_stack([fb['log_mbar'].values, fb['v_flat'].values, np.ones(len(fb))])
    
    c_y, _, _, _ = lstsq(X_ctrl, fb['dm_inner_slope'].values, rcond=None)
    c_x, _, _, _ = lstsq(X_ctrl, fb[pred].values, rcond=None)
    
    res_y = fb['dm_inner_slope'].values - X_ctrl @ c_y
    res_x = fb[pred].values - X_ctrl @ c_x
    
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Can SB classify cusp vs core?
cusp_conc = fb[fb['is_cusp']]['concentration'].values
core_conc = fb[fb['is_core']]['concentration'].values
if len(cusp_conc) >= 5 and len(core_conc) >= 5:
    u_cc, p_cc = stats.mannwhitneyu(cusp_conc, core_conc, alternative='two-sided')
    d_cc = (cusp_conc.mean() - core_conc.mean()) / np.sqrt((cusp_conc.var(ddof=1) + core_conc.var(ddof=1))/2)
    print(f"\n  CUSP vs CORE classification by concentration:")
    print(f"    Cuspy galaxies: concentration = {cusp_conc.mean():.2f} ± {cusp_conc.std():.2f}")
    print(f"    Cored galaxies: concentration = {core_conc.mean():.2f} ± {core_conc.std():.2f}")
    print(f"    Cohen's d = {d_cc:+.3f}, Mann-Whitney p = {p_cc:.4f}")

# ═══════════════════════════════════════════════════════
# FLANK C: ACCELERATION COUPLING
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FLANK C: THE ACCELERATION SCALE")
print("Does SB structure predict the local acceleration where D(r)=2?")
print("i.e., where DM = baryons in force contribution")
print("=" * 70)

flank_c = []
for gal, gd in galaxy_data.items():
    D = gd['D']
    g_bar = gd['g_bar']
    g_obs = gd['g_obs']
    r = gd['r']
    I = gd['I']
    log_r = gd['log_r']
    log_I = gd['log_I']
    vobs = gd['vobs']
    
    valid = ~np.isnan(D) & (g_bar > 0)
    if valid.sum() < 5: continue
    
    # Find g_bar at which D crosses 2.0
    g_cross = np.nan
    for i in range(len(D)-1):
        if valid[i] and valid[i+1]:
            if (D[i] <= 2.0 and D[i+1] > 2.0) or (D[i] >= 2.0 and D[i+1] < 2.0):
                # Linear interpolation
                frac = (2.0 - D[i]) / (D[i+1] - D[i]) if D[i+1] != D[i] else 0.5
                g_cross = g_bar[i] + frac * (g_bar[i+1] - g_bar[i])
                break
    
    if np.isnan(g_cross) or g_cross <= 0: continue
    
    # SB structure
    n30 = max(2, int(0.3*len(I)))
    concentration = np.log10(np.median(I[:n30]) / (np.median(I[-n30:]) + 1e-30))
    sb_slope = stats.linregress(log_r, log_I).slope
    
    n_out = max(3, int(0.4*len(r)))
    v_flat = np.median(vobs[-n_out:])
    m_bar = np.trapezoid(I * r, r) if len(r) > 1 else np.nan
    log_mbar = np.log10(m_bar) if m_bar > 0 else np.nan
    
    flank_c.append({
        'galaxy': gal, 'log_g_cross': np.log10(g_cross),
        'g_cross': g_cross,
        'concentration': concentration, 'sb_slope': sb_slope,
        'v_flat': v_flat, 'log_mbar': log_mbar
    })

fc = pd.DataFrame(flank_c).replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Galaxies with g_cross: {len(fc)}")
print(f"  log(g_cross) range: {fc['log_g_cross'].min():.2f} — {fc['log_g_cross'].max():.2f}")
print(f"  (McGaugh g† ≈ log(3690) = 3.57 in our units)")

# Zero-order
print(f"\n  ZERO-ORDER:")
for pred in ['concentration', 'sb_slope', 'log_mbar', 'v_flat']:
    rho, p = stats.spearmanr(fc[pred], fc['log_g_cross'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ = {rho:+.3f}, p = {p:.2e} {sig}")

# PARTIAL
print(f"\n  PARTIAL (controlling log_mbar):")
for pred in ['concentration', 'sb_slope']:
    o1 = stats.linregress(fc['log_mbar'], fc['log_g_cross'])
    o2 = stats.linregress(fc['log_mbar'], fc[pred])
    res_y = fc['log_g_cross'].values - (o1.slope * fc['log_mbar'].values + o1.intercept)
    res_x = fc[pred].values - (o2.slope * fc['log_mbar'].values + o2.intercept)
    rho, p = stats.spearmanr(res_x, res_y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {pred:18s}: ρ_partial = {rho:+.3f}, p = {p:.2e} {sig}")

# Scatter in g_cross: is it universal?
print(f"\n  Is g_cross universal?")
print(f"    Mean log(g_cross) = {fc['log_g_cross'].mean():.3f} ± {fc['log_g_cross'].std():.3f}")
print(f"    CV = {fc['log_g_cross'].std()/abs(fc['log_g_cross'].mean())*100:.1f}%")
print(f"    After controlling for concentration:")
o_gc = stats.linregress(fc['concentration'], fc['log_g_cross'])
resid_gc = fc['log_g_cross'] - (o_gc.slope * fc['concentration'] + o_gc.intercept)
print(f"    Residual scatter = {resid_gc.std():.3f} (was {fc['log_g_cross'].std():.3f})")
reduction = 100 * (1 - resid_gc.std() / fc['log_g_cross'].std())
print(f"    Scatter reduction: {reduction:.1f}%")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("SUMMARY — FLANK 3 RESULTS")
print("=" * 70)

results = {
    "flank_A_dm_transition": {"n_galaxies": len(fa)},
    "flank_B_cusp_core": {"n_galaxies": len(fb)},
    "flank_C_acceleration": {"n_galaxies": len(fc)}
}

with open('/home/claude/rtm_flank3_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nResults saved.")
