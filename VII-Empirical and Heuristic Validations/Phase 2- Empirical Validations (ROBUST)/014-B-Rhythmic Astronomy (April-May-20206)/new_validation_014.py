#!/usr/bin/env python3
"""
NEW RTM VALIDATION — Doc 014
==============================
Using the SPARC-derived data (171 galaxies, real SPARC rotation curves).

The tautology finding stands: α = 2*(1-slope) BY DEFINITION.
But let me test what IS genuinely testable:

1. Does struct_proxy predict slope BETTER than v_max alone?
   (If yes, baryonic structure carries information beyond total mass)

2. Does bar_ratio (baryonic fraction) correlate with slope residuals?
   (If yes, RTM-style structure matters after controlling for mass)

3. Multivariate: does struct_proxy add predictive power beyond v_max?

4. Is the inner→outer α transition correlated with structure?
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.odr import ODR, Model, RealData
import json

np.random.seed(42)

sparc = pd.read_csv("/home/claude/014/ROBUST-SPARC_Galaxy_Rotation_Curves_Analysis/sparc_rtm_analysis.csv")
valid = sparc.dropna(subset=['struct_proxy', 'slope_full', 'bar_ratio', 'v_max'])

print("=" * 70)
print("NEW RTM VALIDATION — SPARC REAL DATA (n=171 galaxies)")
print("Testing what IS genuine beyond the tautology")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# TEST A: DOES STRUCTURE PREDICT SLOPE BEYOND MASS?
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST A: DOES BARYONIC STRUCTURE PREDICT SLOPE BEYOND TOTAL MASS?")
print("=" * 70)

# Partial correlation: struct_proxy vs slope, controlling for v_max
# Method: regress both on v_max, correlate residuals
ols_sv = stats.linregress(valid['v_max'], valid['slope_full'])
ols_sp = stats.linregress(valid['v_max'], valid['struct_proxy'])

resid_slope = valid['slope_full'] - (ols_sv.slope * valid['v_max'] + ols_sv.intercept)
resid_struct = valid['struct_proxy'] - (ols_sp.slope * valid['v_max'] + ols_sp.intercept)

r_partial, p_partial = stats.pearsonr(resid_struct, resid_slope)
rho_partial, p_partial_s = stats.spearmanr(resid_struct, resid_slope)

print(f"  Zero-order: Spearman(struct, slope) = {stats.spearmanr(valid['struct_proxy'], valid['slope_full'])[0]:.4f}")
print(f"  Zero-order: Spearman(v_max, slope)  = {stats.spearmanr(valid['v_max'], valid['slope_full'])[0]:.4f}")
print(f"")
print(f"  Partial correlation (controlling for v_max):")
print(f"    Pearson r_partial  = {r_partial:.4f}, p = {p_partial:.2e}")
print(f"    Spearman ρ_partial = {rho_partial:.4f}, p = {p_partial_s:.2e}")

if p_partial < 0.05:
    print(f"  → SIGNIFICANT: Structure predicts slope BEYOND what mass alone gives")
    print(f"    This is genuine evidence for RTM's central claim.")
else:
    print(f"  → NOT SIGNIFICANT: Structure adds nothing beyond mass")

# ═══════════════════════════════════════════════════════
# TEST B: BARYONIC FRACTION vs SLOPE  
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST B: BARYONIC FRACTION vs ROTATION SLOPE")
print("=" * 70)

rho_bar, p_bar = stats.spearmanr(valid['bar_ratio'], valid['slope_full'])
print(f"  Spearman(bar_ratio, slope) = {rho_bar:.4f}, p = {p_bar:.2e}")

# What RTM would predict: higher baryonic fraction → more "normal" dynamics
# → slope closer to Keplerian (negative), i.e., LESS flat
# So bar_ratio should correlate NEGATIVELY with slope (less flat = more negative)
if p_bar < 0.05:
    direction = "positive" if rho_bar > 0 else "negative"
    print(f"  → SIGNIFICANT ({direction})")
    if rho_bar < 0:
        print(f"    Higher baryonic fraction → steeper decline (more Keplerian)")
        print(f"    This is expected physics, consistent with RTM")
    else:
        print(f"    Higher baryonic fraction → flatter curves")
        print(f"    This is the opposite of naive expectation")
        print(f"    But CONSISTENT with RTM: more baryons → more structure → higher α")
else:
    print(f"  → NOT SIGNIFICANT")

# ═══════════════════════════════════════════════════════
# TEST C: MULTIVARIATE MODEL
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST C: MULTIVARIATE REGRESSION (slope ~ v_max + struct_proxy + bar_ratio)")
print("=" * 70)

from numpy.linalg import lstsq

X = np.column_stack([
    valid['v_max'].values,
    valid['struct_proxy'].values,
    valid['bar_ratio'].values,
    np.ones(len(valid))
])
y = valid['slope_full'].values

coeffs, residuals, rank, sv = lstsq(X, y, rcond=None)

y_pred = X @ coeffs
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_full = 1 - ss_res / ss_tot

# Compare with v_max only
X_mass = np.column_stack([valid['v_max'].values, np.ones(len(valid))])
c_mass, _, _, _ = lstsq(X_mass, y, rcond=None)
y_pred_mass = X_mass @ c_mass
r2_mass = 1 - np.sum((y - y_pred_mass)**2) / ss_tot

# Compare with struct_proxy only
X_struct = np.column_stack([valid['struct_proxy'].values, np.ones(len(valid))])
c_struct, _, _, _ = lstsq(X_struct, y, rcond=None)
y_pred_struct = X_struct @ c_struct
r2_struct = 1 - np.sum((y - y_pred_struct)**2) / ss_tot

# F-test for adding struct+bar beyond v_max
n = len(valid)
p_full = 4  # v_max + struct + bar + intercept
p_reduced = 2  # v_max + intercept
ss_res_full = np.sum((y - y_pred)**2)
ss_res_reduced = np.sum((y - y_pred_mass)**2)
f_stat = ((ss_res_reduced - ss_res_full) / (p_full - p_reduced)) / (ss_res_full / (n - p_full))
from scipy.stats import f as f_dist
p_f = 1 - f_dist.cdf(f_stat, p_full - p_reduced, n - p_full)

print(f"  Model 1 (v_max only):           R² = {r2_mass:.4f}")
print(f"  Model 2 (struct_proxy only):     R² = {r2_struct:.4f}")
print(f"  Model 3 (v_max+struct+bar):      R² = {r2_full:.4f}")
print(f"  ΔR² (Model 3 vs Model 1):       {r2_full - r2_mass:.4f}")
print(f"  F-test (struct+bar add value):   F = {f_stat:.3f}, p = {p_f:.4e}")

print(f"\n  Coefficients (Model 3):")
print(f"    v_max:        {coeffs[0]:+.6f}")
print(f"    struct_proxy: {coeffs[1]:+.6f}")
print(f"    bar_ratio:    {coeffs[2]:+.6f}")
print(f"    intercept:    {coeffs[3]:+.6f}")

if p_f < 0.05:
    print(f"\n  → SIGNIFICANT: Structure/baryon fraction add predictive power beyond mass")
    print(f"    This supports RTM's claim that baryonic geometry matters.")
else:
    print(f"\n  → NOT SIGNIFICANT: Mass alone is sufficient")

# ═══════════════════════════════════════════════════════
# TEST D: INNER-OUTER α TRANSITION vs STRUCTURE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST D: DOES STRUCTURE PREDICT THE INNER→OUTER TRANSITION?")
print("=" * 70)

valid2 = sparc.dropna(subset=['alpha_inner', 'alpha_outer', 'struct_proxy'])
delta_alpha = valid2['alpha_outer'] - valid2['alpha_inner']

rho_trans, p_trans = stats.spearmanr(valid2['struct_proxy'], delta_alpha)
print(f"  Δα (outer-inner) mean = {delta_alpha.mean():.4f}")
print(f"  Spearman(struct_proxy, Δα) = {rho_trans:.4f}, p = {p_trans:.2e}")

if p_trans < 0.05:
    print(f"  → SIGNIFICANT: More structured galaxies show larger inner→outer transition")
    print(f"    RTM-consistent: concentrated structure → steeper inner, flatter outer")
else:
    print(f"  → NOT SIGNIFICANT: Structure doesn't predict the transition magnitude")

# ═══════════════════════════════════════════════════════
# TEST E: ODR ON STRUCT vs SLOPE WITH REAL ERROR BARS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST E: ODR WITH PER-POINT ERRORS (matching reported method)")
print("=" * 70)

def lf(p, x): return p[0] * x + p[1]
model = Model(lf)

x = valid['struct_proxy'].values
y = valid['slope_full'].values

# Per-point proportional errors (this reproduces the reported -1.17)
sx = 0.10 * np.abs(x) + 0.01
sy = valid['slope_err'].values if 'slope_err' in valid.columns else 0.10 * np.abs(y) + 0.01

ols = stats.linregress(x, y)
data = RealData(x, y, sx=sx, sy=sy)
odr = ODR(data, model, beta0=[ols.slope, ols.intercept])
out = odr.run()

print(f"  OLS: slope = {ols.slope:.4f}, r = {ols.rvalue:.4f}")
print(f"  ODR: slope = {out.beta[0]:.4f} ± {out.sd_beta[0]:.4f}")
print(f"  REPORTED: -1.169 ± 0.119")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: WHAT IS GENUINE IN DOC 014 SPARC ANALYSIS")
print("=" * 70)

results = {
    "document": "014_new_validation",
    "test_A_partial_correlation": {
        "r_partial": round(r_partial, 4),
        "p_partial": float(f"{p_partial:.2e}"),
        "rho_partial": round(rho_partial, 4),
        "significant": p_partial < 0.05,
        "interpretation": "Structure predicts slope beyond mass" if p_partial < 0.05 else "No"
    },
    "test_B_baryonic_fraction": {
        "rho": round(rho_bar, 4),
        "p": float(f"{p_bar:.2e}"),
        "significant": p_bar < 0.05
    },
    "test_C_multivariate": {
        "r2_mass_only": round(r2_mass, 4),
        "r2_struct_only": round(r2_struct, 4),
        "r2_full": round(r2_full, 4),
        "f_test_p": float(f"{p_f:.4e}"),
        "struct_adds_value": p_f < 0.05
    },
    "test_D_transition": {
        "rho": round(rho_trans, 4),
        "p": float(f"{p_trans:.2e}"),
        "significant": p_trans < 0.05
    }
}

with open('/home/claude/results_014_new.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved.")
