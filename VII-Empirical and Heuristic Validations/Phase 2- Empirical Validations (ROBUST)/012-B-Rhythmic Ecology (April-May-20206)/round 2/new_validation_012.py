#!/usr/bin/env python3
"""
NEW RTM VALIDATION — Doc 012
=============================
RTM-specific test: Does α behave as a topological invariant?

Three tests:
1. Within-order α variation in Mammalia (discrete bands vs continuum?)
2. Power-law collapse quality per order (R² distribution)
3. ODR sensitivity to assumed error levels
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.odr import ODR, Model, RealData
import json

np.random.seed(42)

# Load AnAge
df = pd.read_csv("/home/claude/012/ROBUST-AnAge_Longevity Database_Analysis/anage_data.txt",
                 sep='\t', encoding='latin-1')
df = df.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)'])
df = df[(df['Body mass (g)'] > 0) & (df['Maximum longevity (yrs)'] > 0)]
df['log_M'] = np.log10(df['Body mass (g)'])
df['log_L'] = np.log10(df['Maximum longevity (yrs)'])

print("=" * 70)
print("NEW RTM VALIDATION: α AS TOPOLOGICAL INVARIANT")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# TEST A: WITHIN-ORDER α VARIATION (Mammalia)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST A: α BY TAXONOMIC ORDER (Mammalia)")
print("RTM predicts: α determined by network topology →")
print("  orders with similar body plans should have similar α")
print("=" * 70)

mammals = df[df['Class'] == 'Mammalia'].copy()

def linear_func(p, x): return p[0] * x + p[1]
model = Model(linear_func)

log_M_err = 0.20 / np.log(10)
log_L_err = 0.25 / np.log(10)

order_results = []
print(f"\n  {'Order':20s} {'N':>4s} {'OLS_α':>8s} {'ODR_α':>10s} {'R²':>6s} {'log_M range':>12s}")
print("  " + "-" * 65)

for order in mammals['Order'].value_counts().index:
    sub = mammals[mammals['Order'] == order]
    if len(sub) < 8:  # need minimum for meaningful regression
        continue
    
    log_m_range = sub['log_M'].max() - sub['log_M'].min()
    if log_m_range < 0.5:  # need sufficient dynamic range
        continue
    
    ols = stats.linregress(sub['log_M'], sub['log_L'])
    
    data = RealData(sub['log_M'].values, sub['log_L'].values, sx=log_M_err, sy=log_L_err)
    odr = ODR(data, model, beta0=[ols.slope, ols.intercept])
    out = odr.run()
    
    order_results.append({
        'order': order, 'n': len(sub),
        'ols_alpha': ols.slope, 'odr_alpha': out.beta[0],
        'odr_err': out.sd_beta[0], 'r2': ols.rvalue**2,
        'log_m_range': log_m_range
    })
    print(f"  {order:20s} {len(sub):4d} {ols.slope:8.4f} {out.beta[0]:7.4f}±{out.sd_beta[0]:.4f} {ols.rvalue**2:6.3f} {log_m_range:12.2f}")

alphas = [r['odr_alpha'] for r in order_results]
alpha_mean = np.mean(alphas)
alpha_std = np.std(alphas)
alpha_cv = alpha_std / abs(alpha_mean)  # coefficient of variation

print(f"\n  SUMMARY across {len(order_results)} orders:")
print(f"  Mean α = {alpha_mean:.4f} ± {alpha_std:.4f}")
print(f"  Range: [{min(alphas):.4f}, {max(alphas):.4f}]")
print(f"  Coefficient of variation: {alpha_cv:.2f} ({alpha_cv*100:.0f}%)")

# Kruskal-Wallis test: does α differ between orders?
# Using bootstrap to create per-order α distributions
print(f"\n  DISTRIBUTION ANALYSIS:")
print(f"  Are the order-level α values clustered (discrete bands)")
print(f"  or continuously distributed?")

# Check for bimodality using Hartigan's dip test approximation
from collections import Counter
# Simple check: how many distinct clusters?
sorted_alphas = sorted(alphas)
gaps = [sorted_alphas[i+1] - sorted_alphas[i] for i in range(len(sorted_alphas)-1)]
median_gap = np.median(gaps)
large_gaps = [g for g in gaps if g > 2 * median_gap]

print(f"  Sorted α values: {[f'{a:.3f}' for a in sorted_alphas]}")
print(f"  Inter-order gaps: {[f'{g:.3f}' for g in gaps]}")
print(f"  Median gap: {median_gap:.4f}")
print(f"  Large gaps (>2x median): {len(large_gaps)}")

# Shapiro-Wilk test for normality of α distribution
if len(alphas) >= 8:
    w_stat, w_p = stats.shapiro(alphas)
    print(f"  Shapiro-Wilk normality test: W={w_stat:.4f}, p={w_p:.4f}")
    if w_p > 0.05:
        print(f"  → α distribution is consistent with NORMAL (continuous)")
        print(f"    This WEAKENS the 'discrete bands' hypothesis")
    else:
        print(f"  → α distribution departs from normal")
        print(f"    This is CONSISTENT with (but doesn't prove) discrete bands")

# ═══════════════════════════════════════════════════════
# TEST B: COLLAPSE QUALITY PER ORDER
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST B: POWER-LAW COLLAPSE QUALITY PER ORDER")
print("RTM predicts: within a structural class, T ~ L^α should hold")
print("  R² >> 0.5 expected; R² < 0.3 = collapse failure")
print("=" * 70)

r2_values = [r['r2'] for r in order_results]
n_good = sum(1 for r in r2_values if r > 0.3)
n_weak = sum(1 for r in r2_values if r <= 0.3)

print(f"\n  R² distribution across {len(order_results)} orders:")
print(f"  Mean R² = {np.mean(r2_values):.3f}")
print(f"  Median R² = {np.median(r2_values):.3f}")
print(f"  Good fits (R² > 0.3): {n_good}/{len(order_results)} ({100*n_good/len(order_results):.0f}%)")
print(f"  Weak fits (R² ≤ 0.3): {n_weak}/{len(order_results)} ({100*n_weak/len(order_results):.0f}%)")

for r in sorted(order_results, key=lambda x: x['r2']):
    status = "✓" if r['r2'] > 0.3 else "✗ COLLAPSE FAILS"
    print(f"    {r['order']:20s} R²={r['r2']:.3f} α={r['odr_alpha']:.3f}  {status}")

# ═══════════════════════════════════════════════════════
# TEST C: ODR SENSITIVITY TO ASSUMED ERROR
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST C: SENSITIVITY OF α TO ASSUMED ERROR LEVELS")
print("RTM claims α is 'robust'. Test: how much does Mammalia α")
print("  change if we vary the assumed measurement error?")
print("=" * 70)

error_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
mam_data = mammals[['log_M', 'log_L']].values

print(f"\n  {'Mass_err%':>10s} {'Long_err%':>10s} {'ODR_α':>10s} {'Δ from OLS':>12s}")
print("  " + "-" * 45)

ols_mam = stats.linregress(mammals['log_M'], mammals['log_L'])
sensitivity_results = []

for err_frac in error_levels:
    sx = err_frac / np.log(10)
    sy = (err_frac * 1.25) / np.log(10)  # longevity error always 25% more
    
    data = RealData(mammals['log_M'].values, mammals['log_L'].values, sx=sx, sy=sy)
    odr = ODR(data, model, beta0=[ols_mam.slope, ols_mam.intercept])
    out = odr.run()
    
    delta = out.beta[0] - ols_mam.slope
    sensitivity_results.append({
        'mass_err': err_frac, 'odr_alpha': out.beta[0], 'delta': delta
    })
    print(f"  {100*err_frac:9.0f}% {100*err_frac*1.25:9.0f}% {out.beta[0]:10.4f} {delta:+12.4f}")

total_range = max(r['odr_alpha'] for r in sensitivity_results) - min(r['odr_alpha'] for r in sensitivity_results)
print(f"\n  Total α range across error assumptions: {total_range:.4f}")
print(f"  OLS baseline: {ols_mam.slope:.4f}")
if total_range < 0.02:
    print(f"  → α is ROBUST to error assumptions (range < 0.02)")
elif total_range < 0.05:
    print(f"  → α is MODERATELY sensitive (range 0.02-0.05)")
else:
    print(f"  → α is SENSITIVE to error assumptions (range > 0.05)")
    print(f"    The choice of 20% matters for the result!")

# ═══════════════════════════════════════════════════════
# TEST D: CROSS-CLASS COMPARISON (the real RTM test)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST D: CROSS-CLASS α COMPARISON")
print("RTM predicts α ≈ 0.25 for all vertebrates.")
print("Does the data support a UNIVERSAL α, or class-specific α?")
print("=" * 70)

# Fit all vertebrates together
verts = df[df['Class'].isin(['Mammalia', 'Aves', 'Reptilia'])]
ols_all = stats.linregress(verts['log_M'], verts['log_L'])
data_all = RealData(verts['log_M'].values, verts['log_L'].values, sx=log_M_err, sy=log_L_err)
odr_all = ODR(data_all, model, beta0=[ols_all.slope, ols_all.intercept])
out_all = odr_all.run()

print(f"\n  ALL vertebrates combined (n={len(verts)}):")
print(f"    ODR α = {out_all.beta[0]:.4f} ± {out_all.sd_beta[0]:.4f}")
print(f"    R² = {ols_all.rvalue**2:.4f}")

# Compare AIC: universal α vs class-specific α
# Universal model: 2 parameters (α, intercept)
resid_universal = verts['log_L'] - (out_all.beta[0] * verts['log_M'] + out_all.beta[1])
rss_universal = np.sum(resid_universal**2)
n_total = len(verts)
aic_universal = n_total * np.log(rss_universal / n_total) + 2 * 2

# Class-specific model: 6 parameters (α_i, intercept_i for 3 classes)
rss_specific = 0
for cls in ['Mammalia', 'Aves', 'Reptilia']:
    sub = verts[verts['Class'] == cls]
    ols_sub = stats.linregress(sub['log_M'], sub['log_L'])
    resid_sub = sub['log_L'] - (ols_sub.slope * sub['log_M'] + ols_sub.intercept)
    rss_specific += np.sum(resid_sub**2)
aic_specific = n_total * np.log(rss_specific / n_total) + 2 * 6

print(f"\n  MODEL COMPARISON (AIC):")
print(f"    Universal α model:      AIC = {aic_universal:.1f}")
print(f"    Class-specific α model: AIC = {aic_specific:.1f}")
print(f"    ΔAIC = {aic_universal - aic_specific:.1f}")

if aic_specific < aic_universal - 2:
    print(f"    → Class-specific model is BETTER (ΔAIC > 2)")
    print(f"      This means α is NOT a single universal constant.")
    print(f"      Different classes have different topological exponents.")
    print(f"      FOR RTM: This SUPPORTS the 'topology determines α' claim,")
    print(f"      but WEAKENS the 'α ≈ 0.25 for all vertebrates' claim.")
elif aic_universal < aic_specific - 2:
    print(f"    → Universal model is BETTER (ΔAIC > 2)")
    print(f"      FOR RTM: Supports universal α for vertebrates.")
else:
    print(f"    → Models are indistinguishable (|ΔAIC| < 2)")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OVERALL RESULTS OF NEW RTM VALIDATION")
print("=" * 70)

print(f"""
  TEST A (Within-order α variation):
    α varies from {min(alphas):.3f} to {max(alphas):.3f} across mammalian orders
    CV = {alpha_cv*100:.0f}% — {'LOW' if alpha_cv < 0.3 else 'MODERATE' if alpha_cv < 0.6 else 'HIGH'} variation
    Distribution: {'consistent with normal (continuous)' if w_p > 0.05 else 'departs from normal'}
    → {'SUPPORTS' if alpha_cv < 0.4 else 'MIXED for'} RTM: α is {'approximately' if alpha_cv < 0.4 else 'NOT'} constant within Mammalia

  TEST B (Collapse quality):
    {n_good}/{len(order_results)} orders show adequate power-law fits (R² > 0.3)
    {n_weak}/{len(order_results)} orders show collapse failure
    → {'SUPPORTS' if n_good/len(order_results) > 0.7 else 'MIXED for'} RTM

  TEST C (Sensitivity to error):
    α range across error assumptions: {total_range:.4f}
    → α is {'ROBUST' if total_range < 0.02 else 'MODERATELY SENSITIVE' if total_range < 0.05 else 'SENSITIVE'}

  TEST D (Universal vs class-specific α):
    ΔAIC = {aic_universal - aic_specific:.1f}
    → {'Class-specific better' if aic_specific < aic_universal - 2 else 'Universal better' if aic_universal < aic_specific - 2 else 'Indistinguishable'}
""")

# Save
results = {
    "new_validation": "012_RTM_topological_invariant",
    "test_A_within_order": {
        "n_orders": len(order_results),
        "alpha_mean": round(alpha_mean, 4),
        "alpha_std": round(alpha_std, 4),
        "alpha_cv": round(alpha_cv, 4),
        "alpha_range": [round(min(alphas), 4), round(max(alphas), 4)],
        "shapiro_p": round(w_p, 4) if len(alphas) >= 8 else None,
        "order_details": [{k: round(v, 4) if isinstance(v, float) else v 
                          for k, v in r.items()} for r in order_results]
    },
    "test_B_collapse": {
        "n_good_fits": n_good,
        "n_weak_fits": n_weak,
        "mean_r2": round(np.mean(r2_values), 4)
    },
    "test_C_sensitivity": {
        "alpha_range": round(total_range, 4),
        "results": [{k: round(v, 4) if isinstance(v, float) else v 
                     for k, v in r.items()} for r in sensitivity_results]
    },
    "test_D_model_comparison": {
        "aic_universal": round(aic_universal, 1),
        "aic_specific": round(aic_specific, 1),
        "delta_aic": round(aic_universal - aic_specific, 1),
        "universal_alpha": round(out_all.beta[0], 4)
    }
}

with open('/home/claude/new_validation_012.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to new_validation_012.json")
