# RTM Flank Attack: RAR Scatter + Diversity Problem

**Date:** April 28, 2026
**Data:** SPARC table2.dat (135 galaxies, 3,174 radius points)

---

## Strategy

Instead of trying to REPLACE dark matter (which failed), test if baryonic STRUCTURE predicts the PATTERN of the discrepancy. Two flanks:

- **Flank A:** Does local SB structure reduce scatter in the Radial Acceleration Relation?
- **Flank B:** At fixed V_flat, does SB profile shape predict rotation curve shape? (The "diversity problem")

---

## Flank A: RAR Scatter Reduction

### Result: WEAK SIGNAL

| Metric | Value |
|--------|-------|
| RAR baseline scatter | 0.184 dex |
| After SB correction | 0.184 dex |
| Reduction | **0.1%** (negligible) |
| Local gradient ρ with RAR residual | -0.029 (ns) |

The local SB gradient does not meaningfully reduce RAR scatter. The RAR is already so tight that local structure adds virtually nothing.

**Within-galaxy analysis** shows a weak but statistically significant tendency: mean within-galaxy ρ = +0.107, 64% positive, t-test p = 0.01. This means that within individual galaxies, radii with steeper SB gradients tend to have slightly higher RAR residuals. The effect is real but tiny.

**For RTM: MARGINALLY POSITIVE.** The within-galaxy signal exists but is too small to be operationally useful.

---

## Flank B: The Diversity Problem

### Result: GENUINE POSITIVE FINDING

The "diversity problem" in galaxy dynamics: why do galaxies with the same V_flat show different inner rotation curve shapes? ΛCDM requires halo parameter tuning. Can baryonic structure predict this diversity?

**Zero-order correlations (131 galaxies):**

| Correlation | ρ | p |
|-------------|---|---|
| SB_slope → inner_slope | -0.428 | 3.5 × 10⁻⁷ *** |
| SB_slope → rise_ratio | +0.531 | 6.7 × 10⁻¹¹ *** |
| SB_conc → inner_slope | +0.374 | 1.1 × 10⁻⁵ *** |
| SB_conc → rise_ratio | -0.442 | 1.2 × 10⁻⁷ *** |

All highly significant. But these could be driven by mass (heavier galaxies have both steeper SB and different RC shapes).

**The critical test — PARTIAL correlations controlling for V_flat:**

| Correlation | Partial ρ | p | Significant? |
|-------------|-----------|---|-------------|
| SB_conc → inner_slope | V_flat | -0.046 | 0.60 | NO |
| SB_slope → inner_slope | V_flat | -0.166 | 0.059 | Marginal |
| **SB_slope → rise_ratio** | **V_flat** | **+0.329** | **0.0001** | **YES ★★★** |

**The key finding:** At fixed V_flat (i.e., at fixed "total mass"), galaxies with steeper SB falloff (more concentrated light profiles) have higher rise ratios — their inner velocities are faster relative to the outer flat portion. This is exactly what RTM predicts: more concentrated baryonic structure → different inner dynamics.

**Multivariate model:**

| Model | R² | ΔR² | F-test p |
|-------|-----|-----|----------|
| V_flat only | 0.326 | — | — |
| V_flat + SB shape | 0.365 | **+0.039** | **0.022 ★** |

Structure adds 3.9% explained variance to inner RC shape prediction, significant at p = 0.022.

**Within V_flat bins:**

| Bin | n | ρ(SB_slope, rise_ratio) | p |
|-----|---|------------------------|---|
| Slow (16-83 km/s) | 44 | +0.226 | 0.14 (ns) |
| Medium (83-163) | 43 | +0.320 | 0.037 ★ |
| Fast (168-331) | 44 | +0.348 | 0.021 ★ |

The effect holds in medium and fast rotators but not in slow (dwarf) galaxies. This makes physical sense: dwarf galaxies have diffuse, featureless SB profiles with little structural variation to leverage.

---

## What This Means

### This IS the diversity problem, partially solved

ΛCDM cannot explain why galaxies at fixed halo mass show different inner RC shapes without invoking halo concentration scatter. RTM's SB slope predicts this diversity with ρ = +0.329 (p = 0.0001) after controlling for mass. This is a **genuine empirical contribution**.

### Magnitude matters

ΔR² = 3.9% is real and significant but modest. Structure explains about 4% of the inner-shape variance beyond what mass provides. This is consistent with our earlier finding (8.5% for mass discrepancy). The effect is secondary — significant but not dominant.

### What it does NOT do

It does not replace dark matter. It does not predict absolute velocities. It does not compete with NFW on curve fitting. It shows that baryonic geometry carries information about the mass discrepancy pattern that mass alone misses.

---

*All computations reproducible via rtm_flank_attack.py.*
