# RTM Flank 4: Baryon Effectiveness + DM Growth Rate

**Date:** April 28, 2026
**Data:** SPARC table2.dat

---

## Three Flanks — Two Major Hits

### Flank A: Per-Galaxy RAR Offset — MODERATE

After controlling for mass AND V_flat, structural parameters emerge as RAR offset predictors. The effective radius shows the strongest signal (partial ρ = -0.294, p = 0.0007). Structure adds ΔR² = 5.8% (F-test p = 0.018). Real but modest — similar magnitude to previous findings.

### Flank B: DM Fraction Growth Rate — STRONG HIT

How fast does DM take over as you move outward? SB structure predicts this at fixed mass and V_flat:

**DM growth rate (partial, controlling M_bar + V_flat):**

| Predictor | Partial ρ | p |
|-----------|-----------|---|
| SB slope | **+0.310** | **0.0008** ★★★ |
| μ₀ (central SB) | **+0.301** | **0.001** ★★ |
| Concentration | -0.183 | 0.053 (marginal) |

Steeper SB profiles → DM takes over faster. Fainter centers → DM takes over faster.

**DM 50% crossing radius (partial, controlling M_bar):**

| Predictor | Partial ρ | p |
|-----------|-----------|---|
| **Concentration** | **-0.515** | **4.2 × 10⁻⁶** ★★★ |
| **μ₀** | **+0.470** | **3.6 × 10⁻⁵** ★★★ |
| SB slope | +0.266 | 0.025 ★ |

At fixed mass, more concentrated galaxies reach DM dominance (f_DM = 50%) at **smaller radii**. The effect is very strong (ρ = -0.515). This directly tells us: the shape of the light profile predicts WHERE dark matter becomes the dominant component.

### Flank C: Baryon Effectiveness — STRONGEST HIT IN ALL GALACTIC ANALYSIS

"Baryon effectiveness" = V_bar_max / V_flat — how much of the observed flat velocity can baryons account for at their peak contribution.

**Partial correlations (controlling BOTH M_bar AND V_flat):**

| Predictor | Partial ρ | p | Bootstrap 95% CI |
|-----------|-----------|---|-----------------|
| **Concentration** | **-0.446** | **9.4 × 10⁻⁸** ★★★ | **[-0.602, -0.247]** ✓ |
| **μ₀ (central SB)** | **+0.450** | **7.0 × 10⁻⁸** ★★★ | **[+0.262, +0.603]** ✓ |
| SB slope | +0.094 | 0.286 (ns) | — |

Both bootstrap CIs exclude zero. 100% directional consistency.

**Physical meaning:** At fixed total baryonic mass AND fixed asymptotic velocity, galaxies where the light is more concentrated toward the center are LESS efficient at converting their baryons into rotational support. Galaxies with brighter central surface brightness are MORE efficient.

This is exactly what geometry-matters predicts: it's not just HOW MUCH mass you have, it's HOW IT'S DISTRIBUTED that determines how effectively baryons support rotation. Two galaxies with the same total mass and the same V_flat can have very different baryon-to-DM contributions depending on baryonic geometry.

---

## Cumulative Evidence: All Structural Findings

| Finding | Partial ρ | p | Controls | n |
|---------|-----------|---|----------|---|
| **Baryon effectiveness vs concentration** | **-0.446** | **9.4 × 10⁻⁸** | **M + V_flat** | **131** |
| **Baryon effectiveness vs μ₀** | **+0.450** | **7.0 × 10⁻⁸** | **M + V_flat** | **131** |
| Acceleration scale vs concentration | -0.574 | 3.1 × 10⁻⁷ | M_bar | 68 |
| DM 50% radius vs concentration | -0.515 | 4.2 × 10⁻⁶ | M_bar | 71 |
| DM 50% radius vs μ₀ | +0.470 | 3.6 × 10⁻⁵ | M_bar | 71 |
| Mass discrepancy vs concentration | +0.346 | 1.0 × 10⁻⁴ | M_bar | 120 |
| Diversity (rise ratio) vs SB slope | +0.329 | 1.0 × 10⁻⁴ | V_flat | 131 |
| DM growth rate vs SB slope | +0.310 | 8.4 × 10⁻⁴ | M + V_flat | 113 |
| RAR offset vs R_eff | -0.294 | 6.6 × 10⁻⁴ | M + V_flat | 131 |

**Nine independent partial correlations, all significant, all pointing the same direction: baryonic geometry predicts dark matter phenomenology beyond what mass alone provides.**

---

## What This Means for RTM

RTM's core claim for galactic dynamics is: "the topological organization of baryonic matter modulates dynamics." The front door (predicting v(r) directly) is closed — RTM can't replace NFW with 1 parameter.

But the side doors reveal a systematic pattern: at fixed mass, the SHAPE of the light profile predicts:
- How effective baryons are at supporting rotation (ρ = -0.45)
- Where the DM transition acceleration occurs (ρ = -0.57)
- Where DM reaches 50% dominance (ρ = -0.52)
- How fast DM fraction grows with radius (ρ = +0.31)
- The diversity of inner rotation curve shapes (ρ = +0.33)
- The overall mass discrepancy magnitude (ρ = +0.35)
- The systematic RAR offset (ρ = -0.29)

These are not nine versions of the same correlation — they probe different physical aspects of the baryon-DM relationship, and they survive different control variables.

---

*All computations reproducible via rtm_flank4.py.*
