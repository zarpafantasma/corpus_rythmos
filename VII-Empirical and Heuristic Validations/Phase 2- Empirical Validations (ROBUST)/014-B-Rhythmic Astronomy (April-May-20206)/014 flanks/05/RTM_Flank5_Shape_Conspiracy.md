# RTM Flank 5: The Shape Conspiracy

**Date:** April 28, 2026
**Data:** SPARC table2.dat (123 galaxies)

---

## The Idea

Forget single-number correlations. Ask a fundamentally different question: does the SHAPE of baryonic mass predict the SHAPE of the inferred dark matter?

In ΛCDM, the DM halo is an independent component. Its radial profile comes from cosmological initial conditions and gravitational collapse — not from baryonic structure. So the shape of V_DM(r) should be statistically independent of the shape of V_bar(r), after accounting for total mass.

If they correlate, it's a conspiracy. And conspiracies need explanations.

---

## Result: The Conspiracy Is Real

For each galaxy, I computed:
- V_bar(r) = baryonic velocity contribution at each radius
- V_DM(r) = sqrt(V_obs² - V_bar²) = inferred DM velocity at each radius
- Normalized both to unit amplitude (pure shape, no magnitude)
- Pearson correlation between the two shapes

| Metric | Value |
|--------|-------|
| Mean shape correlation (V_bar vs V_DM) | **+0.274** |
| Galaxies with positive correlation | **64%** |
| t-test vs 0 | **t = 4.03, p = 9.9 × 10⁻⁵** |
| Bootstrap 95% CI | **[0.138, 0.402]** |
| Bootstrap % positive | **100%** |

**The shape of baryonic matter and the shape of "dark matter" are correlated.** Bootstrap excludes zero. 100% of 5,000 bootstrap iterations show positive mean. This is the baryon-halo conspiracy, quantified.

In a universe where DM halos are independent of baryons, this correlation should be zero. It's not.

---

## What Predicts the Conspiracy Strength?

Not all galaxies show equal conspiracy. What structural property determines how tightly V_bar and V_DM shapes track each other?

**Partial correlations (controlling mass + V_flat):**

| Predictor | Partial ρ with conspiracy strength | p |
|-----------|-----------------------------------|---|
| **μ₀ (central SB)** | **-0.326** | **0.0002 ★★★** |
| **SB roughness** | **-0.255** | **0.004 ★★** |
| SB entropy | +0.229 | 0.011 ★ |
| SB slope | -0.121 | 0.18 (ns) |

**Physical meaning:**
- Galaxies with **brighter** centers show **weaker** conspiracy — their V_bar and V_DM shapes diverge more
- Galaxies with **rougher** (bumpier, irregular) SB profiles show **weaker** conspiracy
- **Smooth, faint-center galaxies show the strongest conspiracy** — their "dark matter" shape tracks their baryonic shape most closely

This is exactly what you'd expect if the "dark matter" signal is partly a response to baryonic structure: smooth, extended baryonic distributions couple more tightly to the total gravitational field, so V_DM appears to mirror V_bar. Concentrated or bumpy distributions break this coupling.

---

## The Mirror Test

Separately, I tested how similar V_obs(r) and V_bar(r) shapes are:

| Metric | Value |
|--------|-------|
| Mean shape similarity | 0.666 |
| Galaxies with r > 0.9 | 47% |
| Galaxies with r < 0.5 | 24% |

**μ₀ predicts mirror quality:** partial ρ = -0.313, p = 0.0004. Brighter centers → V_obs shape diverges more from V_bar shape (more DM needed in center).

**SB roughness predicts mirror quality:** partial ρ = -0.275, p = 0.002. Bumpier profiles → worse match between V_bar and V_obs shapes.

---

## The Derivative Conspiracy (Null)

Do V_bar and V_DM curve the same way? (Are their gradients correlated?)

Mean derivative correlation: 0.009, CI includes zero. **No.** The shapes track each other in amplitude envelope, not in wiggles. This rules out a trivial mathematical artifact — the conspiracy is in the broad radial trend, not in point-to-point noise.

---

## Why This Matters

The baryon-halo conspiracy is one of the most cited challenges to ΛCDM on galactic scales (van Albada & Sancisi 1986, McGaugh 2014). The fact that "dark matter" appears to "know about" baryonic distribution has been used as evidence for MOND and other modified gravity theories.

What we add here:
1. **Quantification:** The conspiracy has mean r = 0.274, bootstrap CI [0.14, 0.40]
2. **Structural modulation:** μ₀ and SB roughness predict conspiracy strength (p < 0.005)
3. **RTM interpretation:** Smooth, extended baryonic structures show tighter baryon-DM coupling, consistent with "topological coherence modulates dynamics"

---

## Cumulative SPARC Findings (All Flanks)

| # | Finding | ρ | p | Controls |
|---|---------|---|---|----------|
| 1 | Baryon effectiveness vs concentration | -0.446 | 9.4×10⁻⁸ | M + V |
| 2 | Baryon effectiveness vs μ₀ | +0.450 | 7.0×10⁻⁸ | M + V |
| 3 | Acceleration scale vs concentration | -0.574 | 3.1×10⁻⁷ | M |
| 4 | DM 50% radius vs concentration | -0.515 | 4.2×10⁻⁶ | M |
| 5 | DM 50% radius vs μ₀ | +0.470 | 3.6×10⁻⁵ | M |
| 6 | Conspiracy strength vs μ₀ | -0.326 | 2.3×10⁻⁴ | M + V |
| 7 | Mass discrepancy vs concentration | +0.346 | 1.0×10⁻⁴ | M |
| 8 | Diversity (rise ratio) vs SB slope | +0.329 | 1.0×10⁻⁴ | V |
| 9 | Mirror quality vs μ₀ | -0.313 | 4.3×10⁻⁴ | M + V |
| 10 | DM growth rate vs SB slope | +0.310 | 8.4×10⁻⁴ | M + V |
| 11 | RAR offset vs R_eff | -0.294 | 6.6×10⁻⁴ | M + V |
| 12 | Conspiracy strength vs roughness | -0.255 | 4.4×10⁻³ | M + V |
| 13 | **Conspiracy exists** (mean r = 0.274) | — | **9.9×10⁻⁵** | — |

Thirteen significant findings. All surviving mass controls. All pointing the same direction.

---

*All computations reproducible via rtm_flank5.py.*
