# RTM Flank 6: Gas Fraction — The Hidden Modulator

**Date:** April 28, 2026
**Data:** SPARC table2.dat (123 galaxies)

---

## The Surprise

We've been treating all baryons as equal. They're not.

SPARC provides V_gas and V_disk separately. Gas and stars have fundamentally different geometries: gas extends further, fills the potential well more uniformly, and is pressure-supported. Stars are concentrated in a thin disk and collisionless.

If baryonic GEOMETRY drives the baryon-DM coupling (RTM), the gas-to-stellar ratio should modulate everything we've found.

It does. Dramatically.

---

## The Headline Finding

| Galaxy Type | Mean Conspiracy r (V_bar shape vs V_DM shape) | n |
|-------------|-----------------------------------------------|---|
| **Gas-rich** (f_gas > 0.18) | **+0.705** | 61 |
| **Gas-poor** (f_gas ≤ 0.18) | **-0.151** | 62 |
| Mann-Whitney p | **< 10⁻⁴** | |

**Gas-rich galaxies show a strong baryon-halo conspiracy (r = 0.70). Gas-poor galaxies show none (r ≈ -0.15).**

The conspiracy we found in Flank 5 (mean r = 0.274 over all galaxies) was an AVERAGE of two completely different populations. Gas-rich galaxies have V_DM shapes that tightly mirror V_bar shapes. Gas-poor galaxies have V_DM shapes that are uncorrelated or anti-correlated with V_bar.

This is not a mass effect. Partial correlation controlling for mass AND V_flat:

| f_gas → | Partial ρ | p |
|----------|-----------|---|
| Conspiracy strength | **+0.279** | **0.002** ★★ |
| Baryon effectiveness | **-0.371** | **2.4 × 10⁻⁵** ★★★ |
| Break sharpness | **-0.235** | **0.009** ★★ |

All three survive mass control. Gas fraction is a genuine, independent structural predictor.

---

## Gas Fraction Modulates the Structural Link

The concentration → baryon effectiveness correlation we found in Flank 4 is NOT universal. It only works in gas-rich galaxies:

| Subsample | ρ(conc → baryon_eff, controlling M+V) | p |
|-----------|---------------------------------------|---|
| Gas-rich (n=61) | **-0.416** | **0.0008 ★★★** |
| Gas-poor (n=52) | -0.216 | 0.12 (ns) |

**Structural geometry only matters when gas is present.** In gas-poor galaxies, structure doesn't predict baryon effectiveness — mass alone determines dynamics. In gas-rich galaxies, the geometric organization of baryons is a significant modulator.

This makes physical sense: gas fills the potential well more uniformly than stars, so its distribution more directly reflects the overall gravitational structure. When gas dominates, baryonic geometry matters. When stars dominate, the thin-disk approximation holds and structure washes out.

---

## Break Sharpness: MOND-like vs NFW-like

How abruptly does the baryon→DM transition occur? At fixed mass and V_flat:

| Predictor → Break sharpness | Partial ρ | p |
|-----------------------------|-----------|---|
| **Concentration** | **-0.373** | **4.7 × 10⁻⁵** ★★★ |
| **μ₀** | **+0.326** | **0.0004** ★★★ |
| **SB roughness** | **+0.328** | **0.0004** ★★★ |
| SB slope | -0.183 | 0.052 (marginal) |

More concentrated galaxies → **gentler** transitions (more NFW-like).
Fainter-center or rougher galaxies → **sharper** transitions (more MOND-like).

Structure adds ΔR² = 11.6% to break sharpness prediction (F-test p = 0.003).

---

## The Combined Model: Predicting the Conspiracy

Can we predict how tightly a galaxy's "dark matter" shape tracks its baryonic shape?

| Model | R² | ΔR² | F-test p |
|-------|-----|-----|----------|
| Mass + V_flat only | 0.399 | — | — |
| Mass + V + conc + μ₀ + **f_gas** + roughness | **0.502** | **+0.103** | **4.6 × 10⁻⁴** |

Bootstrap ΔR²: mean = 0.120, 95% CI = [0.049, 0.208]. **Excludes zero.**

The dominant coefficient is **f_gas = +1.055** — gas fraction alone drives most of the structural contribution. The combined model explains **50%** of conspiracy variance, up from 40% with mass alone.

---

## Physical Interpretation

Why does gas make the conspiracy stronger?

1. **Gas fills the potential.** Unlike collisionless stars confined to a thin disk, gas is pressure-supported and responds hydrostatically to the total gravitational potential. Its distribution more faithfully traces the overall mass structure.

2. **Gas extends further.** In SPARC galaxies, gas systematically extends beyond the stellar disk. This means gas "sees" the DM-dominated regime directly, while stars only probe the baryon-dominated inner region.

3. **Gas is responsive.** Gas responds to feedback, cooling, and pressure — processes that couple it to the gravitational environment. Stars are frozen artifacts of past formation events.

The implication for RTM: structural coherence as a modulator of dynamics works best when the baryonic medium is fluid and responsive (gas), not when it's frozen and collisionless (stars). This aligns with RTM's framework — "coherence" requires a medium that can propagate structural information across scales.

---

## Updated Cumulative Findings

Adding Flank 6 to the total:

| # | Finding | ρ | p | New? |
|---|---------|---|---|------|
| 1 | Baryon eff vs concentration | -0.446 | 10⁻⁷ | |
| 2 | Baryon eff vs μ₀ | +0.450 | 10⁻⁷ | |
| 3 | Acceleration scale vs concentration | -0.574 | 10⁻⁷ | |
| 4 | DM 50% radius vs concentration | -0.515 | 10⁻⁶ | |
| 5 | DM 50% radius vs μ₀ | +0.470 | 10⁻⁵ | |
| 6 | Conspiracy vs μ₀ | -0.326 | 10⁻⁴ | |
| 7 | Mass discrepancy vs concentration | +0.346 | 10⁻⁴ | |
| 8 | Diversity vs SB slope | +0.329 | 10⁻⁴ | |
| 9 | Mirror quality vs μ₀ | -0.313 | 10⁻⁴ | |
| 10 | DM growth rate vs SB slope | +0.310 | 10⁻³ | |
| 11 | RAR offset vs R_eff | -0.294 | 10⁻³ | |
| 12 | **f_gas → baryon eff** | **-0.371** | **10⁻⁵** | **✓** |
| 13 | **Break sharpness vs concentration** | **-0.373** | **10⁻⁵** | **✓** |
| 14 | **Break sharpness vs μ₀** | **+0.326** | **10⁻⁴** | **✓** |
| 15 | **Break sharpness vs roughness** | **+0.328** | **10⁻⁴** | **✓** |
| 16 | **f_gas → conspiracy** | **+0.279** | **0.002** | **✓** |
| 17 | **f_gas → break sharpness** | **-0.235** | **0.009** | **✓** |
| 18 | Conspiracy exists (mean r=0.274) | — | 10⁻⁴ | |
| 19 | **Gas-rich conspiracy = 0.70, Gas-poor = -0.15** | — | **< 10⁻⁴** | **✓** |

**19 significant findings. 7 new from this flank. All controlling for mass.**

---

*All computations reproducible via rtm_flank6.py.*
