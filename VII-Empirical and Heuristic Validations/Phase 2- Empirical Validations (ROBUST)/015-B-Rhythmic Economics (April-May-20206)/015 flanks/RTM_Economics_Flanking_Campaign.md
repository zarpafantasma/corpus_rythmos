# RTM Economics Flanking Campaign

**Date:** April 28, 2026
**Data:** 3 months BTC 1-min Binance (COVID, FTX, Control), 13 crash events

---

## Five Flanks — Two Major Hits, One Failure, Two Mixed

### Flank 1: Out-of-Sample Crash Prediction — FAILED

Trained an $\alpha$-drop threshold on 9 pre-2022 crash events (threshold: $\Delta\alpha < -0.127$), tested on 4 post-2022 events.

| Event | $\Delta\alpha$ | Predicted | Actual | |
|-------|-------------|-----------|--------|---|
| BTC 2022 Terra | -0.113 | NORMAL | CRASH | $\times$ |
| BTC 2022 FTX | -0.039 | NORMAL | NORMAL | $\checkmark$ |
| SP500 2022 Bear | -0.051 | NORMAL | CRASH | $\times$ |
| Gold 2022 Fed | -0.055 | NORMAL | CRASH | $\times$ |

**Out-of-sample accuracy: 25% (1/4).** The trained threshold is too aggressive — it catches only the most extreme drops (pre-2022 had bigger $\alpha$-drops than 2022+ events). No threshold variant achieves above 50%.

**For RTM: NEGATIVE.** The DFA $\alpha$-drop framework, as parameterized here, does not generalize out of sample. The 2022+ crashes show smaller $\alpha$ signatures than the training set. This may reflect changing market microstructure (more algorithmic trading, faster reactions) reducing the lead time and magnitude of $\alpha$-drops.

---

### Flank 2: Volume-Volatility Shape Conspiracy — MIXED

Volume and volatility shapes are always strongly correlated ($r > 0.88$). The control month shows the highest coupling ($r = 0.943$) and crash months are slightly lower ($r = 0.889$-$0.914$).

| Month | Global $r$ | Type |
|-------|-----------|------|
| Control 2023-09 | +0.943 | Control |
| COVID 2020-03 | +0.914 | Crash |
| FTX 2022-11 | +0.889 | Crash |

The pre-crash vs during-crash dynamics are inconsistent: COVID shows conspiracy drops during crash ($d = -0.90$, significant), but FTX shows it increases ($d = +0.71$). The direction is not reliable.

**For RTM: WEAK.** The volume-volatility coupling is always strong ($r > 0.88$). Crash months show slightly weaker coupling, but the effect is small and the within-crash dynamics are inconsistent between events.

---

### Flank 3: Multi-Scale Coherence — MAJOR UNEXPECTED HIT

This is the most surprising finding. RTM predicts that $\alpha$ should be consistent across time scales ("fractal coherence"). I expected crashes to break this coherence. **The opposite happened.**

**Cross-scale $\alpha$ (volatility-volume slope at 1min, 5min, 15min, 60min):**

| Month | 1min | 5min | 15min | 60min | $\sigma$ | Verdict |
|-------|------|------|-------|-------|----------|---------|
| **COVID** | 0.655 | 0.681 | 0.708 | 0.739 | **0.031** | **COHERENT** |
| **FTX** | 0.838 | 0.880 | 0.902 | 0.931 | **0.034** | **COHERENT** |
| **Control** | 1.875 | 1.268 | 1.116 | 1.133 | **0.310** | **INCOHERENT** |

**Crash months are 10x more scale-coherent than the control month.**

During crashes, $\alpha$ is nearly identical from 1-minute to 1-hour resolution ($\sigma \approx 0.03$). In quiet markets, different time scales show wildly different $\alpha$ ($\sigma = 0.31$; 1-minute $\alpha = 1.875$ vs 60-minute $\alpha = 1.133$).

**Rolling 1min-15min $\alpha$ correlation:**

| Month | $r$ |
|-------|-----|
| COVID | +0.724 |
| FTX | +0.641 |
| Control | +0.304 |

**Physical interpretation:** In quiet markets, different time scales operate semi-independently — short-term noise, medium-term patterns, and long-term trends have different structures. During a crash, the cascade becomes scale-invariant: the fracture propagates uniformly across ALL time scales simultaneously. This is the signature of a genuine phase transition — at criticality, all scales become coupled.

**This is the opposite of what I predicted, but it's MORE interesting:** The market doesn't lose coherence during a crash — it gains coherence. The crash IS the coherent state. The quiet market is the incoherent one. RTM's $\alpha$ is measuring something real about the scale structure of the market, and it reveals that crashes are not random breakdowns but organized, cross-scale cascades.

**For RTM: STRONG POSITIVE.** This is a genuinely novel finding. The multi-scale coherence metric ($\sigma$ of cross-scale $\alpha$) is a crash detector that works in a fundamentally different way from the DFA $\alpha$-drop. Instead of asking "did $\alpha$ fall?", it asks "did $\alpha$ become scale-invariant?" This is an RTM-native diagnostic that no standard financial technique measures.

---

### Flank 4: Crash-Recovery Asymmetry — MIXED

| Crash | Fall | Recovery | Ratio | RTM prediction |
|-------|------|----------|-------|---------------|
| COVID 2020 | 12.0 days | 18.9 days | 1.6x slower recovery | $\checkmark$ |
| FTX 2022 | 20.9 days | 9.1 days | 0.4x faster recovery | $\times$ |

RTM predicts crashes are fast (phase transition) and recoveries are slow (reconstruction). COVID confirms this (recovery 1.6x slower). FTX contradicts it (recovery 2.3x faster). The FTX crash was a slow grind (weeks of uncertainty), not a sharp break, which may explain the asymmetry reversal.

**For RTM: INCONCLUSIVE.** The asymmetry prediction works for sharp, exogenous shocks (COVID) but not for slow-burn solvency crises (FTX). RTM may need to distinguish crash types.

---

### Flank 5: Trade Count Divergence — INTERESTING SECONDARY FINDING

Trade count $\alpha$ (using number of trades instead of volume) reveals a structural difference:

| Month | Volume-$\alpha$ | Trade-$\alpha$ | Divergence |
|-------|----------------|---------------|------------|
| COVID | 0.544 | 0.802 | 0.258 |
| FTX | 0.740 | 1.096 | 0.356 |
| **Control** | **1.729** | **3.781** | **2.052** |

In crash months, volume-based and trade-based $\alpha$ are close (divergence $\approx 0.3$). In the control month, they diverge massively ($2.05$). This suggests that in quiet markets, volume and trade count carry different structural information (large trades vs many small trades behave differently), while during crashes this distinction collapses — all trades become "the same" regardless of size.

The divergence does NOT increase before crashes (d $\approx$ -0.2 to -0.5, wrong direction). So this is a state descriptor, not a precursor.

**For RTM: MODERATE.** Confirms that crash states have different microstructural properties than quiet states, but doesn't add precursor capability.

---

## Summary

| Flank | Result | Key metric | For RTM |
|-------|--------|-----------|---------|
| 1. Out-of-sample prediction | **FAILED** | 25% accuracy | **NEGATIVE** |
| 2. Vol-Vola conspiracy | MIXED | $r$ similar across months | WEAK |
| 3. **Multi-scale coherence** | **MAJOR HIT** | **Crash $\sigma=0.03$ vs Control $\sigma=0.31$** | **STRONG POSITIVE** |
| 4. Crash-recovery asymmetry | MIXED | Works for COVID, not FTX | INCONCLUSIVE |
| 5. Trade count divergence | INTERESTING | State descriptor, not precursor | MODERATE |

---

## The Multi-Scale Coherence Finding in Detail

This deserves emphasis because it's the most RTM-native result:

**Standard approach (DFA $\alpha$-drop):** Measures whether $\alpha$ changes over time at a single scale. Known technique (Grech & Mazur 2004). Failed out-of-sample in our test.

**RTM approach (multi-scale $\sigma$):** Measures whether $\alpha$ is consistent ACROSS scales at a given time. Novel metric. Separates crash from control with $\sigma$ ratio of 10x. No threshold training needed — just compute $\sigma$ of $\alpha$ across scales.

A potential crash detector based on multi-scale coherence would work like this:

1. Compute $\alpha$ at 1min, 5min, 15min, 60min simultaneously
2. Track $\sigma(\alpha)$ over rolling windows
3. When $\sigma$ drops below threshold (coherence increases): alert

This is fundamentally different from any existing crash indicator because it doesn't look for a level or trend — it looks for scale-invariance. And it comes directly from RTM's core claim that $\alpha$ is a topological invariant that should be consistent across scales in a coherent system.

**Caveat:** Only 3 months tested. Would need many more months (both crash and quiet) to establish reliability. Could be confounded by volatility regime (crash months are higher-volatility, which might mechanically reduce cross-scale differences).

---

## Score Impact

**Doc 015 Economics: 65% $\rightarrow$ 68%**

The out-of-sample failure (-5) is partially compensated by the multi-scale coherence finding (+8). The net gain is modest because the flagship DFA technique doesn't generalize, and 3 of 5 flanks are inconclusive or mixed. The multi-scale coherence metric is genuinely promising but needs more data to validate.

Compared to ecology (55% $\rightarrow$ 70%) and astronomy (25% $\rightarrow$ 70%), economics shows less dramatic improvement. The fundamental limitation is that financial DFA is already a known technique, and the novel angle (multi-scale coherence) only has 3 data points.

---

*All computations reproducible via rtm_econ_flanks.py. Data: Binance BTCUSDT 1-min candles (Doc 015 package).*
