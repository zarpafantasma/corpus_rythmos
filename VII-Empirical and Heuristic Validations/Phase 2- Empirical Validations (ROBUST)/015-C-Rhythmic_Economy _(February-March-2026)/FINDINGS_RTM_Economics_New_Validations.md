# RTM Economics — New Empirical Validations on Real Market Data

**Forensic audit of prior validation data + independent replication suite**

Date: July 5, 2026
Deliverables: `forensic_audit_crash_alpha.py`, `rtm_econ_new_validations.py`, raw data CSVs, results CSVs, this document.

---

## Executive Summary

This work set out to reinforce the empirical basis of Doc 015 (Rhythmic Economics) with new, real-data validations. It produced two categories of findings, reported here with equal weight:

**1. A critical forensic finding.** The dataset `crash_alpha_analysis.csv` — the basis of Doc 015 Chapter 12's flagship result ("13 historical crashes, Cohen's d = −1.45, 9.75-day mean lead time, severity correlation r = 0.97") — is **synthetically generated**, not measured from market data. All 13 rows of its alpha columns and lead times reproduce, to machine precision, the output of `np.random.seed(42)` with a simple generator (5 uniform draws per event). The d = −1.45 separation, the ~10-day lead time, and the r = 0.97 severity correlation are **properties of the simulation design, not of markets**. Chapter 12's in-sample claims must be reclassified from "empirical validation" to "simulation/illustration."

**2. New validations on genuinely real data.** Using verifiable public data (5,104 days of S&P 500 OHLCV 2000–2020 covering three major crises; daily VIX 1990–2026; Shiller monthly S&P 1871–2026), five independent tests were run. Summary of what survives and what does not:

| # | Claim tested | Result on real data | Status |
|---|---|---|---|
| A | DFA α is lower in crisis periods (claimed d = −1.45) | d = −0.22 (drawdown labels, p = 0.22); d = +0.03 (VIX labels, p = 0.93) | **Not supported** — real effect is ~6× smaller than the synthetic claim and not significant |
| B | H2: α drops before crisis onsets (anticipation) | α **rose** before both testable onsets (GFC +0.05, COVID +0.07) | **Contradicted** |
| C | Multi-scale coherence: cross-scale σ(α) collapses in crises (claimed 10× on BTC) | Direction replicates: crisis σ median 0.549 vs calm 0.720 (d = −0.37, p = 0.039, AUC = 0.60); ratio 1.31× | **Partially supported** — real, weak, and far below the 10× BTC claim |
| D | Inverse cubic law of return tails (α ≈ 3) | Hill estimator on raw daily returns: α = 2.9–3.3 at the far tail (k = 2.5%) | **Supported** (convergent with Gabaix et al. 2003), now independently computed rather than cited |
| E | Century-scale behavior (new descriptive test) | Monthly DFA α ≈ 0.68 ± 0.07 (persistent); no pre-crash α depression before 1929/1973/2000/2008 | Descriptive; **no anticipation signal** at monthly scale |

**Bottom line:** the honest empirical core of RTM Economics after this audit is (i) the multi-scale coherence effect — real but weak on equities at daily resolution, and the only RTM-native signal that survives out-of-sample transfer — and (ii) consistency with known scaling laws (inverse cubic tails). The DFA α-drop crash-prediction narrative does not survive contact with real data, in either its level form (Test A) or its anticipation form (Test B). This is consistent with, and strengthens, the Red Team's earlier finding of 25% out-of-sample accuracy.

---

## Part 1 — Forensic Finding: Synthetic Provenance of `crash_alpha_analysis.csv`

### 1.1 What was found

The alpha columns of `crash_alpha_analysis.csv` exactly reproduce a pseudo-random generator seeded at 42. Replaying `np.random.seed(42)` and consuming 5 uniform draws per event (r0…r4):

| Column | Generator formula | Exact matches |
|---|---|---|
| `Baseline_Alpha` | 0.5 + 0.1·r0 (i.e., U(0.50, 0.60)) | **13/13** |
| `Pre_Alpha` | Baseline + (−0.03 + 0.06·r1) | **13/13** |
| `Lead_Time_Hours` | int(72 + 432·r4) (i.e., U(3, 21) days) | **13/13** |
| `Immediate_Alpha` | f(Baseline, \|Drop%\|, r2) — severity is an *input* | R² = 0.86 vs linear reconstruction |

The match is to machine precision (< 10⁻¹²) on 39 of 39 tested values across three independent columns. The probability of this occurring with genuinely measured data is zero. Additional signatures: two events carry `Alpha_Drop` = −0.03 to 17 decimal places (a clip/floor artifact), and the celebrated severity correlation (r = 0.97 between α-drop and crash depth) is **built into the generator** — `Immediate_Alpha` is constructed as a function of `Drop_Pct`.

Reproduce with: `python forensic_audit_crash_alpha.py crash_alpha_analysis.csv` → `forensic_reconstruction.csv`.

### 1.2 What this means and does not mean

- The event list itself (dates, crash magnitudes) is real historical record. Only the α measurements and lead times — the scientific content — are synthetic.
- The downstream "ROBUST" pipeline (`analyze_financial_robust.py`) is not the origin of the problem: it honestly reads the CSV and adds Monte Carlo noise. The synthetic table entered the pipeline upstream, at data-creation time, and was thereafter treated as empirical.
- The April 2026 Red Team **reproduced the statistics of the CSV** (d, lead times) but did not test data provenance. Its independent analysis of raw BTC 1-minute data — which found the intra-event precursor *inconsistent* — now makes sense: real data never contained the clean pattern.
- This finding does **not** automatically invalidate the BTC 1-minute analyses (Chapters 11 / 12.5 flanks), which were computed from raw Binance candles. It invalidates specifically the 13-event cross-market table and every number derived from it: d = −1.45, the 9.75-day lead, the r = 0.97 severity correlation, and README claims that these "definitively validate the RTM Early Warning Indicator."

### 1.3 Required corrections to Doc 015

1. Chapter 12 and the abstract must reclassify the 13-event analysis as a **simulation/illustration**, or remove it.
2. The README of the "Reproducible" package must be corrected: the pipeline is reproducible, but its input is synthetic.
3. The Red Team addendum should note that provenance checking is now part of the audit standard (statistical reproduction alone is insufficient).

---

## Part 2 — New Validations on Real Data

### 2.0 Data (all public, verifiable, downloaded at runtime)

| Dataset | Coverage | Source |
|---|---|---|
| S&P 500 daily OHLCV | 2000-01-03 → 2020-04-17 (5,104 days) | vega-datasets `sp500-2000.csv` (Yahoo Finance origin) |
| VIX daily | 1990-01-02 → 2026-07-03 (9,221 days) | `datasets/finance-vix` (CBOE origin) |
| S&P 500 monthly | 1871-01 → 2026-06 (1,865 months) | `datasets/s-and-p-500` (Shiller origin) |

The daily window contains three canonical crises, fixed ex ante from public record: Dot-Com bear (2000-09-01→2002-10-09), GFC bear (2007-10-09→2009-03-09), COVID crash (2020-02-19→2020-04-17). The DFA implementation was calibrated on known processes before use (white noise α = 0.507 vs 0.50 expected; Brownian motion α = 1.500 vs 1.50 expected).

### 2.1 Test A — DFA α crisis discrimination

Rolling 250-day DFA α of daily log returns (21-day step, 232 windows: 40 crisis, 192 calm):

| Labeling | Crisis α | Calm α | Cohen's d | p (MW) |
|---|---|---|---|---|
| Drawdown windows | 0.464 ± 0.065 | 0.479 ± 0.065 | **−0.218** | 0.223 |
| VIX > 30 (>25% of window) | 0.478 ± 0.051 | 0.476 ± 0.067 | **+0.030** | 0.925 |
| Non-overlapping (robustness) | n = 4 | n = 16 | −0.590 | small-N |

Full-sample daily α = 0.497 — the S&P is, at daily resolution, essentially a random walk, consistent with decades of market-efficiency literature. **The synthetic claim (baseline 0.55 → crash 0.46, d = −1.45) does not replicate**: on real data the direction is weakly consistent under one labeling, reverses under the other, and is never significant. Spearman correlation between α and mean VIX across windows: ρ = 0.015 (p = 0.82) — α carries essentially no volatility-regime information at this resolution.

### 2.2 Test B — H2 anticipation (event study)

Does α drop in the 250 days *before* a crisis onset, relative to a baseline window one year earlier?

| Onset | Baseline α (−1y) | Pre-onset α | Δ | H2 prediction |
|---|---|---|---|---|
| Dot-Com (2000-09) | — | — | — | not testable (data starts 2000-01) |
| GFC (2007-10) | 0.376 | 0.429 | **+0.052** | drop — **contradicted** |
| COVID (2020-02) | 0.453 | 0.528 | **+0.074** | drop — **contradicted** |

In both testable cases, α **rose** into the crisis onset. Combined with Test E (no pre-crash depression before 1929/1973/2000/2008 at monthly scale) and the Red Team's 25% out-of-sample accuracy, hypothesis H2 (α decline anticipates crises) is now contradicted by three independent lines of real-data evidence. H2 should be retired or fundamentally reformulated.

### 2.3 Test C — Multi-scale coherence (the novel metric, out-of-sample)

This was the flanking campaign's most promising finding (BTC 1-min: crash-month cross-scale σ ≈ 0.03 vs control ≈ 0.31 — a 10× separation on 3 months of data). The stated future work was to test it on held-out data and other asset classes. This test does exactly that: annual windows of daily S&P data, volatility–volume regression α at 1/2/5/10-day aggregations, σ across scales, 232 windows (41 crisis / 191 calm).

| Metric | Crisis | Calm | Statistic |
|---|---|---|---|
| σ_cross-scale (median) | 0.549 | 0.720 | ratio 1.31× |
| σ_cross-scale (mean ± SD) | 0.645 ± 0.379 | 0.819 ± 0.543 | d = −0.37 |
| Mann–Whitney | | | p = 0.039 |
| AUC (crisis = lower σ) | | | 0.603 |
| Non-overlapping robustness | n = 4 crisis | n = 16 calm | d = −0.87 |

**Verdict — the honest middle ground.** The effect replicates in *direction* on an independent asset class, timescale, and 20-year period: crisis windows are more scale-coherent than calm windows, exactly as the RTM phase-transition interpretation predicts. But the *magnitude* is an order of magnitude weaker than the BTC claim: 1.31× median separation instead of 10×, AUC 0.60 instead of near-perfect. Possible reconciliations (untested): the effect may genuinely be stronger at intraday resolution; the BTC 10× may be inflated by having only 3 months; or crypto microstructure may be more scale-coupled than equities. As it stands, multi-scale coherence is a **real but weak** regime descriptor on equities — the only RTM-native economic signal that survives out-of-sample transfer, and the correct focus for future work, with realistic expectations.

### 2.4 Test D — Hill tail exponents (independent inverse-cubic test)

The Red Team correctly noted that Doc 015's inverse-cubic table was a literature compilation, not independent analysis. Here the tail exponent is computed directly from the 5,104 real daily returns via the Hill estimator:

| Tail fraction k | Positive tail α | Negative tail α |
|---|---|---|
| 2.5% | 3.32 | 2.89 |
| 5% | 2.56 | 2.71 |
| 10% | 2.29 | 2.55 |

At the far tail (k = 2.5%) both tails bracket α = 3, consistent with the inverse cubic law (Gopikrishnan et al. 1999; Gabaix et al. 2003); at larger k the estimator drifts down as bulk observations contaminate the tail, which is standard Hill behavior. **This upgrades the inverse-cubic claim from "cited" to "independently reproduced"** — while remaining, as before, a convergent result with established econophysics rather than an RTM discovery.

### 2.5 Test E — Century-scale DFA (Shiller monthly, 1871–2026)

A test no prior RTM document ran: rolling 20-year DFA α on 155 years of monthly returns. Full-sample monthly α = 0.565; rolling mean 0.682 ± 0.065 (range 0.53–0.84). Monthly returns show genuine persistence (α > 0.5) — in contrast to daily returns (α ≈ 0.5) — consistent with the momentum literature and interesting as an RTM cross-timescale observation: **the persistence exponent of the same market is itself scale-dependent**. However, α in windows ending ≤2 years before the four largest crashes (1929: 0.656; 1973: 0.753; 2000: 0.571; 2008: 0.606) scatters around the overall mean with no systematic depression — again no anticipation signal.

---

## Part 3 — Reconciliation Table (Doc 015 claims after this audit)

| Doc 015 claim | Prior status | Status after real-data validation |
|---|---|---|
| "Baseline α = 0.55, crash α = 0.46, d = −1.45" | Flagship empirical (Ch. 12) | **Synthetic** (seed-42 generator); real-data d ≈ −0.2, ns |
| "α-drop precedes trough by 9.75 days" | Flagship empirical (Ch. 12) | **Synthetic** (lead = U(3,21) days by construction) |
| "Severity correlation r = 0.97" | Flagship empirical | **Synthetic** (built into generator) |
| H2: α decline anticipates crises 6–18 months | Hypothesis, "backtested" in sims | **Contradicted** on real data (α rose before GFC & COVID; no monthly-scale signal 1871–2026; RT out-of-sample 25%) |
| Multi-scale coherence σ: 10× crash/control (BTC) | Novel, 3 data points | **Direction replicates** on equities (p = 0.039) but at 1.3×, AUC 0.60 — real, weak |
| Inverse cubic law α ≈ 3 | Literature citation | **Independently reproduced** from raw returns (far-tail Hill 2.9–3.3) |
| Monthly-vs-daily persistence gap (new) | — | New descriptive finding: α_monthly ≈ 0.68 vs α_daily ≈ 0.50 |

## Part 4 — Limitations of These New Tests

Honesty applies in both directions; these tests have their own boundaries. (1) The daily S&P file ends 2020-04-17, so the 2022 bear market is untested at daily resolution. (2) Only 3 crises fall in the daily window and only 2 onsets are testable for H2 — small event-N is intrinsic to crash research. (3) Rolling windows overlap; non-overlapping robustness checks are reported but have small N. (4) The multi-scale α here uses daily bars (1/2/5/10-day); it is not a resolution-matched replication of the BTC 1/5/15/60-minute analysis, so the 10× vs 1.3× comparison spans both asset class and timescale. (5) The volatility–volume α definition follows the flanking campaign's method; other α definitions (pure DFA per scale) may behave differently — metric sensitivity was already flagged by the Red Team. (6) Crisis labels, while fixed ex ante from public record, involve dating choices; the VIX-based alternative labeling is provided precisely to expose label sensitivity (and it does: Test A's sign flips).

## Part 5 — Reproducibility

Every number in this document regenerates from two scripts and public data:

```
python forensic_audit_crash_alpha.py crash_alpha_analysis.csv
python rtm_econ_new_validations.py     # downloads data, runs Tests A–E, writes results_*.csv
```

Outputs: `forensic_reconstruction.csv`, `results_dfa_rolling.csv`, `results_event_study.csv`, `results_multiscale_coherence.csv`, `results_hill_tails.csv`, `results_shiller_centennial.csv`, `results_summary.csv`. Raw data copies included: `sp500_daily.csv`, `vix_daily.csv`, `shiller_sp500.csv`. Fixed seeds; no synthetic data anywhere in the new pipeline.

---

*Prepared as an independent adversarial validation for the RTM corpus (Doc 015). All computations executed on real, publicly verifiable market data. Negative results are reported as results.*
