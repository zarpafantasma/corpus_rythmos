# RTM Meteorology — New Flanking Campaign on Real Data

**Five new flanks opening angles the prior campaigns never tested**

Date: July 5, 2026
Deliverables: `rtm_meteo_new_flanks.py`, real data CSVs, six `results_*.csv`, this document.

---

## Executive Summary

The prior RTM-Atmo campaigns settled two things: the tornado result is strong and real (already a standalone arXiv paper), and the hurricane α is circular (α derives from wind, then is correlated with wind-driven intensification). Two other domains — seismology and the climate 1/f spectrum — were reported as "validations" but rest on weaker ground: the seismology α = 1 is **tautological** (rupture duration τ is *defined* as L/v, so α ≈ 1 is forced; it survives even after shuffling the L–v pairing), and the climate spectrum was tested on a curated summary table, not raw data.

This campaign opens **five genuinely new angles**, each on real public data (23,412 USGS earthquakes; global monthly temperature 1850–2026; daily Melbourne temperature 1981–1990), chosen so that RTM's scaling claims are tested against laws the framework did **not** hand-pick. The honest scorecard:

| Flank | Test | Result | Verdict for RTM |
|---|---|---|---|
| **F1** | Gutenberg-Richter b-value (frequency-magnitude) | b = 1.00 (MLE), 1.11 (LS), R² = 0.99 | **CONFIRMED, non-tautological** — replaces the circular rupture-α with a real scale-free law |
| **F2** | 1/f spectrum on *real* temperature | Global monthly β = 0.94 ✓; local daily β = 0.38 ✗ | **PARTIAL** — RTM's β≈1 claim is **scale-dependent**, true globally, false locally |
| **F3** | Omori aftershock decay (8 great quakes) | mean p = 0.95 ± 0.17 | **CONFIRMED, non-tautological** — a real temporal scaling law RTM never examined |
| **F4** | b-value by tectonic setting | ridge b = 1.25 > subduction b ≈ 1.0 | **CONFIRMED** — the exponent carries topological (tectonic) information, exactly RTM's thesis |
| **F5** | Fractal dimension of epicenters | D2 = 1.18, R² = 0.999 | **CONFIRMED** — earthquakes are spatially scale-free (fractal), consistent with RTM |

**Bottom line — good news, honestly bounded.** Four of five flanks confirm real scale-free structure, and critically, **F1 and F3 replace the tautological seismology "validation" with two genuine, non-circular scaling laws** (frequency-magnitude and temporal-decay) that RTM's ballistic/scale-free framing accommodates naturally. F4 is the most RTM-native result: the scaling exponent measurably shifts with tectonic topology, which is the framework's central claim. F2 is the most valuable *negative*: RTM's "the atmosphere is 1/f" claim is real only at the global scale and fails locally — a boundary condition the document should state. None of these is an unprecedented discovery (all are known geophysics), but together they upgrade the seismology/climate domains from "tautology + curated table" to "real, reproducible, non-circular scaling laws with an honest scale-dependence caveat."

---

## Part 1 — Why These Flanks (the gap being filled)

The prior seismology "validation" claimed α = 1.007 proves "ballistic transport." Section 0 of the earlier meteorology analysis established this is tautological: the catalog defines rupture duration as τ = L / v_rupture, and because rupture velocity varies little (CV ≈ 0.14), α ≈ 1 is nearly forced. The decisive demonstration: **shuffling which velocity pairs with which length leaves α = 1.000 ± 0.016** — the exponent comes from the definition "duration = length ÷ speed," not from any measured L–τ physics.

So the seismology domain needed real, non-derived scaling laws. Earthquakes offer several that RTM never touched: the Gutenberg-Richter frequency-magnitude law, the Omori aftershock-decay law, spatial fractal clustering, and tectonic b-value variation. All are computed here from a real 23k-event USGS catalog, and none is a τ = L/v tautology.

The climate domain needed the same treatment: RTM's β ≈ 1 spectrum was asserted from a 3-row summary table. Here it is tested on real daily and monthly temperature series.

---

## Part 2 — The Five Flanks in Detail

### F1 — Gutenberg-Richter b-value (CONFIRMED, replaces the tautology)

The Gutenberg-Richter law, N(≥M) ∝ 10^(−bM), is a genuine scale-free relationship between earthquake frequency and magnitude. On 23,409 real USGS events (M ≥ 5.5, 1965–2016):

- Least-squares fit: **b = 1.11 ± 0.02** (R² = 0.991)
- Aki maximum-likelihood estimator: **b = 1.00**

The universal b-value is ≈ 1.0 (Gutenberg & Richter 1944), meaning the magnitude-frequency distribution is scale-free — each unit of magnitude is ~10× rarer. This is the *real* seismological analog of RTM's "ballistic/scale-free" class, and unlike the rupture-τ result it is **not** a definitional artifact: it is a statistical property of independently catalogued events. This flank gives the seismology domain a legitimate scaling law to stand on.

### F2 — 1/f spectrum on real temperature (PARTIAL — the valuable negative)

RTM claims the atmosphere sits at β ≈ 1 (1/f "critical" noise). Tested on real data via DFA (β = 2α−1):

| Series | n | DFA α | β | Verdict |
|---|---|---|---|---|
| Global monthly (GCAG, detrended) | 2,115 | 0.97 | **0.94** | ✓ ≈ 1/f |
| Local daily (Melbourne, deseasonalized) | 3,650 | 0.69 | **0.38** | ✗ closer to white |

**The β ≈ 1 claim is scale-dependent.** Globally aggregated temperature is genuinely near 1/f (β = 0.94), supporting the RTM "critical" framing at planetary scale. But local daily temperature is far from 1/f (β = 0.38, closer to white/weakly-persistent noise). This is exactly the kind of boundary the document should state: the "1/f atmosphere" is an emergent property of *global aggregation*, not a universal feature of temperature at every scale. Reporting only the global number (as the curated table did) overstates the generality.

### F3 — Omori aftershock law (CONFIRMED, non-tautological)

Omori's law states aftershock rate decays as n(t) ∝ t^(−p), p ≈ 1. Measured across the eight largest earthquakes in the catalog (M ≥ 8.3), counting aftershocks within 500 km over 100 days:

| Mainshock | n aftershocks | Omori p | R² |
|---|---|---|---|
| M9.1 2004 (Sumatra) | 57 | 0.70 ± 0.21 | 0.57 |
| M9.1 2011 (Tōhoku) | 96 | 0.93 ± 0.12 | 0.87 |
| M8.8 2010 (Chile) | 37 | 0.81 ± 0.13 | 0.84 |
| M8.7 1965 (Rat Is.) | 26 | 1.23 ± 0.15 | 0.90 |
| M8.6 2005 (Nias) | 34 | 0.82 ± 0.16 | 0.77 |
| M8.4 2007 (Bengkulu) | 28 | 1.02 ± 0.18 | 0.83 |
| M8.3 1994 (Kuril) | 27 | 0.98 ± 0.20 | 0.80 |
| M8.3 2015 (Illapel) | 39 | 1.11 ± 0.20 | 0.79 |

**Mean p = 0.95 ± 0.17.** This is a real temporal scaling law — the rate at which a fault system relaxes after a great earthquake follows a clean power law near p = 1, consistent across eight independent events on three continents. RTM never examined temporal aftershock decay; it is a legitimate new scaling result for the seismology domain and, unlike the rupture-α, carries genuine physical content (the p-value reflects real relaxation dynamics, not a definition).

### F4 — b-value by tectonic setting (CONFIRMED, most RTM-native)

RTM's central thesis is that scaling exponents encode structural/topological differences. Testing whether the b-value shifts with tectonic setting:

| Region | Setting | b-value | n |
|---|---|---|---|
| Mid-Atlantic Ridge | Spreading/transform | **1.25** | 789 |
| Indonesia | Subduction | 1.03 | 2,572 |
| Japan/Kuril | Subduction | 0.99 | 1,986 |

The mid-ocean ridge (extensional, transform-dominated) shows a distinctly **higher** b-value (1.25) than the subduction zones (≈ 1.0), matching the known seismological result that b-value tracks stress regime and tectonic style (Schorlemmer et al. 2005). This is the most RTM-consistent of the five: the same scaling exponent takes measurably different values in structurally different transport regimes — the framework's core claim, demonstrated on real data without any circularity.

### F5 — Fractal dimension of epicenters (CONFIRMED)

The correlation (Grassberger-Procaccia) dimension of 1,500 sampled epicenters: **D2 = 1.18 ± 0.01** (R² = 0.999). Earthquakes do not fill the 2D Earth surface uniformly; they cluster on a fractal set of dimension ≈ 1.2, concentrated along plate-boundary networks. This spatial scale-invariance (Kagan & Knopoff 1980) is consistent with RTM's picture of geophysical processes organized on scale-free structural networks. Known result, cleanly reproduced.

---

## Part 3 — Honest Grading

**Good (real, reproducible, non-circular):**
- **F1 (b = 1.00)** and **F3 (Omori p = 0.95)** are the important wins: they give the seismology domain two *genuine* scaling laws to replace the tautological rupture-α. Both are computed from a real 23k-event catalog and neither is a definitional artifact.
- **F4** is the strongest RTM-thematic result: the scaling exponent demonstrably encodes tectonic topology (ridge 1.25 vs subduction 1.0).
- **F5** confirms spatial scale-invariance (D2 = 1.18).

**Honestly qualified:**
- All five are **known geophysics** (Gutenberg-Richter 1944; Omori 1894/Utsu 1961; Schorlemmer 2005; Kagan-Knopoff 1980). RTM's contribution is the *unifying framing* — treating b, p, D2 as members of one "scale-free transport" family — not the discovery of any individual law. Present them as "RTM accommodates these established laws within a single scaling ontology," not as new findings.
- **F2 is a genuine negative for the strong version of the claim:** the "1/f atmosphere" holds only under global aggregation (β = 0.94) and fails locally (β = 0.38). This should be stated as a boundary condition, not buried.

**What this does NOT fix:**
- The hurricane α circularity (unchanged — that door stays closed).
- The fact that these are consistency/convergence results, not novel predictions. The tornado result remains the only genuinely novel, operationally useful RTM-Atmo finding.

---

## Part 4 — Recommendation for Doc 013

1. **Replace, don't defend, the seismology tautology.** Swap the rupture-α "ballistic validation" for F1 (Gutenberg-Richter b = 1.00) and F3 (Omori p = 0.95) as the seismology domain's real scaling content. Keep the rupture-α only as an explicit "dimensional consistency check (τ = L/v ⇒ α = 1 by construction)," honestly labeled as near-tautological.
2. **Add F4 as the domain's headline** — the b-value/tectonic-topology link is the most RTM-native real result and the cleanest demonstration of "exponent encodes structure."
3. **State F2's scale-dependence** — correct the "1/f atmosphere" claim to "1/f under global aggregation; local temperature is closer to white noise." This turns an overclaim into an honest, interesting boundary condition.
4. **Frame all as convergence, not discovery.** These strengthen the *coherence* of RTM's scaling ontology across geophysics; they do not add novel predictions. The tornado paper remains the flagship.

---

## Reproducibility

```
python rtm_meteo_new_flanks.py     # downloads data, runs F1-F5, writes results_*.csv
```
Outputs: `results_summary.csv`, `results_gutenberg_richter.csv`, `results_spectral_1overf.csv`, `results_omori.csv`, `results_bvalue_tectonic.csv`, `results_fractal_dimension.csv`. Data (`eq23k.csv`, `monthly_temp.csv`, `daily_temps.csv`) fetched from public mirrors at runtime. Fixed seeds. No synthetic data.

---

*Independent flanking campaign for the RTM corpus (Doc 013). All analyses on real, public geophysical data. Confirmations, the scale-dependence of the 1/f claim, and the tautology being replaced are all reported plainly. Known results are labeled as known.*
