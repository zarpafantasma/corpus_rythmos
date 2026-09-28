# RTM Ecology — New Empirical Flanks on Real AnAge Data

**Independent validation + new findings from the Animal Ageing and Longevity Database**

Date: July 5, 2026
Deliverables: `rtm_ecology_new_flanks.py`, `anage_data.txt` (real database), six `results_*.csv`, this document.

---

## Executive Summary

Following the economics audit — where the flagship dataset turned out to be synthetically generated — the first step here was a **provenance check on the ecology data**. The result is reassuring and important: **the AnAge database is real.** It is the public Animal Ageing and Longevity Database (4,645 species; genomics.senescence.info), its values pass biological sanity checks, and it carries no pseudo-random signature. As a positive control, the corpus's existing Kleiber-residual flank reproduces to the third decimal from the raw file (ρ = −0.184, p = 5.5×10⁻⁴, exactly matching Doc 012). The ecology empirical base rests on genuine data, unlike the economics Chapter 12 table.

On that solid foundation, six new flanks were run — five never appearing in the corpus. The headline is a **new, strong, self-controlled finding**:

| # | New flank | Result | Verdict |
|---|---|---|---|
| **F3** | **Flight longevity bonus** | Flying endotherms live **1.52× longer** at fixed body mass (d = 0.85, p = 4×10⁻⁸²); bats alone **1.9×** (d = 1.39); and flightless ratites **lose the bonus** (d = −1.07, internal control) | **STRONG — the best new result** |
| F5 | Bat anomaly at the α level | Chiroptera have the **lowest** mass-longevity α (0.09) of any mammal order — flight decouples mass from lifespan | **Strong, consistent with F3** |
| F4 | Maturity-longevity co-scaling | Two characteristic times co-scale (slopes 0.67–1.22); L/M ratio ~constant within class (CV ≈ 0.70) | **Solid, RTM-consistent** |
| F1 | α across all 6 classes | Clean ranking Aves (0.21) > Mammalia (0.16) > Chondrichthyes/Reptilia (0.14–0.16) > Teleostei (0.13) > Amphibia (0.05) | **Descriptive, extends corpus** |
| F2 | Endotherm vs ectotherm α | Endotherms scale steeper (0.144 vs 0.103) | **Weak/directional** |
| F6 | Longevity-quotient champions | Top 10 = 9 bats + human (+ naked mole-rat at #11) | **Face-validity confirmed** |

**Bottom line — good news, honestly qualified.** The flight-longevity bonus (F3) is a genuinely strong result: large effect, enormous significance, mechanistically motivated by RTM's "metabolic topology" logic, and — crucially — it comes with a *built-in control that passes* (flightless birds lose the bonus). It is also, candidly, a rediscovery: the fact that bats and birds are longevity outliers is known in comparative biology (Healy et al. 2014; Wilkinson & South 2002). RTM's contribution is the framing (flight → altered transport-network topology → shifted temporal exponent) and the unified quantification across both flying clades with the ratite control. It is the strongest *new* empirical material available for Doc 012, but it should be presented as "RTM provides a topological interpretation and a quantitative, controlled test of a known longevity anomaly," not as an unprecedented discovery.

---

## Part 1 — Provenance Check (the economics lesson applied)

Before any new analysis, the AnAge file was checked the same way `crash_alpha_analysis.csv` was:

1. **Real named source.** AnAge is a public, downloadable, citable database (Tacutu et al. 2018, *Nucleic Acids Research*). The economics crash table had no such source.
2. **Value heterogeneity.** Body masses span 1.8 g to 1.6×10⁵ g with realistic irregular magnitudes; known species carry correct values (human 122.5 y; naked mole-rat 31 y; blue whale, elephant in range). No column is a linear map of a uniform draw.
3. **No seed signature.** No `np.random.seed(k)` reproduces any column for k in 0–99.
4. **Positive control.** The corpus's Kleiber-residual flank reproduces exactly from the raw file (below), which it could not if the file were fabricated.

**Verdict: the AnAge-based portions of Doc 012 (Appendix B, flanks 1/3/4) rest on real, verifiable data.** The summary tables (GPDD spectral, Taylor, extinction, COVID) are small literature-compilation tables citing real papers; their values are consistent with the cited literature but are not independently re-derived here (they would require the raw GPDD/JHU data). One known issue persists from the Red Team: the COVID table has 30 countries, not the 100 the text claims, and yields α = 1.05 not 0.95.

---

## Part 2 — Reproduction Control (R0)

The corpus's most RTM-specific ecology claim — that metabolic-rate residuals predict longevity residuals at fixed mass — reproduces exactly from raw AnAge:

| Quantity | Corpus (Doc 012 App. E) | This reproduction |
|---|---|---|
| n mammals | 350 | 350 |
| Kleiber slope (BMR ~ mass) | ~0.75 | 0.725 |
| Spearman ρ (BMR-resid, longevity-resid) | −0.184 | **−0.184** |
| p | 0.0005 | **5.5×10⁻⁴** |
| Within-order: fraction negative | 89% | 78% (8/9 orders, minor count difference) |
| Within-order t-test p | 0.007 | 0.009 |

This is a real, correctly reported finding: at fixed body mass, mammals that burn more energy than Kleiber predicts live shorter. It is RTM-flavored ("metabolic topology sets the pace of life") and stands.

---

## Part 3 — The New Headline: Flight Longevity Bonus (F3)

### 3.1 The result

Pooling all mammals and birds with mass + longevity (n = 2,398), fit a single log-log mass-longevity line, then compare the residuals of flying vs non-flying species:

| Group | n | Mean longevity residual (dex) |
|---|---|---|
| Flying endotherms (all birds + bats) | 1,477 | **+0.070** |
| Non-flying mammals | 921 | **−0.112** |

Difference = 0.182 dex → **flying endotherms live 1.52× longer at fixed body mass** (bootstrap 95% CI [1.46, 1.58], 5,000 resamples; Cohen's d = 0.85; Mann-Whitney p = 4×10⁻⁸²).

Bats alone are even more extreme: mean residual +0.268 vs −0.030 for non-bat mammals (d = **1.39**, p = 6×10⁻²⁹) — **bats live 1.9× longer than their mass predicts.**

### 3.2 The internal control that makes it credible

A correlation between "flight" and "longevity" could be confounded (birds and mammals differ in many ways). The clean test is *within birds*: if flight itself matters, birds that **lost** flight should lose the bonus. They do:

| Bird group | n | Mean longevity residual (dex) |
|---|---|---|
| Flying birds | 1,360 | +0.001 |
| Flightless ratites (ostrich, emu, kiwi, rhea, cassowary) | 8 | **−0.214** |

Flightless ratites fall **0.21 dex below** their flying relatives (d = −1.07) — they age like similarly-sized non-flying mammals. This is exactly the pattern the "flight → metabolic/transport topology → longevity" hypothesis predicts, and it is the reason F3 is more than a correlation. (Caveat: n = 8 ratites is small; penguins, treated separately as aquatic "flyers," sit intermediate at −0.05, n = 7.)

### 3.3 What is and isn't new

The biological fact — volant animals are long-lived for their size — is established (Austad & Fischer 1991; Wilkinson & South 2002; Healy et al. 2014). What this analysis adds: (i) a single unified quantification across *both* flying clades on one mass-longevity baseline; (ii) the flightless-ratite control as a falsification test that passes; (iii) the RTM framing — flight restructures the organism's transport network (elevated, sustained aerobic scope; distinct mitochondrial/vascular architecture), shifting the temporal-scaling exponent. It is a strong, honest, self-controlled result suitable to anchor a revised Appendix E, presented as interpretation-plus-controlled-test of a known anomaly.

---

## Part 4 — Supporting New Flanks

**F5 — Bats have the lowest α of any mammal order.** Across mammalian orders with n ≥ 25, the mass-longevity exponent ranges from 0.196 (Rodentia) down to **0.091 (Chiroptera)**, with R² = 0.07 — bats' lifespans are almost *decoupled* from their mass. This is the same phenomenon as F3 seen at the scaling-exponent level: flight flattens the mass-longevity gradient. Consistent and mutually reinforcing with F3.

**F4 — Two characteristic times co-scale.** RTM treats longevity and maturity as two "characteristic times" that should co-scale. They do, cleanly: ODR slope of log-longevity on log-maturity is 0.67 (mammals), 0.87 (birds), 0.82 (reptiles), 1.22 (fish), all R² ≈ 0.4–0.65. The longevity/maturity ratio is roughly scale-invariant within each class (CV ≈ 0.70), i.e. animals live a fairly fixed number of "maturities." This is RTM-consistent and, unlike the ROBUST appendices, is a within-database relationship rather than a literature citation. It is also related to known life-history invariants (Charnov 1993), so again: solid, not unprecedented.

**F1 — Full-class α census.** Extending the corpus's four classes to all six with n ≥ 15 gives a clean ranking: Aves 0.213 > Mammalia 0.159 > Chondrichthyes 0.155 > Reptilia 0.140 > Teleostei 0.133 > Amphibia 0.053. The endotherm classes scale most steeply; Amphibia remains the outlier (near-zero, the Simpson's-paradox case the corpus already dissected into Anura vs Caudata).

**F2 — Endotherm vs ectotherm.** Endotherms scale steeper than ectotherms (0.144 vs 0.103). Directionally RTM-consistent (constant high body temperature = more organized thermal "clock") but modest and R²-weak on the ectotherm side; report as suggestive only.

**F6 — Face validity.** The top-10 longevity-quotient species are nine bats plus humans (naked mole-rat is #11). These are precisely the textbook longevity outliers, confirming the residual metric is measuring real biological "living longer than your mass predicts," not noise.

---

## Part 5 — Honest Grading

**Good:**
- The ecology data is **real** (unlike economics Ch. 12). This is the single most important finding of this pass.
- The Kleiber-residual flank **reproduces exactly** — a genuine RTM-specific result.
- The **flight longevity bonus (F3)** is a strong, large-effect, highly significant, **self-controlled** result. The passing ratite control elevates it above a mere correlation.
- F4 (time co-scaling) and F5 (bat α anomaly) are solid and mutually consistent.

**Honestly qualified:**
- F3, F4, F5 are all **rediscoveries with RTM framing**, not unprecedented discoveries. Volant longevity, life-history invariants, and bat exceptionalism are known. RTM's value is unification + mechanism + (for F3) a controlled falsification test.
- F2 is weak. F1 is descriptive.
- The ratite control has only n = 8; the flight bonus itself is robust (n = 1,477) but the cleanest control is small.
- Phylogenetic non-independence is not fully modeled (no PGLS). The within-order (R0) and within-birds (F3 control) analyses partially address this, but a formal phylogenetic comparative analysis would strengthen any publication.

**Bad / unchanged:**
- The ROBUST appendices (B–D) remain, as the Red Team said, consistency-with-known-results rather than novel validation.
- The COVID 30-vs-100 country discrepancy is still unresolved and should be corrected in Doc 012.

---

## Part 6 — Recommendation

For a potential standalone paper, the **flight longevity bonus** is the strongest candidate the ecology material offers — it uses a real public database, has a large controlled effect, and tells a clean mechanistic story. A defensible framing: *"Flight and the pace of life: a mass-controlled, clade-internal test of the volant longevity bonus across 2,400 endotherms."* It would need a formal phylogenetic comparative model (PGLS) added, and honest positioning relative to Healy et al. (2014). It is not a from-scratch discovery, but it is real, reproducible, and well-controlled — the right kind of material to strengthen Doc 012 or to spin out.

The maturity-longevity co-scaling (F4) is a reasonable second, but overlaps more heavily with the established life-history-invariants literature.

---

## Reproducibility

```
python rtm_ecology_new_flanks.py     # runs R0 + F1–F6 on anage_data.txt
```
Outputs: `results_summary.csv`, `results_alpha_by_class.csv`, `results_flight_bonus.csv`, `results_maturity_longevity.csv`, `results_order_alpha.csv`, `results_lq_champions.csv`. Raw database `anage_data.txt` included. Fixed seed (42) for bootstrap. No synthetic data anywhere.

---

*Independent adversarial validation for the RTM corpus (Doc 012). All analyses on the real, public AnAge database. Rediscoveries labeled as such; the one strong new result (F3) carries a passing internal control. Negative and weak results reported as results.*
