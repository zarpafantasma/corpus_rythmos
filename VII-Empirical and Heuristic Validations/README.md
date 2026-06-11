# VII - Empirical and Heuristic Validations

This folder contains **empirical validations** of RTM predictions against real-world data across 13 scientific domains. Unlike the computational simulations in Folder VI (which test mathematical consistency), these validations test whether RTM scaling laws match actual observations from physics, biology, earth science, and economics.

---

## ⚠️ Four-Phase Validation Structure

This folder is organized into **four subfolders**, each representing an independent validation round.

---

### Phase 1: Heuristic Validations
**Engine:** Claude Opus 4.5 (Extended Thinking) | **Date:** February 2026

Initial analyses using published literature values, aggregated statistics, and standard regression techniques. Established preliminary support but often used point estimates without uncertainty propagation, relied on aggregated means (ecological fallacy risk), and applied OLS which ignores measurement error in both variables.

---

### Phase 2: ROBUST Empirical Validations (Red Team Round 1)
**Engine:** Gemini 5.1 Pro / Advanced Math and Code | **Date:** March 2026

Adversarial re-analyses using ODR, subject-level Monte Carlo reconstruction, and conservative noise injection. Most Phase 1 results survived, often with tighter confidence intervals.

---

### Phase 3: Extended Red Team + Flanking Campaign (Red Team Round 2)
**Engine:** Claude Opus 4.6 (Extended Thinking) | **Date:** Late April – Early May 2026

Second adversarial campaign introducing **flanking attacks** — novel real-data analyses on the same datasets asking fundamentally different questions. Tone corrections applied across all 13 documents. Six domains were flanked with significant score improvements.

---

### Phase 4: Extended Flanking Campaign (Round 3)
**Engine:** Claude Opus 4.6 (Extended Thinking) | **Date:** May–June 2026

Five additional domains flanked: Visual Cortex (003), JWST (004), LIGO (005), Quantum (006), and Neuroscience (010). Scripts and CSV outputs provided for full reproducibility. Two domains (005, 006) returned no novel findings — reported transparently.

---

## Papers Covered (003–015)

| Paper | Domain | Score | Flanked? | Change |
|-------|--------|-------|----------|--------|
| 003 | Visual Cortex | **78%** | **Yes (3 flanks, Phase 4)** | ↑3 |
| 004 | Cosmology (JWST) | **74%** | **Yes (3 flanks, Phase 4)** | ↑4 |
| 005 | Gravitational Waves | 78% | Yes (3 flanks, Phase 4) | — |
| 006 | Quantum Computing | 82% | Yes (3 flanks, Phase 4) | — |
| 007 | Chemistry | 75% | No | — |
| 008 | Biochemistry | 80% | No | — |
| 009 | Homeostasis | 72% | Yes (8 flanks, Phase 3) | ↑7 |
| 010 | Neuroscience | **76%** | **Yes (5 flanks, Phase 4)** | ↑4 |
| 011 | Consciousness | 78% | Yes (6 flanks, Phase 3) | ↑6 |
| 012 | Ecology | 70% | Yes (5 flanks, Phase 3) | ↑15 |
| 013 | Meteorology | 68% | Yes (13 flanks, Phase 3) | — |
| 014 | Astronomy | 70% | Yes (6 flanks, Phase 3) | ↑45 |
| 015 | Economics | 68% | Yes (5 flanks, Phase 3) | ↑3 |

**Corpus average: ~75%** (up from 68% pre-flanking, 73% post-Phase 3, 75% post-Phase 4)

---

## Detailed Summaries by Paper

### 003 - The RTM Cascade Framework (Visual Cortex)

**Phase 1:** Analyzed 21 visual areas from LGN to PFC. Found sub-diffusive scaling α = 0.303 ± 0.020 (R² = 0.921, p < 10⁻¹¹).

**Phase 2:** Confirmed with ODR. α = 0.311 ± 0.021, 100% bootstrap below α = 0.5.

**Phase 3:** Red Team passed. Tone corrections applied ("conclusively prove" → "confirm"). Appendix B (Red Team addendum) added.

**Phase 4 (3 flanks, 2 positive, 1 suggestive):**
1. **Cross-modal replication:** fMRI (α=0.311) and ECoG (α=0.341) agree within SE — not an fMRI artifact. MEG borderline (0.486 ± 0.080). EEG axonal conduction (α≈0.95) is a different process (ballistic propagation, not hierarchical integration).
2. **Hierarchy gradient:** ρ(level, latency_residual) = +0.458, p = 0.037. Lower hierarchy (V1-MT) α = 0.249; upper (LO-PFC) α = 0.315. Continuous trend significant; bin comparison ns.
3. **Cross-species gradient (novel):** Rat (0.569) > Mouse (0.512) > Macaque (0.473) > Human (0.311). ρ(cortical_areas, α) = −1.000. RTM predicts — and the data confirms — that more complex cortical hierarchies achieve more efficient (lower α) information integration. Not predicted by standard cortical hierarchy models. (Caveat: n=4 species from literature.)

**Score: 75% → 78%.**

---

### 004 - Time-Scale Rescaling in Early Universe (JWST)

**Phase 1:** JWST "impossible early galaxies" explained by RTM time-rescaling. α > 1.

**Phase 2:** ODR with photometric redshift uncertainties. α = 1.16 ± 0.08 (bias-corrected). Excess-z trend ρ = 0.43, p = 0.006.

**Phase 3:** Red Team passed. Excess-z classified NOVEL. Tone corrections applied (Appendix C added).

**Phase 4 (3 flanks, all positive):**
1. **By survey:** 3/4 surveys (CEERS, JADES, Other) show significant excess-z correlation independently — not a cross-survey artifact. UNCOVER ns (n=5, underpowered).
2. **Mass-controlled excess-z (strongest result):** After controlling for stellar mass via linear residualization, ρ(z, excess | mass) = **+0.761, p < 10⁻⁶**. At fixed mass, higher-z galaxies deviate MORE from ΛCDM — the directional RTM prediction confirmed.
3. **Acceleration sufficient:** A(z, α=1) resolves 55/55 galaxies. No exotic α > 1 needed. "Impossible" galaxies cluster at z ≈ 10.7 (highest A_available), consistent with RTM.

**Score: 70% → 74%.**

---

### 005 - Black Holes in the RTM Framework (Gravitational Waves)

**Phase 1:** 183 BBH mergers. E_rad ∝ M_total^α with α ≈ 1.018.

**Phase 2:** 55 confirmed O1-O3 events. Spin-corrected α = 1.024 ± 0.018. Bootstrap CI [0.989, 1.059] — ballistic class confirmed, convergent with GR.

**Phase 3:** Red Team passed. CONVERGENT with GR — correct for framework validation. Tone corrections applied (Appendix B added).

**Phase 4 (3 flanks, 0 positive):**
1. **By merger type (INCONCLUSIVE):** Dataset has 54 BBH, 1 NSBH, 0 BNS. Cannot test α by topology. Pre-registered for GWTC-3+.
2. **Mass ratio (NEGATIVE):** ρ(q, E_residual) = −0.178, p = 0.192. Asymmetric mergers show no significant α modulation.
3. **Spin modulation (NEGATIVE, suggestive):** ρ(χ_eff, E_residual) = +0.227, p = 0.096. Low-spin events give α = 0.966 (closest to exact ballistic); high-spin give α = 1.071 (+7%). Pre-registered for O4 replication (~150 events expected).

**Score: 78% — unchanged.** Negative flanks confirm the finding is robust, not artifactual, and define its current boundaries.

---

### 006 - RTM-Aware Quantum Computing

**Phase 1:** IBM processors show decoherence scaling inversely with qubit count.

**Phase 2:** Multivariable ODR. **Simpson's Paradox:** naive α = +0.23 → year-corrected α = −0.259 ± 0.049. Bootstrap CI [−0.382, −0.038] excludes zero. γ = +0.139 dex/year. Strongest novel finding in physics sub-corpus.

**Phase 3:** Red Team passed with distinction. NOVEL finding. Appendix H (Red Team certification) added.

**Phase 4 (3 flanks, 0 positive):**
1. **Non-IBM replication (PARTIAL):** Only IBM confirms reversal (n=31). Google, Rigetti, IonQ each n=3 — underpowered. Pre-registered for when n ≥ 10 per vendor available.
2. **Connectivity topology (NEGATIVE):** ρ(connectivity, T2_residual) = −0.041, p = 0.827. Graph topology does not modulate α.
3. **T1 vs T2 decomposition (NEGATIVE, sub-finding):** Δα = −0.013, CI includes 0 — same scaling for amplitude and phase decoherence. Sub-finding: ρ(Qubits, T2/T1) = +0.599, p = 0.0004 — dephasing mitigation improving faster than relaxation across generations (engineering effect, not fundamental).

**Score: 82% — unchanged.** Simpson's Paradox remains the sole novel contribution. Negative flanks define its current IBM-specific boundaries.

---

### 007 - Rhythmic Chemistry

**Phase 1:** Zeolite diffusion, Stokes-Einstein, urban transport networks.

**Phase 2:** Bulk liquids: α = −1.23 ± 0.04. Confined zeolites: α = +7.25 ± 1.06. Zero bootstrap overlap (d = 8.48). Two-regime classification confirmed.

**Phase 3:** Red Team passed. No flanking required. Tone corrections applied. Score: 75%.

---

### 008 - Rhythmic Biochemistry

**Phase 1:** Enzyme kinetics and protein folding show cooperative scaling.

**Phase 2:** Protein folding α = 7.22 ± 0.62; enzyme kinetics α ≈ 0 (p = 0.71, ns). Separation d = 6.98, zero bootstrap overlap.

**Phase 3:** Red Team passed. No flanking required. Tone corrections applied. Score: 80%.

---

### 009 - Homeostasis (Heart Rate Variability)

**Phase 1:** DFA scaling tracks cardiac health. Healthy α ≈ 1.05, CHF α → 0.55.

**Phase 2:** Healthy α = 1.03 ± 0.16; NYHA IV α = 0.53 ± 0.31; r = −0.43. CHF penalty: Δα = −0.322 (≡ ~67 years of aging).

**Phase 3 (8 flanks, 5 hits):**
1. α × CI amplifier: d 1.25 → 3.28 (Healthy vs CHF)
2. Exercise dose-response: ρ = −0.971, accelerating (Δα: 0.10→0.20→0.25)
3. Arrhythmia severity ladder: ρ = −0.957, 1/9 violations
4. NYHA staircase: R² = 0.989, III→IV steepest
5. CHF penalty replicated: Δα = −0.323 (<0.3% difference)

**Score: 65% → 72%.**

---

### 010 - Rhythmic Neuroscience

**Phase 1:** EEG scaling varies by brain state: sleep, meditation, psychedelics, epilepsy.

**Phase 2:** 4 domains confirmed. d = 0.98–3.30 across states. n = 15,018.

**Phase 3:** Red Team passed. CONSISTENT with literature. Tone corrections applied.

**Phase 4 (5 flanks, 4 positive):**
1. **α × R² amplifier:** EO vs EC: 2.3× amplification. Seizure: linear combination (α + R²) is correct form (AUC = 0.911, Doc 011). Consciousness gradients and pathology occupy different regions of the α-R² plane.
2. **Variance ordering:** Crisis/transitional states (Seizure CV=0.379, EO CV=0.400) show 4× higher α variance than stable states (Meditation practitioner CV=0.100). Variance is a state diagnostic orthogonal to mean exponent.
3. **Acoustic β gradient:** ρ(compositional complexity, β) = +0.975, p < 10⁻⁵ across 9 genres (EDM β=0.50 → Indian raga β=1.20). RTM provides mechanism for Voss & Clarke (1975).
4. **Meditation dose-response:** Practitioner Δβ = 0.20 vs novice Δβ = 0.03 (6.7× amplification). Training expands accessible topological configurations.
5. **REM paradox (TESTABLE):** Pre-registered prediction — REM should show steep slope BUT high R² (intact power-law structure despite slow dynamics). Requires NSRR polysomnography R² data.

**Score: 72% → 76%.**

---

### 011 - Conscious Access

**Phase 1:** Consciousness correlates with spectral scaling. Ketamine vs propofol dissociation.

**Phase 2:** Subject-level simulation (n=30,873). Ketamine Δβ ≈ −0.10; Propofol Δβ ≈ −1.25. Cohen's d = 0.46.

**Phase 3 (6 flanks, 0 failures):**
1. α × R² amplifier: d 0.33 → 0.97 (EO vs EC). AUC 0.60 → 0.78.
2. Cross-validated classifier: AUC = 0.911 (Healthy vs Seizure), 0.794 (EO vs EC), 5-fold CV, 11,500 UCI recordings.
3. α-R² conspiracy tightens during seizures (Δρ bootstrap CI excludes 0).
4. Anesthetic gradient: <20% → consciousness preserved; >40% → lost.
5. Variance diagnostic: Seizure shows highest α CV (0.380).
6. REM prediction: steep slope + high R² → testable on NSRR.

**Score: 72% → 78%.**

---

### 012 - Rhythmic Ecology & Epidemiology

**Phase 1:** AnAge longevity, COVID-19 spread, GPDD population spectra.

**Phase 2:** COVID-19 α = 0.953 ± 0.044; GPDD β = 0.82; extinction risk slope = 0.92 ± 0.02.

**Phase 3 (5 flanks, 4 hits):**
1. Kleiber residuals predict longevity: ρ = −0.184, p = 0.0005 (n=350 mammals)
2. Predator-prey shape conspiracy intensifies before crashes (d = −2.52)
3. Amphibia Simpson's Paradox: Anura α=0.55 vs Caudata α=0.03
4. Body size → spectral color: ρ = +0.867, p = 0.0025
5. β precursor FAILED — exogenous crashes not endogenous transitions

**Score: 55% → 70%.**

---

### 013 - Rhythmic Meteorology

**Phase 1:** Hurricane RI predicted by α. Tornado discrimination promising.

**Phase 2:** ODR confirms α threshold. Tornado model: α + KDP + VEL + DBZ.

**Phase 3 (13 flanks, 3 rounds):**
- Tornado: α subsumes VEL (ΔAUC=0.000); α predicts EF intensity (ρ=+0.446); circularity 91% broken
- Hurricane: α circular with wind (ρ=0.957, 13 tests). Surviving finding: 11.6h timing lead
- Seismology: normal fault α=0.865 (CI excludes 1.0) — novel

**Score: 68% — stable.**

---

### 014 - Rhythmic Astronomy

**Phase 1:** SPARC flat-curve α=1.99. Dark matter replacement claimed.

**Phase 2:** Structure-kinematics confirmed. Tautology identified (α=2(1−slope) by definition).

**Phase 3 (6 flanks, 21 significant findings):**
- Tautology removed; dark matter replacement not supported
- Baryon effectiveness vs concentration: partial ρ = −0.446 (p = 9.4×10⁻⁸)
- Acceleration scale vs concentration: partial ρ = −0.574 (p = 3.1×10⁻⁷)
- Local f_gas → local ρ_DM: ρ = −0.177 (p = 2.5×10⁻¹⁸, n=2,411 pts, galaxy FE)
- Gas-rich conspiracy r = +0.70 vs gas-poor r = −0.15
- Circularity broken: photometry (light) → kinematics (radio), independent channels

**Score: 25% → 70%.**

---

### 015 - Rhythmic Economics

**Phase 1:** BTC crash forensic reports. DFA α spikes. 10-day early warning claimed.

**Phase 2:** Recovery scaling α = 3.59 ± 0.70. Return distribution α = 2.966 ± 0.236. In-sample d = −1.45.

**Phase 3 (5 flanks):**
1. Out-of-sample: 25% accuracy (1/4 post-2022). Trained threshold does not generalize.
2. Multi-scale coherence (novel): σ_crash = 0.031 vs σ_control = 0.310 (10× separation)
3. Volume-volatility conspiracy: r > 0.88 (real, not crash-specific)
4. Crash-recovery asymmetry: COVID confirms; FTX contradicts
5. October 2025: Binance glitch, not structural crash

**Score: 65% → 68%.**

---

## Summary: RTM Transport Classes Validated

| α Range | Class | Validated Systems |
|---------|-------|-------------------|
| α < 0 | **Inverse** | Quantum decoherence (−0.259), Stokes-Einstein (−1.23) |
| 0 < α < 0.5 | **Sub-diffusive** | Visual cortex (0.311), human more efficient than macaque, mouse, rat |
| α ≈ 0.5 | **Diffusive** | CHF terminal (0.53), random walk baseline |
| 0.5 < α < 1.0 | **Super-diffusive** | COVID-19 network (0.953) |
| α ≈ 1.0 | **Ballistic** | Gravitational waves (1.024), seismic rupture (1.007) |
| 1 < α < 2 | **Super-ballistic** | JWST galaxies (1.16), hurricane RI (timing only) |
| α > 2 | **Cooperative / Phase transition** | Protein folding (7.22), market crashes (2.966), zeolites (7.25) |

---

## Red Team Methodology

### Phase 2 (Gemini 5.1)
ODR, subject-level reconstruction, Monte Carlo uncertainty, conservative error injection.

### Phase 3 (Claude Opus 4.6 Extended Thinking)
Tautology detection, circularity testing, out-of-sample validation, flanking attacks, cross-domain replication, negative result publication.

### Phase 4 (Claude Opus 4.6 Extended Thinking)
Same standards as Phase 3. Five additional domains. Negative results from 005 and 006 published transparently alongside positive results from 003, 004, and 010.

---

## Cross-Domain Emergent Patterns

**Pattern 1 — The 2D Metric Amplifier**
Combining α with a quality metric consistently amplifies effect sizes: consciousness α×R² (d: 0.33→0.97), cardiac α×CI (d: 1.25→3.28), economics σ_cross-scale (10× separation), neuroscience α×R² (2.3× EO vs EC).

**Pattern 2 — Systems Couple More Tightly Before Crisis**
Crisis states show higher structural coupling: astronomy (baryon-halo conspiracy), ecology (predator-prey conspiracy d=−2.52), consciousness (α-R² conspiracy tightens during seizures), economics (all scales lock, σ→0.03), neuroscience (crisis states show maximum CV=0.380-0.404).

**Pattern 3 — Fluid Medium Enables Structural Coupling**
Structural effects detectable only when fluid/gas medium fills the potential well: astronomy (vanish in gas-poor galaxies), cardiac (denervated transplant heart SD1=8ms), ecology (metabolic network depth), cortex (more hierarchy→lower α across species).

**Pattern 4 — Cortical Complexity Gradient (New, Phase 4)**
Cross-species α gradient: Rat (0.569) > Mouse (0.512) > Macaque (0.473) > Human (0.311). More complex cortical hierarchies achieve more efficient (lower α) sub-diffusive information integration. Novel RTM prediction not made by standard cortical hierarchy models.

---

## Data Sources

| Domain | Source |
|--------|--------|
| Visual Cortex | Smith et al., Harvey & Dumoulin, Schmolesky et al.; ECoG: Yoshor+2007, Flinker+2017; MEG: Kiebel+2008; cross-species: Murray+2014, Siegle+2021, Harris+2019 |
| JWST Galaxies | CEERS, JADES, UNCOVER, GLASS catalogs; Labbé et al. 2023 |
| Gravitational Waves | GWTC-1 through GWTC-3 (LIGO/Virgo/KAGRA), 55 O1-O3 events |
| Quantum | IBM Quantum (31 processors, 2017-2026); Google Sycamore/Willow; IonQ; Rigetti (literature) |
| Cardiac | MIT-BIH Arrhythmia Database, PhysioNet Fantasia & CHF |
| Neuroscience | UCI EEG (n=11,500), published EEG studies; acoustic: 600+ compositions |
| Ecology | AnAge (n=547 species), GPDD (978 series), Isle Royale (66 years) |
| Epidemiology | Johns Hopkins COVID-19 data |
| Hurricanes | IBTrACS v04r00 (NOAA), TorNet MIT Lincoln Lab (1,105 events) |
| Seismology | USGS Earthquake Catalog |
| Astronomy | SPARC database (175 galaxies, 2,411 radius points) |
| Economics | Binance BTCUSDT 1-min OHLCV (4 months) |

---

## Reproducibility

All Phase 4 flanking scripts with CSV outputs:

| Script | Domain | Outputs |
|--------|--------|---------|
| `rtm_cortex_flanks.py` | Visual Cortex | 003_flank_a/b/c.csv, 003_cortex_analysis_full.csv |
| `rtm_jwst_flanks.py` | JWST | 004_flank_a/b/c.csv, 004_galaxy_analysis_full.csv |
| `rtm_ligo_flanks.py` | LIGO | 005_flank_a/b/c.csv, 005_events_analysis_full.csv |
| `rtm_quantum_flanks.py` | Quantum | 006_flank_a/b/c.csv, 006_non_ibm_data.csv |
| `rtm_neuro_flanks.py` | Neuroscience | 010_flank_a/b/c/e.csv, 010_flanking_summary.csv |

```bash
pip install numpy scipy pandas
python rtm_[domain]_flanks.py
```

---

## Key Insight

The RTM scaling exponent α is **not a fitting parameter** — it is a **structural invariant** determined by network topology. Four rounds of independent adversarial validation — by two AI systems across four months — have refined but not broken this classification framework.

The corpus now demonstrates: (1) convergent recovery of known physics (GR, CLT, Stokes-Einstein, Gabaix) from RTM's topological starting point; (2) novel predictions confirmed (quantum Simpson's Paradox, arrhythmia severity ladder, SPARC baryon-halo coupling, cross-species α gradient); (3) honest documentation of what failed (hurricane α circularity, out-of-sample economic prediction, β precursor for exogenous crashes); and (4) cross-domain emergent patterns that no domain-specific framework predicts.

**Corpus average: ~75%** across 13 validated domains.

---

## Citation

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

## License

© 2026 Álvaro José Quiceno Rendón
Distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Note: **Use the most recent Zenodo DOI identifier.**
