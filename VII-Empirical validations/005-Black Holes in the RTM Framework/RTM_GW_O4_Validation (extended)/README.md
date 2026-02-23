# RTM Gravitational Waves Validation - O4 Extended 🌌

**Status:** ✓ BALLISTIC TRANSPORT VALIDATED  
**Data Sources:** LIGO-Virgo-KAGRA (GWTC-1 through GWTC-4.0)  
**Events Analyzed:** 183 BBH mergers (O1-O4)  
**Total Confident (GWTC-4.0):** 218 events  
**Total Candidates (O4 complete):** ~391 events  
**Date:** February 2026

---

## Executive Summary

This analysis validates RTM predictions using gravitational wave data from all LIGO-Virgo-KAGRA observing runs (O1-O4, 2015-2025), demonstrating that binary black hole (BBH) merger dynamics follow **BALLISTIC transport scaling** (α ≈ 1.0).

### Key Results

| Metric | Value | RTM Prediction | Status |
|--------|-------|----------------|--------|
| **α (raw)** | 1.018 ± 0.022 | α → 1 | ✓ VALIDATED |
| **α (spin-corrected)** | 1.020 | α → 1 | ✓ VALIDATED |
| **R²** | 0.922 | High correlation | ✓ |
| **p-value** | 2.3×10⁻¹⁰² | Highly significant | ✓ |

---

## RTM Transport Theory for Gravitational Waves

### The Scaling Law

RTM predicts energy transport follows:

**E_radiated ~ M_total^α**

For gravitational waves from BBH mergers:
- **E_radiated** = energy carried away by GWs ≈ (M₁ + M₂) - M_final
- **M_total** = M₁ + M₂

### Transport Classes

| Class | Exponent α | Physical Regime | GW Status |
|-------|------------|-----------------|-----------|
| Super-ballistic | α > 1.2 | Accelerated | Not observed |
| **BALLISTIC** | **α ≈ 1.0** | **Linear transport** | **✓ VALIDATED** |
| Sub-ballistic | 0.5 < α < 1 | Sub-diffusive | Not observed |
| Diffusive | α ≈ 0.5 | Random walk | Not observed |

### Why α = 1?

Gravitational waves exhibit **ballistic transport** because:
1. Energy radiates directly from the source
2. No scattering or trapping in spacetime
3. GW energy scales linearly with total mass
4. Einstein's equations predict linear energy-momentum relationship

---

## Observing Run Summary

### Detection Evolution (2015-2025)

| Run | Period | Events | Confident | Rate/Month | BNS Range |
|-----|--------|--------|-----------|------------|-----------|
| **O1** | 2015-2016 | 3 | 3 | 0.75 | 70 Mpc |
| **O2** | 2016-2017 | 8 | 8 | 0.89 | 100 Mpc |
| **O3a** | 2019 | 39 | 39 | 6.5 | 130 Mpc |
| **O3b** | 2019-2020 | 40 | 40 | 8.0 | 130 Mpc |
| **O4a** | 2023-2024 | 128 | 128 | 16.0 | 160 Mpc |
| **O4b/c** | 2024-2025 | 173 | (pending) | 12.4 | 170 Mpc |
| **TOTAL** | 2015-2025 | **391** | **218** | - | - |

### Key Milestones

- **2015-09-14:** First detection (GW150914)
- **2017-08-17:** First neutron star merger (GW170817)
- **2019-05-21:** First intermediate-mass BH (GW190521, 142 M☉)
- **2023-05-24:** O4 begins
- **2024-03:** 200th O4 candidate
- **2025-11-18:** O4 concludes with ~250 candidates

---

## Chirp Mass Distribution

### Bimodal Structure (GWTC-4.0)

The chirp mass distribution shows two clear peaks:

| Peak | Location | Origin |
|------|----------|--------|
| **Peak 1** | ~8-10 M☉ | Stellar-mass BH (standard core collapse) |
| **Peak 2** | ~25-30 M☉ | Massive BH (failed SN / pair instability gap) |

**Statistics:**
- Mean: 18.6 M☉
- Median: 13.2 M☉
- Range: 2.4 - 64.0 M☉

This bimodal structure supports recent stellar evolution predictions (Schneider et al. 2023, Maltsev et al. 2025).

---

## Effective Spin Distribution

### χeff = (m₁χ₁ + m₂χ₂)/(m₁ + m₂) · cos(θ)

**Statistics:**
- Mean: 0.061
- Std: 0.158
- Range: -0.47 to +0.63

**Key Findings:**
- **79.2%** have |χeff| < 0.2 → low spins dominate
- **31.7%** have χeff < 0 → anti-aligned spins present
- Distribution centered at ~0 → supports **isolated binary formation**

---

## RTM Scaling Analysis

### Raw Scaling (No Corrections)

```
log(E_radiated) = α · log(M_total) + c

α = 1.0181 ± 0.0220
R² = 0.9223
p-value = 2.3 × 10⁻¹⁰²
n = 183 events
```

### Spin-Corrected Scaling

Accounting for spin effects on radiated energy:
```
E_corrected = E_radiated / (1 + 0.3·|χeff|)

α = 1.0197
R² = 0.9214
```

### Interpretation

Both raw and spin-corrected analyses yield **α ≈ 1.0**, confirming RTM's prediction of **BALLISTIC transport** for gravitational wave energy.

---

## Notable Events

### Highest Mass: GW190521
- **M₁:** 85 M☉, **M₂:** 66 M☉
- **M_final:** 142 M☉ (intermediate-mass BH!)
- **z:** 0.82
- First IMBH ever observed

### Highest SNR: GW200129
- **SNR:** 26.4
- **M₁:** 34.5 M☉, **M₂:** 28.9 M☉
- Excellent waveform characterization

### First Detection: GW150914
- **M₁:** 35.6 M☉, **M₂:** 30.6 M☉
- **M_final:** 63.1 M☉
- 3 M☉c² radiated as gravitational waves!

---

## Files

```
rtm_gw_o4/
├── analyze_gw_rtm.py         # Main analysis script (THIS FILE)
├── README.md                  # Documentation
├── requirements.txt           # Dependencies
└── output/
    ├── rtm_gw_o4_6panels.png/pdf   # Main validation figure
    ├── rtm_gw_scaling.png          # Mass-energy scaling detail
    ├── rtm_gw_mass_plot.png        # M₁ vs M₂ scatter
    ├── bbh_events_all.csv          # All 183 events
    ├── bbh_events_o1_o3.csv        # O1-O3 events
    ├── bbh_events_o4.csv           # O4 events
    ├── catalog_summary.csv         # Run statistics
    └── rtm_scaling_results.csv     # α values and statistics
```

---

## RTM Predictions for Future Runs

### O5 (Expected 2027)
- BNS range: 250+ Mpc
- Expected: ~1000+ events
- RTM predicts: α should remain ≈ 1.0

### Next-Generation Detectors
- Einstein Telescope (ET)
- Cosmic Explorer (CE)
- LISA (space-based)

**RTM Prediction:** α = 1.0 will hold across all mass scales and frequencies.

---

## References

### LIGO-Virgo-KAGRA Publications
1. LIGO Scientific Collaboration et al. (2025). *GWTC-4.0: Updated Catalog*
2. Abbott et al. (2023). *GWTC-3: Compact Binary Coalescences*. Phys. Rev. X 13, 041039
3. Abbott et al. (2021). *GWTC-2: Compact Binary Coalescences*. Phys. Rev. X 11, 021053
4. Abbott et al. (2019). *GWTC-1: First Catalog*. Phys. Rev. X 9, 031040

### Astrophysics
5. Schneider et al. (2023). *Bimodal black hole mass distribution*
6. Maltsev et al. (2025). *Remnant mass model*
7. Mapelli (2021). *Binary Black Hole Mergers*. Front. Astron. Space Sci.

### Data Sources
8. GWOSC: https://gwosc.org/
9. GraceDB: https://gracedb.ligo.org/
10. GWTC-4.0: https://gwosc.org/GWTC-4.0/

---

## Citation

```bibtex
@misc{rtm_gw_o4_2026,
  author       = {RTM Research},
  title        = {RTM Gravitational Waves Validation - O4 Extended},
  year         = {2026},
  note         = {183 BBH mergers, α=1.018±0.022, R²=0.922, BALLISTIC validated}
}
```

---

## License

CC BY 4.0
