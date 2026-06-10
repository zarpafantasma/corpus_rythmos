# z-apps — RTM Structural Monitoring Tools

This folder contains **working Streamlit applications** that demonstrate RTM principles in real-time structural monitoring. Both apps were rebuilt from scratch (v3, May 2026) after an independent adversarial audit (Red Team, Claude Opus 4.6 Extended Thinking) invalidated several claims from previous versions. These apps show **only what survived**.

---

## Applications

### 1. Atmospheric Structural Radar (`atmospheric-monitor/`)

> **NOTICE:** Strictly academic proof of concept. Not an official meteorological alert system. Does not abuse, spam, or mass-scrape commercial data APIs.

**Modules:**

| Module | Function | Data Source |
|--------|----------|-------------|
| **TORNADO VORTEX RADAR** | Live TOR classification via RTM α proxy | NWS API (api.weather.gov) |
| **MULTI-SCALE COHERENCE** | Cross-scale σ monitoring | Open-Meteo (7-day hourly) |
| **SEISMOLOGY REFERENCE** | Calibration anchor + fault-type table | Published catalogs |
| **RED TEAM FINDINGS** | Full audit transparency | Red Team April 2026 |

**Tornado Vortex Radar (headline — strongest finding in entire RTM corpus):**
- Fetches active Tornado Warnings from NWS in real-time
- Computes α proxy = log₁₀(VEL_estimated) / log₁₀(L_polygon) from alert text
- Classifies using TorNet-calibrated threshold: α > 0.74 → confirmed tornado class
- Dark map with color-coded warning polygons and classification table
- Reference: d = 0.96, CV AUC = 0.751, α subsumes VEL (ΔAUC = 0.000), circularity 91% broken

**Multi-Scale Coherence (novel metric — Red Team survivor):**
- Computes α at 4 time scales (1h, 3h, 6h, 12h) simultaneously
- Tracks cross-scale σ: crisis states → 0.03 (all scales locked), normal → 0.31 (independent)
- Same 10x separation observed in financial crashes (cross-domain finding)

**What was removed from previous versions:** Hurricane Tracker (α circular with wind, ρ = 0.957, 13 tests), historical hurricane simulations (pre-programmed curves, not predictions), ocean dynamics (synthetic data), "EVACUATE" language.

**Run:**
```bash
cd atmospheric-monitor
pip install -r requirements.txt
streamlit run app_rtm.py
```

---

### 2. Economic Structural Radar (`cryptocurrency_monitor/`)

> **DISCLAIMER:** Academic, read-only topological analysis tool. Does NOT execute trades, does NOT mine cryptocurrency, and is NOT financial advice. Out-of-sample crash prediction accuracy: 25%.

**Modules:**

| Module | Function | Data Source |
|--------|----------|-------------|
| **MULTI-SCALE COHERENCE** | Cross-scale σ (live + historical) | Kraken API (hourly) + Binance CSVs |
| **LIVE MICROSTRUCTURE** | Real-time α monitoring (4 assets) | Kraken API (1-min) |
| **FORENSIC LABORATORY** | Historical crash anatomy (post-hoc) | Binance 1-min CSVs |
| **MARKET PHYSICS** | Fat tails + recovery scaling | Convergent results |
| **RED TEAM FINDINGS** | Full audit transparency | Red Team April 2026 |

**Multi-Scale Coherence (headline — the only genuinely novel RTM economic metric):**
- Live: fetches 14 days of hourly data from Kraken, computes α at 1h/3h/6h/12h scales
- Historical: loads Binance 1-min CSVs, computes α at 1/5/15/60 min scales
- Tracks cross-scale σ with gauge, time series, and reference values
- Crash months σ = 0.031-0.034 vs control σ = 0.310 (10x separation)

**Forensic Events (labeled as post-hoc, not prospective prediction):**

| Event | Date | Finding | Status |
|-------|------|---------|--------|
| **FTX Collapse** | Nov 2022 | Chronic Viscosity (α ≈ 1.2, 4 days) | Forensic |
| **Black Thursday** | Mar 2020 | Sudden Bifurcation (α = 1.76) | Forensic |
| **China Ban** | May 2021 | Turbulence, no fracture (α = 1.33) | Forensic |
| **Control Group** | Sep 2023 | Laminar (α ≈ 0.45, zero false alarms) | Confirmed |
| **Binance Glitch** | Oct 2025 | Technical anomaly, not fundamental crash | Not validated |

**What was removed from previous versions:** "EXIT MARKETS" command, "crash early warning system" framing, "96-hour FTX warning" claim, "15-hour October prediction" claim, all language implying operational trading signals.

**Run:**
```bash
cd cryptocurrency_monitor
pip install -r requirements.txt
streamlit run app_crypto.py
```

---

## The Multi-Scale Coherence Metric (σ) — Shared Across Both Applications

Both applications share the same novel finding as their headline module: **Multi-Scale Coherence**.

The concept: compute α at multiple time scales simultaneously. If all scales show the same α (low σ), the system is in a coherent, coupled state — a phase transition is underway. If each scale shows a different α (high σ), the system is operating normally — no phase transition.

| σ Range | State | Atmospheric Meaning | Financial Meaning |
|---------|-------|---------------------|-------------------|
| σ < 0.05 | **HYPER-COHERENT** | All scales locked — structural crisis | All scales locked — crash signature |
| 0.05 < σ < 0.15 | **ELEVATED** | Scales coupling — monitor | Scales coupling — watch |
| σ > 0.15 | **NORMAL** | Scales independent | Scales independent |

**Reference values (from Red Team BTC analysis):**
- Crisis (COVID March 2020): σ = 0.031
- Crisis (FTX November 2022): σ = 0.034
- Control (September 2023): σ = 0.310

This metric is RTM-native: it is not measured by any standard meteorological or financial indicator. The cross-domain consistency (atmosphere + markets) is the central empirical contribution of the RTM monitoring framework.

---

## Technical Stack

| Component | Library |
|-----------|---------|
| UI Framework | Streamlit |
| Visualization | Plotly, Folium (maps) |
| Data Processing | Pandas, NumPy |
| Financial Data | ccxt (Kraken API) |
| Weather Data | Open-Meteo API |
| Tornado Alerts | NWS API (api.weather.gov) |
| Typography | JetBrains Mono, Inter |
| Design System | GitHub-dark palette |

**Requirements:**
- Python 3.8+
- Internet connection (for live data feeds)
- No scipy or heavy dependencies

---

## Relationship to RTM Corpus

| Application | Source Paper | Score | Key Surviving Finding |
|-------------|-------------|-------|-----------------------|
| Atmospheric Radar | Doc 013 (Rhythmic Meteorology) | 68% | Tornado d = 0.96, α subsumes VEL |
| Economic Radar | Doc 015 (Rhythmic Economics) | 68% | Multi-Scale σ: 10x crash vs control |

These applications do not introduce new theory — they package validated RTM metrics into real-time monitoring interfaces. Previous versions contained claims that were invalidated by the Red Team audit. v3 apps contain only what survived.

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
