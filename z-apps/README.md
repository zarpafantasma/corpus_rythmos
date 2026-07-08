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

## Technical Stack

| Component | Library |
|-----------|---------|
| UI Framework | Streamlit |
| Visualization | Plotly, Folium (maps) |
| Data Processing | Pandas, NumPy |
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

This application do not introduce new theory — they package validated RTM metrics into real-time monitoring interfaces. Previous versions contained claims that were invalidated by the Red Team audit. v3 apps contain only what survived.

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
