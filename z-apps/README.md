# z-apps — Functional RTM Applications

This folder contains **working Streamlit applications** that demonstrate RTM principles in real-time monitoring scenarios. These are not simulations or validations — they are operational tools designed to detect phase transitions in atmospheric and financial systems.

---

## Applications

### 1. Atmospheric Monitor (`atmospheric-monitor/`)

A dual-module climate intelligence system.

**Modules:**

| Module | Function | Data Source |
|--------|----------|-------------|
| **CLIMATE EXTREMES** | Real-time extreme weather monitoring | Weather APIs (Open-Meteo, etc.) |
| **HURRICANE TRACKER** | Rapid Intensification prediction via α | IBTrACS + simulated real-time feeds |

**Hurricane Tracker Features:**
- **Live α Calculation:** Computes wind-pressure coupling exponent in real-time
- **Phase Detection:** LAMINAR (α > 1.5) → DECAY (1.25 < α < 1.5) → FRACTURE (α < 1.25)
- **Countdown Timer:** T-MINUS hours to predicted intensity explosion
- **Historical Replays:** Otis (2023), Milton (2024), Patricia (2015)

**Key RTM Insight:** The α-drop (Topological Fracture) precedes NHC official alerts by 12-14 hours. The application demonstrates this predictive lead time with annotated historical data.

**Run:**
```bash
cd atmospheric-monitor
pip install -r requirements.txt
streamlit run app_rtm.py
```

---

### 2. Cryptocurrency Monitor (`cryptocurrency_monitor/`)

A financial market radar using RTM coherence physics.

**Modules:**

| Module | Function |
|--------|----------|
| **LIVE RADAR** | Real-time BTC/ETH monitoring via Binance API |
| **SYSTEMIC HEALTH** | Market health dashboard with α-based diagnostics |
| **FORENSIC ANALYSIS** | Historical crash replays with RTM annotations |
| **MARKET PHYSICS** | Fat tails, power laws, recovery scaling calculator |

**Forensic Events Included:**

| Event | Date | RTM Finding |
|-------|------|-------------|
| **FTX Collapse** | Nov 2022 | Chronic Viscosity (α ≈ 1.2) for 4 days — 100h warning |
| **Black Thursday** | Mar 2020 | Phase Bifurcation (α = 1.76) — 60 min warning |
| **China Ban** | May 2021 | High-Energy Turbulence (α = 1.33) — instant recovery predicted |
| **Control Group** | Sep 2023 | Laminar Flow (α ≈ 0.45) — 0% false alarm rate |

**Market Physics Features:**
- **Inverse Cubic Law:** Global α ≈ 2.97 (fat tails are structural, not anomalies)
- **Recovery Calculator:** Uses robust ODR slope (3.59) to estimate recovery time from drawdown
- **α Distribution:** Simulated 10-year histogram showing fracture probability

**Run:**
```bash
cd cryptocurrency_monitor
pip install -r requirements.txt
streamlit run app.py
```

---

## The RTM Coherence Exponent (α) in Both Applications

Both applications use the same fundamental metric: the **RTM Coherence Exponent (α)**, which measures how efficiently a system transports information/energy across scales.

| α Range | State | Atmospheric Meaning | Financial Meaning |
|---------|-------|---------------------|-------------------|
| α > 1.5 | **LAMINAR** | Stable atmosphere | Healthy market flow |
| 1.2 < α < 1.5 | **DECAY** | Structural weakening | Viscosity warning |
| α < 1.2 | **FRACTURE** | Rapid intensification imminent | Crash/bifurcation imminent |
| α > 2.0 | **BIFURCATION** | (Not applicable — hurricanes intensify) | Market structure breaking |

**The key insight:** α measures *topological coherence*, not kinetic activity. A hurricane's winds may be calm while α collapses (predicting future explosion). A market's price may be stable while α rises (predicting future crash). The structural geometry breaks *before* the observable symptoms appear.

---

## Technical Stack

| Component | Library |
|-----------|---------|
| UI Framework | Streamlit |
| Visualization | Plotly, Folium (maps) |
| Data Processing | Pandas, NumPy |
| Financial Data | ccxt (Binance API) |
| Weather Data | Open-Meteo, custom APIs |

**Requirements:**
- Python 3.8+
- Internet connection (for live data)
- ~500MB RAM per application

---

## Data Files Included

**Cryptocurrency Monitor:**
- `BTCUSDT-1m-2020-03.csv` — Black Thursday (COVID crash)
- `BTCUSDT-1m-2021-05.csv` — China Ban shock
- `BTCUSDT-1m-2022-11.csv` — FTX collapse
- `BTCUSDT-1m-2023-09.csv` — Control group (stable period)
- `BTCUSDT-1m-2025-10.csv` — Binance glitch anomaly
- `crash_alpha_analysis.csv` — Pre-computed α values for forensic events

**Atmospheric Monitor:**
- `RTM CLIMATE-Global-Architecture-Vision.pdf` — System architecture documentation

---

## Disclaimers

### Hurricane Tracker
```
⚠️ EXPERIMENTAL TOOL — NOT FOR EMERGENCY DECISIONS
This application is a research demonstration of RTM atmospheric physics.
For life-safety decisions, always follow official NHC/local emergency guidance.
```

### Cryptocurrency Monitor
```
⚠️ NOT FINANCIAL ADVICE
This application demonstrates RTM market physics for educational purposes.
Past performance does not guarantee future results.
Do not make trading decisions based solely on this tool.
```

---

## Relationship to RTM Corpus

These applications operationalize findings from:

| Application | Source Papers |
|-------------|---------------|
| Hurricane Tracker | Paper 013 (RTM-Atmo), Appendix F (Tornado FAR) |
| Cryptocurrency Monitor | Paper 015 (Rhythmic Economics), Phase 2 validations |

The applications do not introduce new theory — they package validated RTM metrics into usable real-time interfaces.

---

## Citation

If you use this work, please cite:

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

---

## License

© 2026 Álvaro José Quiceno Rendón  
Distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
Note: **Use the most recent Zenodo DOI identifier.**
