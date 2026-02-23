# RTM Empirical Validation: Systematic Rapid Intensification Analysis 🌀

**Date:** February 2026  
**Dataset:** IBTrACS v04r00 (East Pacific Basin 2021-2024)  
**Total Records:** 16,429  
**Storms Analyzed:** 48  

---

## Executive Summary

This analysis validates the **RTM Cascade Framework** by systematically testing whether the wind-pressure coupling exponent (α) can predict Rapid Intensification (RI) in tropical cyclones.

### Key Result

| Metric | Value |
|--------|-------|
| **Statistical significance** | **p < 0.0001** |
| **Effect size (Cohen's d)** | **3.07** (very large) |
| **Mean lead time** | **12 hours** |
| **Lead time range** | **6-18 hours** |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
python analyze_hurricane_ri.py
```

The script will automatically download IBTrACS data from NOAA if not present.

### 3. View Results

Results are saved to the `output/` directory:
- `RI_Systematic_Analysis_EP.png` - Main figure
- `RI_Systematic_Analysis_EP.pdf` - Publication quality
- `ep_storms_alpha_summary.csv` - All storm statistics
- `ri_events_ep.csv` - RI events with α data
- `ri_lead_times.csv` - Lead time analysis

---

## Configuration

Edit the top of `analyze_hurricane_ri.py` to customize:

```python
# Basin selection
BASIN = "EP"  # Options: "EP" (East Pacific), "NA" (Atlantic), "WP" (West Pacific)

# Analysis parameters
MIN_WIND_KT = 35           # Minimum intensity to include
REFERENCE_PRESSURE = 1010  # Reference pressure for ΔP calculation
RI_THRESHOLD_KT = 30       # NHC RI definition
```

---

## Methodology

### Wind-Pressure Coupling (α)

$$\alpha = \frac{\ln(V)}{\ln(\Delta P)}$$

where:
- V = Maximum sustained wind (kt)
- ΔP = Pressure deficit (1010 mb - P_center)

**Interpretation:**
- Lower α → More efficient coupling → Better organized storm
- α < 1.3 → High probability of rapid intensification

### Rapid Intensification Definition

RI = Wind increase ≥ 30 kt in 24 hours (NHC standard)

---

## Results

### 1. α Clearly Separates Intensification Categories

| Category | n | α_min (mean ± SD) |
|----------|---|-------------------|
| **RAPID (≥30 kt/24h)** | 26 | **1.221 ± 0.101** |
| MODERATE (15-30) | 15 | 1.472 ± 0.198 |
| SLOW (<15) | 6 | 1.688 ± 0.190 |

### 2. Statistical Test: RAPID vs SLOW

| Metric | Value |
|--------|-------|
| t-statistic | -8.576 |
| p-value | < 0.0001 |
| Cohen's d | **3.07** |

*Effect sizes > 0.8 are considered "large"; 3.07 is exceptional.*

### 3. Lead Time Analysis

The α-drop precedes RI onset by:

| Statistic | Hours |
|-----------|-------|
| Mean | 12 |
| Median | 12 |
| Minimum | 6 |
| Maximum | 18 |

### 4. Top Rapid Intensifiers

| Storm | Max Δ24h (kt) | α_min | Max Wind (kt) |
|-------|---------------|-------|---------------|
| **OTIS** | **93** | **1.11** | **145** |
| JOVA | 80 | 1.12 | 140 |
| DARBY | 65 | 1.18 | 120 |
| FERNANDA | 65 | 1.15 | 115 |
| HILARY | 60 | 1.15 | 120 |
| ROSLYN | 55 | 1.17 | 115 |
| NORMA | 55 | 1.11 | 115 |
| LIDIA | 50 | 1.13 | 120 |

---

## Physical Interpretation

1. **α represents structural efficiency**: Lower α = tighter wind-pressure relationship
2. **RI storms are "superfluid"**: They convert pressure deficit to wind more efficiently
3. **The α-drop signals transition**: When α falls below ~1.3, rapid organization is occurring
4. **Lead time is actionable**: 6-18 hours provides useful forecast window

---

## Extending the Analysis

### Analyze Different Basins

```python
# In analyze_hurricane_ri.py:
BASIN = "NA"  # North Atlantic
# or
BASIN = "WP"  # West Pacific
```

### Use Historical Data

Download extended IBTrACS data:
- Full history: `ibtracs.ALL.list.v04r00.csv`
- Since 1980: `ibtracs.since1980.list.v04r00.csv`

From: https://www.ncei.noaa.gov/products/international-best-track-archive

---

## Files Included

```
ri_analysis/
├── analyze_hurricane_ri.py      # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── output/                      # Generated results
    ├── RI_Systematic_Analysis_EP.png
    ├── RI_Systematic_Analysis_EP.pdf
    ├── ep_storms_alpha_summary.csv
    ├── ri_events_ep.csv
    └── ri_lead_times.csv
```

---

## Comparison with Single-Case Otis Analysis

| Metric | Original Otis Analysis | Systematic Analysis |
|--------|------------------------|---------------------|
| Lead time | 12h (single case) | 6-18h (8 cases) |
| α threshold | ~0.37 (different formula) | ~1.2-1.3 |
| Validation | n=1 | **n=48 storms, 26 RI events** |

The systematic analysis **validates and extends** the original Otis finding.

---

## Limitations

1. **East Pacific only**: Need validation in other basins (Atlantic, West Pacific)
2. **Best-track data**: 3-hour temporal resolution limits precision
3. **Retrospective**: Needs prospective operational testing
4. **α formula sensitivity**: Different reference pressures may affect values

---

## Conclusion

**The RTM wind-pressure coupling exponent (α) is a statistically robust predictor of Rapid Intensification.**

- Effect size is **exceptional** (d = 3.07)
- Lead time is **operationally useful** (6-18 hours)
- Pattern is **consistent** across 26 RI events
- Otis finding is **validated** by systematic analysis

This elevates RTM from "interesting single case" to "validated predictive framework."

---

## Data Source

**IBTrACS:** International Best Track Archive for Climate Stewardship  
**Version:** v04r00  
**DOI:** 10.25921/82ty-9e16  
**URL:** https://www.ncei.noaa.gov/products/international-best-track-archive

---

## Citation

```bibtex
@misc{rtm_hurricane_2026,
  author       = {RTM Research},
  title        = {RTM Hurricane Analysis: Systematic Validation of Wind-Pressure 
                  Coupling as Rapid Intensification Predictor},
  year         = {2026},
  note         = {Data: IBTrACS v04r00 (NOAA)}
}
```

---

## License

This analysis code is provided under CC BY 4.0. 
IBTrACS data is provided by NOAA/NCEI.
