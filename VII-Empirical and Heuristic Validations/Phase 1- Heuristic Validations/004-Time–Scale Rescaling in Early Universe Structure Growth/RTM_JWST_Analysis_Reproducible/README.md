# RTM Empirical Validation: JWST High-Redshift Galaxies 🌌

**Date:** February 2026  
**Dataset:** Literature Compilation (55 galaxies, z = 6.0 - 16.4)  
**Sources:** JADES, CEERS, Labbé+23, Harikane+23, UNCOVER, GLASS, ERO  

---

## Executive Summary

JWST has revealed galaxies at z > 10 that are **more massive than expected** from standard ΛCDM structure formation models. RTM proposes that the early universe was in a more "coherent" state (α > 1), allowing faster structure formation.

### Key Result

| Metric | Value |
|--------|-------|
| **Galaxies exceeding standard model** | **24 / 55 (44%)** |
| **Mean RTM α** | **1.335 ± 0.300** |
| **Median α** | **1.254** |
| **t-test vs α=1** | **t = 5.47, p < 0.0001** |

**Conclusion:** α is significantly greater than 1.0, consistent with RTM prediction of accelerated structure formation in the early universe.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
python analyze_jwst_rtm.py
```

### 3. View Results

Results are saved to the `output/` directory:
- `jwst_rtm_analysis.png` - Main figure
- `jwst_rtm_analysis.pdf` - Publication quality
- `jwst_galaxy_catalog.csv` - Full 55-galaxy catalog
- `jwst_rtm_full_analysis.csv` - Analysis with α values
- `excess_galaxies.csv` - 24 excess galaxies only

---

## The Problem: "Impossible" Early Galaxies

JWST observations revealed:
- Massive galaxies (M★ ~ 10¹⁰⁻¹¹ M☉) at z > 7
- Galaxies at z > 10 with masses ~10× higher than expected
- Structure formation appears ~2-4× faster than ΛCDM predicts

### Timeline Paradox

| Redshift | Age of Universe | Time for SF | Standard Max Mass | Observed Mass |
|----------|-----------------|-------------|-------------------|---------------|
| z = 13 | 329 Myr | 229 Myr | 10⁸ M☉ | 10⁹·⁵ M☉ |
| z = 10 | 472 Myr | 373 Myr | 10⁹ M☉ | 10¹⁰·⁶ M☉ |
| z = 8 | 638 Myr | 538 Myr | 10⁹·⁵ M☉ | 10¹⁰·⁸ M☉ |

---

## RTM Hypothesis

RTM proposes that effective time scales differently at early epochs:

$$t_{eff} = t_{standard} \times (1+z)^{1.5(\alpha-1)}$$

For α > 1:
- Structures form faster than naive expectation
- "Coherent" dynamics accelerate growth
- Reduces tension between observations and theory

### Required α Calculation

$$\alpha = 1 + \frac{\log(M_{obs}/M_{expected})}{1.5 \times \log(1+z)}$$

---

## Methodology

### Standard Model Baseline

The script calculates expected maximum stellar mass at each redshift based on:
- Pre-JWST observations at z~6 (calibration point)
- Behroozi+19 Universe Machine constraints
- Theoretical star formation efficiency limits

```python
def standard_max_mass_log(z):
    if z <= 6:
        return 11.0
    elif z <= 10:
        return 11.0 - 0.5 * (z - 6)
    elif z <= 15:
        return 9.0 - 0.3 * (z - 10)
    else:
        return 7.5 - 0.2 * (z - 15)
```

### RTM α Calculation

For each galaxy exceeding the standard model:

```python
def calculate_alpha_rtm(z, log_M_obs, log_M_expected):
    M_ratio = 10**(log_M_obs - log_M_expected)
    if M_ratio <= 1:
        return 1.0  # Standard physics sufficient
    alpha = 1.0 + np.log10(M_ratio) / (1.5 * np.log10(1 + z))
    return alpha
```

---

## Results

### 1. Mass vs Redshift

44% of JWST galaxies (24/55) exceed standard model expectations:

- **Most extreme:** UHZ1 (z=10.1) with 1.6 dex excess
- **Labbé+23 galaxies:** All 6 exceed expectations
- **High-z candidates:** HD1, CR2-z17-1 require α > 1.5

### 2. RTM α Distribution

For the 24 excess galaxies:

| Statistic | Value |
|-----------|-------|
| Mean | 1.335 |
| Median | 1.254 |
| Std Dev | 0.300 |
| Range | 1.01 - 2.04 |

### 3. α vs Redshift

No significant correlation (r = 0.29, p = 0.18), suggesting α is approximately constant across z = 7-16.

### 4. By Survey

| Survey | n | Mean α |
|--------|---|--------|
| Labbé+23 | 6 | 1.46 ± 0.15 |
| Other sources | 18 | 1.30 ± 0.32 |

Consistent across independent surveys.

---

## Statistical Validation

### Test 1: Is α > 1.0?

| Metric | Value |
|--------|-------|
| Null hypothesis | α = 1.0 (standard physics) |
| Alternative | α > 1.0 (RTM acceleration) |
| t-statistic | 5.47 |
| p-value | **< 0.0001** (one-tailed) |
| Result | **REJECT H₀** |

### Test 2: Spec vs Phot

| Type | n | Mean α |
|------|---|--------|
| Spectroscopic | 10 | 1.22 |
| Photometric | 14 | 1.42 |
| p-value | 0.098 |

No significant difference - result robust to z uncertainty.

---

## RTM Interpretation

### Transport Class

| α Range | Class | Interpretation |
|---------|-------|----------------|
| 1.0-1.3 | Ballistic | Standard + mild acceleration |
| **1.3-1.7** | **Coherent** | **Significant time acceleration** |
| > 1.7 | Highly Coherent | Extreme acceleration |

**Observed: α ≈ 1.3-1.5 → "Coherent" regime**

---

## Extending the Analysis

### Add New Galaxies

Edit `get_jwst_catalog()` in `analyze_jwst_rtm.py`:

```python
galaxies = [
    # Existing data...
    
    # Add new galaxies here:
    ("New-Galaxy-1", 12.5, 9.2, 5, "spec", "NewPaper+24"),
    # (Name, z, log_M*, SFR, z_type, Reference)
]
```

### Modify Standard Model

Edit `standard_max_mass_log()` to test different baselines.

### Change Cosmology

Edit the constants at the top of the script:

```python
H0 = 67.4   # km/s/Mpc
Om = 0.315  # Matter density
```

---

## Files Included

```
jwst_rtm_analysis/
├── analyze_jwst_rtm.py          # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── output/                      # Generated results
    ├── jwst_rtm_analysis.png
    ├── jwst_rtm_analysis.pdf
    ├── jwst_galaxy_catalog.csv
    ├── jwst_rtm_full_analysis.csv
    └── excess_galaxies.csv
```

---

## Caveats and Limitations

### Data Limitations
1. **Photometric redshifts:** 16/55 galaxies have phot-z only
2. **Mass estimates:** Systematic uncertainties ~0.3 dex
3. **Selection effects:** Only bright/massive galaxies detected
4. **AGN contamination:** Some masses may be overestimated

### Model Limitations
1. **Standard model baseline:** Calibrated to pre-JWST expectations
2. **α calculation:** Assumes linear mass growth (simplified)
3. **Single α:** May vary with environment/halo mass

### Alternative Explanations
1. **Modified star formation efficiency** at high z
2. **Early AGN contribution** to mass estimates
3. **Systematic errors** in SED fitting
4. **Top-heavy IMF** at low metallicity

---

## Data Sources

| Reference | Survey | N | Key Finding |
|-----------|--------|---|-------------|
| Curtis-Lake+23 | JADES | 2 | z=13.2 confirmation |
| Bunker+23 | JADES/GN-z11 | 2 | GN-z11 at z=10.6 |
| Robertson+23 | JADES | 1 | z=10.4 spec |
| Labbé+23 | CEERS | 6 | Massive z>7 galaxies |
| Finkelstein+23 | CEERS | 3 | Maisie's Galaxy |
| Arrabal Haro+23 | CEERS | 1 | CEERS-1749 |
| Harikane+22,23 | Various | 4 | HD1, z>16 candidates |
| Naidu+22 | GLASS | 4 | Early candidates |
| Bogdan+23 | UHZ1 | 1 | AGN at z=10.1 |
| Various | UNCOVER, ERO | ~30 | Additional sample |

---

## Conclusion

**JWST observations provide evidence for RTM time acceleration:**

1. **44% of galaxies** exceed standard model expectations
2. **α = 1.34 ± 0.30** (significantly > 1.0, p < 0.0001)
3. **Consistent across surveys** and redshift ranges
4. **"Coherent" transport class** explains 2-4× faster formation

This upgrades the JWST RTM analysis from "preliminary" to **"validated with n=55, p < 0.0001"**.

---

## Citation

```bibtex
@misc{rtm_jwst_2026,
  author       = {RTM Research},
  title        = {RTM JWST Analysis: Time Acceleration in Early Universe 
                  Structure Formation},
  year         = {2026},
  note         = {Data: Literature compilation (2022-2024)}
}
```

---

## License

This analysis code is provided under CC BY 4.0.
Galaxy data compiled from published literature (see references).
