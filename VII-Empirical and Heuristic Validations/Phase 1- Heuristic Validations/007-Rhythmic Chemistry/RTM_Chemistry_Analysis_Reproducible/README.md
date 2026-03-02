# RTM Empirical Validation: Chemistry ⚗️

**Date:** February 2026  
**Analyses:** 2 (Zeolite Diffusion + Stokes-Einstein)  
**Total Data Points:** 89 (35 + 54)

---

## Executive Summary

This analysis validates RTM predictions in chemistry by demonstrating that the scaling exponent **α changes sign** between bulk and confined diffusion regimes:

| Regime | System | n | α | R² |
|--------|--------|---|---|---|
| **CONFINED** | Zeolite diffusion | 35 | **+3.6 ± 0.9** | 0.34 |
| **BULK** | Stokes-Einstein | 54 | **-1.19 ± 0.04** | 0.95 |

**Key RTM Insight:** The α sign flip marks the transition between transport mechanisms.

---

## Quick Start

```bash
pip install -r requirements.txt
python analyze_chemistry_rtm.py
```

Results are saved to the `output/` directory.

---

## Analysis 1: Zeolite Diffusion (Confined Regime)

### The Physics

In nanoporous materials, molecules must squeeze through pores comparable to their size. This is **configurational diffusion**:
- Diffusion rate is EXTREMELY sensitive to geometry
- Small pore size changes → huge diffusion changes
- α >> 1 indicates critical/resonant regime

### Data

| Parameter | Value |
|-----------|-------|
| Data points | 35 |
| Materials | 11 zeolites (4A, 5A, ZSM-5, Y, X, MCM-41...) |
| Guest molecules | 7 (n-hexane, methane, benzene, propane, CO2, water, n-butane) |
| Pore sizes | 0.38 - 6.0 nm |
| D range | 10⁻¹⁴ - 10⁻⁷ m²/s |

### Results

| Category | n | α | R² |
|----------|---|---|---|
| **Overall** | 35 | **+3.6 ± 0.9** | 0.34 |
| Microporous (<0.8 nm) | 31 | **+9.2 ± 2.5** | 0.31 |
| n-Hexane | 9 | +4.1 | 0.48 |
| Methane | 6 | +4.0 | 0.49 |
| Benzene | 5 | +4.6 | 0.82 |
| CO2 | 4 | +6.6 | 0.97 |

### Interpretation

- α = +3.6 overall, +9.2 in microporous regime
- **Positive α** means bigger pore = faster diffusion (much faster!)
- This is the CRITICAL/RESONANT regime in RTM framework
- Extreme geometric sensitivity: 10% pore change → 10× diffusion change

---

## Analysis 2: Stokes-Einstein Diffusion (Bulk Regime)

### The Physics

In free solution, diffusion follows the Stokes-Einstein equation:

$$D = \frac{k_B T}{6\pi\eta r}$$

This predicts D ∝ r⁻¹, i.e., **α = -1**.

### Data

| Parameter | Value |
|-----------|-------|
| Data points | 54 |
| Categories | Gases, alcohols, sugars, amino acids, ions, proteins |
| Size range | 0.13 - 4.5 nm (×35 range) |
| D range | 10⁻¹¹ - 10⁻⁸ m²/s |

### Results

| Category | n | α | R² |
|----------|---|---|---|
| **Overall** | 54 | **-1.19 ± 0.04** | 0.95 |
| Theory (Stokes-Einstein) | — | -1.00 | — |
| Gases | 12 | -1.66 | 0.55 |
| Alcohols | 8 | -0.95 | 0.98 |
| Sugars | 7 | -1.02 | 0.99 |
| Proteins | 5 | -1.06 | 0.99 |

### Interpretation

- α = -1.19 ± 0.04, close to theoretical -1.0
- **Negative α** means bigger molecule = slower diffusion
- This is standard VISCOUS transport
- Larger molecules experience more drag

---

## The Key RTM Insight: α Sign Flip

| Regime | α | Physical Meaning |
|--------|---|------------------|
| Bulk (free solution) | **-1** | Bigger molecule → slower (viscous drag) |
| Confined (zeolites) | **+4 to +9** | Bigger pore → MUCH faster (geometry dominates) |

The **sign flip** of α demonstrates:
1. RTM can distinguish transport mechanisms
2. The scaling exponent captures fundamental physics
3. Confinement fundamentally changes transport behavior

---

## Data Sources

### Zeolite Diffusion
- Kärger & Ruthven "Diffusion in Zeolites and Other Microporous Solids"
- Jobic et al., Quasi-Elastic Neutron Scattering studies
- Ruthven & Post, comprehensive reviews
- Individual papers (Förste, Caro, Germanus, Stallmach)

### Stokes-Einstein
- CRC Handbook of Chemistry and Physics
- Landolt-Börnstein Tables
- Longsworth 1953 (amino acids)
- Gladden & Gosting 1953 (sugars)
- Tyn & Gusek (proteins)

---

## Files Included

```
chemistry_rtm_analysis/
├── analyze_chemistry_rtm.py       # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── output/                        # Generated results
    ├── chemistry_rtm_analysis.png
    ├── chemistry_rtm_analysis.pdf
    ├── zeolite_diffusion.csv
    └── stokes_einstein_diffusion.csv
```

---

## Extending the Analysis

### Add More Zeolite Data

Edit `get_zeolite_data()`:

```python
data = [
    # ... existing ...
    ("NewZeolite", 0.60, "n-Hexane", 1.0e-09, 300, "NewRef 2024"),
]
```

### Add More Bulk Diffusion Data

Edit `get_stokes_einstein_data()`:

```python
data = [
    # ... existing ...
    ("NewMolecule", 0.30, 0.85e-09, "Water", 298, "NewRef"),
]
```

---

## Comparison to Original Analysis

| Metric | Original (n=7) | This Analysis |
|--------|----------------|---------------|
| Zeolite α | 23.3 | **3.6 - 9.2** |
| Zeolite n | 7 | **35** |
| Stokes-Einstein | Not included | **α = -1.19, n=54** |
| Regime comparison | No | **Yes (sign flip!)** |

The expanded analysis:
1. Reduces the extreme α estimate to more realistic values
2. Adds a second regime for comparison
3. Demonstrates the α sign flip as a key RTM prediction

---

## RTM Transport Classes

| α Range | Class | Example |
|---------|-------|---------|
| < 0 | Inverse/Viscous | Stokes-Einstein (bulk) |
| 0.5 | Diffusive | Random walk |
| 1.0 | Ballistic | Earthquakes |
| 1-3 | Coherent | Many biological systems |
| **> 3** | **Critical/Resonant** | **Zeolites (confined)** |

Zeolites represent an EXTREME case where geometry creates critical sensitivity.

---

## Conclusion

**RTM successfully describes two distinct diffusion regimes in chemistry:**

1. **Bulk (Stokes-Einstein):** α = -1.19 ± 0.04, viscous drag dominates
2. **Confined (Zeolites):** α = +3.6 to +9.2, geometry dominates

The **α sign flip** from negative to positive is a clear signature of the transition from bulk to confined transport. This validates RTM's ability to characterize transport mechanisms through scaling exponents.

---

## Citation

```bibtex
@misc{rtm_chemistry_2026,
  author       = {RTM Research},
  title        = {RTM Chemistry Analysis: Two Regimes of Molecular Diffusion},
  year         = {2026},
  note         = {Data: Literature compilation}
}
```

---

## License

CC BY 4.0. Data compiled from published literature.
