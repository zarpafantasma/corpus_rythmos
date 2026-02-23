# RTM Empirical Validation: Visual Cortex 👁️

**Date:** February 2026  
**Visual Areas:** 21 (expanded from n=10)  
**Status:** ✓ VALIDATED

---

## Executive Summary

This analysis reveals how temporal processing scales with spatial receptive field (RF) size across the visual cortex hierarchy.

| Metric | Value |
|--------|-------|
| **α** | **0.303 ± 0.020** |
| R² | 0.921 |
| p-value | 6.07 × 10⁻¹² |
| Transport Class | **SUB-DIFFUSIVE** |

**Key Finding:** The visual system shows SUB-DIFFUSIVE scaling (α < 0.5), meaning it is MORE EFFICIENT than random diffusion at integrating information across space.

---

## Quick Start

```bash
pip install -r requirements.txt
python analyze_visual_cortex_rtm.py
```

Results are saved to the `output/` directory.

---

## The Physics: Why Sub-Diffusive?

### RTM Prediction

RTM predicts that temporal processing time (τ) scales with spatial extent (L) as:

$$\tau \propto L^\alpha$$

Different transport mechanisms give different α:

| α | Mechanism | Example |
|---|-----------|---------|
| α < 0 | Inverse | Quantum decoherence |
| **0 < α < 0.5** | **Sub-diffusive** | **Visual cortex** |
| α = 0.5 | Diffusive | Random walk |
| α = 1 | Ballistic | Wave propagation |
| α > 1 | Super-linear | Cooperative folding |

### Why α ≈ 0.30 in Visual Cortex?

The visual system achieves sub-diffusive scaling through:

1. **Parallel processing:** All points in RF processed simultaneously
2. **Hierarchical coding:** Predictive signals reduce integration time
3. **Feedforward dominance:** Fast sweeps outpace lateral integration

If processing were serial (random walk), we'd expect α = 0.5.
If processing were fully parallel, we'd expect α → 0.
The observed α ≈ 0.30 reflects a mix of both.

---

## Dataset

### Visual Areas (n=21)

| Level | Areas | RF (deg) | Latency (ms) |
|-------|-------|----------|--------------|
| 0 | LGN-M, LGN-P | 0.15-0.30 | 25-35 |
| 1 | V1 | 0.8 | 45 |
| 2 | V2 | 2.0 | 55 |
| 3 | V3, V3A | 3.5-4.5 | 60-65 |
| 4 | V4/hV4, MT/V5 | 5.5-6.0 | 70-75 |
| 5 | LO1, LO2, VO1, VO2, MST, IPS0-1 | 8-14 | 85-105 |
| 6 | pIT, cIT, IPS2 | 15-20 | 105-120 |
| 7 | aIT, FEF | 22-25 | 130-135 |
| 8 | PFC | 35 | 150 |

### Data Sources

- **RF Sizes:** 
  - Smith et al. (2001) *Cerebral Cortex*
  - Motter (2009) *J Neurosci*
  - Harvey & Dumoulin (2011) *J Neurosci*

- **Response Latencies:**
  - Schmolesky et al. (1998) *J Neurophysiol*
  - Temporal dynamics studies in *PNAS*, *Nature Comm*

---

## Results

### Main Scaling Relationship

$$\text{Latency} \propto \text{RF}^{0.303}$$

| Metric | Value |
|--------|-------|
| α | 0.303 ± 0.020 |
| R² | 0.921 |
| p | 6.07 × 10⁻¹² |

### By Visual Stream

| Stream | n | α | R² |
|--------|---|---|---|
| Ventral (What) | 11 | 0.335 | 0.972 |
| Dorsal (Where) | 10 | 0.292 | 0.908 |

Both streams show remarkably similar scaling!

### Test vs Diffusive (α = 0.5)

| Test | Result |
|------|--------|
| t-statistic | -9.67 |
| p-value | < 0.0001 |

The visual system is **significantly more efficient** than diffusive processing.

---

## Comparison: Original vs Expanded

| Metric | Original | Expanded |
|--------|----------|----------|
| Visual areas | 10 | **21** |
| Hierarchy levels | 4 | **9** |
| α | ~0.3 | **0.303 ± 0.020** |
| R² | ~0.8 | **0.921** |
| Statistical test | Limited | **p < 10⁻¹¹** |
| Includes streams | No | **Yes** |

The expanded analysis:
- Doubles the number of areas
- Extends from LGN to PFC
- Provides stream-specific analysis
- Achieves excellent statistical power

---

## RTM Transport Classes (Updated)

| α | Class | Systems |
|---|-------|---------|
| α < 0 | **Inverse** | Quantum decoherence (-0.35), Stokes-Einstein (-1.19) |
| 0 < α < 0.5 | **Sub-diffusive** | **Visual cortex (0.30)** |
| α ≈ 0.5 | **Diffusive** | HRV (0.5), random walk |
| α ≈ 1 | **Ballistic** | Earthquakes (1.0), GW ringdown (1.06) |
| α > 1 | **Super-linear** | Protein folding (7.2), JWST galaxies (1.34) |

Visual cortex represents a unique transport class: parallel processing that's more efficient than diffusion.

---

## Physical Interpretation

### The Efficiency Principle

In a random walk (diffusion), time scales as distance²:
$$\tau \propto L^{0.5}$$

In the visual system:
$$\tau \propto L^{0.30}$$

This means doubling RF size only increases latency by ~23% (not 41% as for diffusion).

### Information Flow

```
LGN → V1 → V2 → V3 → V4 → IT → PFC
 ↓     ↓     ↓     ↓     ↓    ↓
25ms  45ms  55ms  60ms  75ms 150ms

RF:  0.2°  0.8°  2°   3.5°  5.5°  35°
```

Each hierarchical step:
- Increases RF by ~2-3×
- Increases latency by ~15-30%
- Maintains α ≈ 0.30 scaling

---

## Files Included

```
visual_cortex_rtm/
├── analyze_visual_cortex_rtm.py    # Main script
├── requirements.txt                 # Dependencies
├── README.md                       # This file
└── output/
    ├── visual_cortex_rtm.png
    ├── visual_cortex_rtm.pdf
    └── visual_cortex_data.csv
```

---

## Extending the Analysis

### Add More Areas

Edit `get_visual_cortex_data()`:

```python
rf_data = [
    # ... existing ...
    ("NewArea", rf_deg, rf_std, latency_ms, latency_std, n_studies, level),
]
```

### Eccentricity Dependence

RF size varies with eccentricity. For a full analysis, create separate datasets for:
- Foveal (0-2°)
- Parafoveal (2-5°)
- Peripheral (>10°)

The scaling exponent may differ by eccentricity.

---

## Implications

### For Neuroscience

1. Visual processing is highly optimized
2. Parallel and hierarchical coding work together
3. Sub-diffusive scaling may be a design principle

### For AI/Computer Vision

1. Biological inspiration for efficient architectures
2. Parallel processing at each level is key
3. Hierarchical structure enables scaling

### For RTM

1. Visual cortex represents a new transport class
2. Sub-diffusive scaling bridges diffusive and ballistic
3. Parallel processing can achieve α < 0.5

---

## References

### Key Papers

- Hubel & Wiesel (1962). Receptive fields in striate cortex.
- Felleman & Van Essen (1991). Distributed hierarchical processing.
- Schmolesky et al. (1998). Signal timing across macaque visual system.
- Harvey & Dumoulin (2011). CMF and pRF in human visual cortex.

### RTM Framework

- RTM Papers 001-020 (internal)

---

## Citation

```bibtex
@misc{rtm_visual_cortex_2026,
  author       = {RTM Research},
  title        = {RTM Visual Cortex: Temporal-Spatial Scaling},
  year         = {2026},
  note         = {n=21 visual areas, α=0.30, sub-diffusive transport}
}
```

---

## License

CC BY 4.0. Neurophysiology data compiled from published literature.
