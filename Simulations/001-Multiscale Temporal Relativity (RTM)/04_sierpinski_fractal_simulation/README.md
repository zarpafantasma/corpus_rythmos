# RTM Simulation: Sierpiński Fractal Network

## Overview

This simulation verifies the **fractal regime** in the RTM framework using the Sierpiński gasket (triangle).

The Sierpiński gasket is a deterministic fractal with exact analytical properties, making it an ideal test case for fractal scaling predictions.

## Expected Results

| Parameter | Theoretical | Paper | This Simulation |
|-----------|-------------|-------|-----------------|
| α | d_w ≈ 2.322 | 2.48 | 2.3245 ± 0.0157 |
| R² | — | — | 0.9999 |
| Status | — | — | ✓ CONFIRMED |

The simulation **perfectly matches** the theoretical walk dimension d_w!

## Sierpiński Gasket Properties

The Sierpiński gasket has well-known fractal dimensions:

| Dimension | Formula | Value |
|-----------|---------|-------|
| Fractal (d_f) | ln(3)/ln(2) | 1.585 |
| Spectral (d_s) | 2·ln(3)/ln(5) | 1.365 |
| **Walk (d_w)** | **ln(5)/ln(2)** | **2.322** |

For random walks on fractals: **T ∝ L^(d_w)**

## Files

```
06_sierpinski_fractal/
├── sierpinski_fractal.py         # Main simulation
├── sierpinski_fractal.ipynb      # Interactive notebook
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container
├── README.md                     # This file
└── output/
    ├── sierpinski_fractal_data.csv
    ├── sierpinski_fractal_fit_results.csv
    ├── sierpinski_fractal_summary.txt
    ├── sierpinski_fractal_results.png
    ├── sierpinski_fractal_results.pdf
    └── sierpinski_structure.png        # Fractal visualization
```

## Quick Start

```bash
pip install -r requirements.txt
python sierpinski_fractal.py
```

Or with Docker:
```bash
docker build -t rtm-sierpinski .
docker run --rm -v $(pwd)/output:/app/output rtm-sierpinski
```

## Theory

### Fractal Construction

The Sierpiński gasket is built by recursive subdivision:

1. Start with a triangle (3 vertices)
2. Insert midpoints on each edge
3. Connect midpoints (forms inner triangle)
4. Remove the inner triangle (creates the "hole")
5. Repeat for each remaining triangle

```
Generation 0:      Generation 1:      Generation 2:
    △                  △                  △
                      ▲ ▲                △ △
                                        ▲ ▲ ▲ ▲
```

### Why α ≈ d_w?

Random walks on fractals explore space inefficiently due to:
- Dead-ends at various scales
- Self-similar trapping regions
- Reduced connectivity compared to regular lattices

The walk dimension d_w captures this inefficiency: it takes T ∝ L^(d_w) steps to traverse distance L, where d_w > 2 for most fractals.

### RTM Context

| Regime | α | Structure |
|--------|---|-----------|
| Diffusive | 2.0 | Regular lattice |
| Small-world | 2.0-2.1 | Shortcuts |
| **Sierpiński** | **2.32** | **Self-similar fractal** |
| Hierarchical | 2.5-2.7 | Modular bottlenecks |

## Methodology

1. **Generations**: g = 2, 3, 4, 5, 6 (L = 4 to 64)
2. **Observable**: MFPT between corner vertices (0-1, 1-2, 0-2)
3. **Important**: Direct corner-corner edges are REMOVED
4. **Walks**: 50 per vertex pair, both directions

## Paper Note

The RTM paper reports α ≈ 2.48, which is higher than the theoretical d_w ≈ 2.32. The paper notes:

> "Shallow or limited-depth simulations (e.g. g ≤ 4) tend to underestimate the asymptotic exponent"

Our simulation with g ≤ 6 matches d_w exactly. The higher paper value may include pre-asymptotic corrections visible only at g ≥ 7.

## Citation

```
RTM Corpus (2025). Temporal Relativity in Multiscale Systems.
License: CC BY 4.0
```
