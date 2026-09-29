# RTM Simulation: Vascular Tree (Fractal Lattice)

## Overview

This simulation verifies the **fractal vascular regime** in the RTM framework using a synthetic 3D vascular network.

The network mimics biological vasculature with Murray-style branching:
- Hierarchical tree structure
- Geometrically decreasing segment lengths
- Random 3D angular directions

## Expected Results

| Parameter | RTM Prediction | Paper | This Simulation |
|-----------|---------------|-------|-----------------|
| α | 2.4 – 2.6 | 2.5 ± 0.03 | 2.39 ± 0.16 |
| R² | — | — | 0.987 |
| Status | — | — | ✓ CONFIRMED |

## Files

```
07_vascular_tree/
├── vascular_tree.py          # Main simulation
├── vascular_tree.ipynb       # Interactive notebook
├── requirements.txt          # Dependencies
├── Dockerfile                # Container
├── README.md                 # This file
└── output/
    ├── vascular_tree_data.csv
    ├── vascular_tree_fit_results.csv
    ├── vascular_tree_summary.txt
    ├── vascular_tree_results.png
    ├── vascular_tree_results.pdf
    └── vascular_tree_structure.png
```

## Quick Start

```bash
pip install -r requirements.txt
python vascular_tree.py
```

Or with Docker:
```bash
docker build -t rtm-vascular-tree .
docker run --rm -v $(pwd)/output:/app/output rtm-vascular-tree
```

## Theory

### Murray's Law

Biological vascular networks follow Murray's law, which states that the cube of the parent vessel radius equals the sum of cubes of daughter radii:

$$r_0^3 = r_1^3 + r_2^3 + ... + r_n^3$$

This minimizes the power required for blood flow.

### Network Structure

Our synthetic network:
1. Starts with root node (generation 0)
2. Each node branches into 3 children
3. Segment lengths scale as L_d = L_0 × s^d (s = 0.7)
4. Angular directions are randomized isotropically

### Why α ≈ 2.5?

The fractal hierarchy creates bottlenecks:
- Random walker must navigate through branching structure
- Probability of finding a leaf decreases with depth
- Transport is slower than diffusion (α = 2) but optimized

This α ≈ 2.5 is characteristic of:
- Vascular systems (arteries → capillaries)
- Bronchial trees (bronchi → alveoli)
- Neural dendrites (soma → synapses)

## RTM Context

| Regime | α | Structure |
|--------|---|-----------|
| Diffusive | 2.0 | Regular lattice |
| Sierpiński | 2.3 | Self-similar fractal |
| **Vascular** | **2.5** | **Hierarchical tree** |
| Hierarchical SW | 2.6 | Modular bottlenecks |
| Holographic | 3.0 | Long-range decay |

## Biological Relevance

The α ≈ 2.5 value explains why:
- Blood circulation time scales with organism size^0.25 (Kleiber's law)
- Neural processing time increases with brain complexity
- Metabolic rate scales allometrically

## Citation

```
RTM Corpus (2025). Temporal Relativity in Multiscale Systems.
License: CC BY 4.0
```
