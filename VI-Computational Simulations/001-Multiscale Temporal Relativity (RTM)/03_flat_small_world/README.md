# RTM Simulation: Flat Small-World Network (MFPT)

## Overview

This simulation verifies the **small-world network regime** in the RTM (Relatividad Temporal Multiescala) framework.

The Watts-Strogatz small-world model creates networks with:
- High clustering (like regular lattices)
- Short average path lengths (like random graphs)

For flat small-world networks, the Mean First-Passage Time (MFPT) scales as:

$$T \propto L^\alpha \quad \text{where} \quad L = \sqrt{N} \quad \text{and} \quad \alpha \approx 2.1$$

## Expected Results

| Parameter | Expected Value | This Simulation |
|-----------|---------------|-----------------|
| α (exponent) | ~2.1 | 2.0428 ± 0.0146 |
| R² | ~1.0 | 0.999797 |
| Status | — | ✓ CONFIRMED |

## Files

```
03_flat_small_world/
├── flat_small_world.py       # Main simulation script
├── flat_small_world.ipynb    # Interactive Jupyter notebook
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container for reproducibility
├── README.md                 # This file
└── output/
    ├── flat_small_world_data.csv           # Raw simulation data
    ├── flat_small_world_fit_results.csv    # Fitting results
    ├── flat_small_world_summary.txt        # Human-readable summary
    ├── flat_small_world_results.png        # Visualization
    └── flat_small_world_results.pdf        # Publication-quality figure
```

## Quick Start

### Option 1: Direct Python Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python flat_small_world.py
```

### Option 2: Jupyter Notebook

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook flat_small_world.ipynb
```

### Option 3: Docker (Recommended for Reproducibility)

```bash
# Build the container
docker build -t rtm-flat-small-world .

# Run the simulation
docker run --rm -v $(pwd)/output:/app/output rtm-flat-small-world
```

## Theory

### Watts-Strogatz Small-World Model

The model starts with a ring of N nodes, each connected to k nearest neighbors. Then, with probability p, each edge is "rewired" to a random node.

- **p = 0**: Regular ring lattice (high clustering, long paths)
- **p = 1**: Random graph (low clustering, short paths)
- **0 < p < 1**: Small-world (high clustering AND short paths)

### Small-World Effect

The "shortcuts" created by rewiring dramatically reduce average path lengths while preserving local clustering. This is why small-world networks are efficient for information transfer.

### MFPT Scaling

The RTM paper notes two perspectives:

1. **Logarithmic scaling**: Average path length scales as $\ell \propto \log(N)$
2. **Power-law scaling**: When using $L = \sqrt{N}$, MFPT scales as $T \propto L^\alpha$ with $\alpha \approx 2.0-2.1$

The power-law interpretation with $\alpha \approx 2$ indicates that small-world networks sit between:
- Ballistic (α ≈ 1): Direct paths
- Diffusive (α ≈ 2): Random exploration

### RTM Context

| Regime | α Value | Physical Mechanism |
|--------|---------|-------------------|
| Ballistic | ≈ 1.0 | Straight-line propagation |
| Diffusive | ≈ 2.0 | Random walk (Brownian motion) |
| **Small-World** | **≈ 2.0-2.1** | **Shortcuts but random exploration** |
| Fractal | ≈ 2.5 | Self-similar hierarchical structure |
| Quantum-confined | ≈ 3.5 | Quantum coherence effects |

## Methodology

1. **Network sizes**: N = 100, 200, 400, 800, 1600, 3200
2. **Watts-Strogatz parameters**: k = 6, p = 0.1
3. **Realizations per size**: 5 independent networks
4. **Pairs per network**: 50 random source-target pairs
5. **Walks per pair**: 10 random walk trials
6. **Observable**: Mean First-Passage Time (MFPT)

## Physical Interpretation

In small-world networks, random walkers:
- Benefit from shortcuts that reduce average path lengths
- But don't "know" which edges are shortcuts
- Still explore the network somewhat randomly

This explains why α ≈ 2 (near diffusive) rather than α ≈ 1 (ballistic): the shortcuts exist but aren't exploited optimally by random walks.

## Results Interpretation

The fitted exponent α ≈ 2.04 confirms:

1. Small-world topology accelerates transport vs. regular lattices
2. But random walks remain near-diffusive (α ≈ 2)
3. This matches RTM predictions for neural-like networks

## Citation

If you use this simulation, please cite the RTM corpus:

```
RTM Corpus (2025). Temporal Relativity in Multiscale Systems.
Available at: [repository URL]
License: CC BY 4.0
```

## License

CC BY 4.0 - Creative Commons Attribution 4.0 International
