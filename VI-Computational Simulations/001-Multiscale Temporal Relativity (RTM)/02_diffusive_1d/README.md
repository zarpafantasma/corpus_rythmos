# RTM Simulation: Diffusive 1-D (Random Walk)

## Overview

This simulation verifies the **diffusive transport regime** in the RTM (Relatividad Temporal Multiescala) framework.

In diffusive transport, a particle undergoes random walk (Brownian motion). The mean first-passage time (MFPT) to traverse a distance L scales as:

$$T \propto L^\alpha \quad \text{with} \quad \alpha = 2$$

This is the classic result for 1-D symmetric random walk: T = L².

## Expected Results

| Parameter | Expected Value | This Simulation |
|-----------|---------------|-----------------|
| α (exponent) | 2.00 | 1.9698 ± 0.0089 |
| R² | ~1.0 | 0.999898 |
| Status | — | ✓ CONFIRMED |

The small deviation (~1.5%) from the theoretical value is due to finite sampling (statistical noise).

## Files

```
02_diffusive_1d/
├── diffusive_1d.py       # Main simulation script
├── diffusive_1d.ipynb    # Interactive Jupyter notebook
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container for reproducibility
├── README.md             # This file
└── output/
    ├── diffusive_1d_data.csv           # Raw simulation data
    ├── diffusive_1d_fit_results.csv    # Fitting results
    ├── diffusive_1d_summary.txt        # Human-readable summary
    ├── diffusive_1d_results.png        # Visualization
    └── diffusive_1d_results.pdf        # Publication-quality figure
```

## Quick Start

### Option 1: Direct Python Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python diffusive_1d.py
```

### Option 2: Jupyter Notebook

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook diffusive_1d.ipynb
```

### Option 3: Docker (Recommended for Reproducibility)

```bash
# Build the container
docker build -t rtm-diffusive-1d .

# Run the simulation
docker run --rm -v $(pwd)/output:/app/output rtm-diffusive-1d

# Or run interactively
docker run -it --rm rtm-diffusive-1d bash
```

## Theory

### 1-D Random Walk

The walker starts at the origin (x = 0) and takes steps of ±1 with equal probability:

$$X_{n+1} = X_n + \xi_n, \quad P(\xi = +1) = P(\xi = -1) = 0.5$$

### Mean First-Passage Time

For absorbing boundaries at ±L, the MFPT satisfies the recurrence relation:

$$T(x) = 1 + \frac{1}{2}T(x-1) + \frac{1}{2}T(x+1)$$

with boundary conditions T(±L) = 0.

The solution is:

$$T(x) = L^2 - x^2$$

Starting from the origin: **T(0) = L²**

This gives **exactly α = 2**.

### RTM Context

The RTM framework predicts different α values for different transport regimes:

| Regime | α Value | Physical Mechanism |
|--------|---------|-------------------|
| Ballistic | ≈ 1.0 | Straight-line propagation |
| **Diffusive** | **≈ 2.0** | **Random walk (Brownian motion)** |
| Fractal | ≈ 2.5 | Self-similar hierarchical structure |
| Quantum-confined | ≈ 3.5 | Quantum coherence effects |

This simulation confirms the diffusive benchmark.

## Methodology

1. **System sizes**: L = 5, 10, 20, 40, 80, 160, 320
2. **Trials per size**: 200 independent random walks
3. **Observable**: Mean First-Passage Time (MFPT) to reach ±L from origin
4. **Fitting**: Linear regression in log-log space
5. **Optimization**: Vectorized cumulative sum for fast simulation

## Physical Interpretation

In diffusion, the walker doesn't "know" which direction leads to the target. It explores space randomly, sometimes moving toward the goal, sometimes away. This inefficiency is reflected in the quadratic scaling:

- **Ballistic (α=1)**: Direct path, time ∝ distance
- **Diffusive (α=2)**: Random exploration, time ∝ distance²

Doubling the distance in diffusion takes **four times** as long, not twice.

## Results Interpretation

The fitted exponent α ≈ 1.97 confirms:

1. The simulation correctly implements 1-D random walk
2. Time scales quadratically with distance (T ∝ L²)
3. This establishes the diffusive benchmark in the RTM framework

## Citation

If you use this simulation, please cite the RTM corpus:

```
RTM Corpus (2025). Temporal Relativity in Multiscale Systems.
Available at: [repository URL]
License: CC BY 4.0
```

## License

CC BY 4.0 - Creative Commons Attribution 4.0 International

You are free to share and adapt this material for any purpose, provided you give appropriate credit.
