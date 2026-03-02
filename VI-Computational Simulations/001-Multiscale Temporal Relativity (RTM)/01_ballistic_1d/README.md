# RTM Simulation: Ballistic 1-D Propagation

## Overview

This simulation verifies the **ballistic transport regime** in the RTM (Relatividad Temporal Multiescala) framework.

In ballistic transport, a particle moves at constant velocity without scattering. The characteristic time T to traverse a distance L scales as:

$$T \propto L^\alpha \quad \text{with} \quad \alpha = 1$$

This is the **lower benchmark** for RTM temporal-relativity tests.

## Expected Results

| Parameter | Expected Value | This Simulation |
|-----------|---------------|-----------------|
| α (exponent) | 1.00 | 1.0000 ± 0.0001 |
| R² | ~1.0 | 1.000000 |
| Status | — | ✓ CONFIRMED |

## Files

```
01_ballistic_1d/
├── ballistic_1d.py       # Main simulation script
├── ballistic_1d.ipynb    # Interactive Jupyter notebook
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container for reproducibility
├── README.md             # This file
└── output/
    ├── ballistic_1d_data.csv           # Raw simulation data
    ├── ballistic_1d_fit_results.csv    # Fitting results
    ├── ballistic_1d_summary.txt        # Human-readable summary
    ├── ballistic_1d_results.png        # Visualization
    └── ballistic_1d_results.pdf        # Publication-quality figure
```

## Quick Start

### Option 1: Direct Python Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python ballistic_1d.py
```

### Option 2: Jupyter Notebook

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook ballistic_1d.ipynb
```

### Option 3: Docker (Recommended for Reproducibility)

```bash
# Build the container
docker build -t rtm-ballistic-1d .

# Run the simulation
docker run --rm -v $(pwd)/output:/app/output rtm-ballistic-1d

# Or run interactively
docker run -it --rm rtm-ballistic-1d bash
```

## Theory

### Ballistic Transport

In the ballistic regime, particles move in straight lines at constant velocity:

$$T = \frac{L}{v}$$

where:
- T = traversal time
- L = system size (distance)
- v = constant velocity

This gives the scaling relation:

$$T \propto L^1$$

### RTM Context

The RTM framework predicts different α values for different transport regimes:

| Regime | α Value | Physical Mechanism |
|--------|---------|-------------------|
| **Ballistic** | **≈ 1.0** | **Straight-line propagation** |
| Diffusive | ≈ 2.0 | Random walk (Brownian motion) |
| Fractal | ≈ 2.5 | Self-similar hierarchical structure |
| Quantum-confined | ≈ 3.5 | Quantum coherence effects |

This simulation confirms the ballistic baseline.

## Methodology

1. **System sizes**: L = 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
2. **Trials per size**: 100 independent runs
3. **Velocity model**: Constant base velocity with 1% Gaussian noise
4. **Fitting**: Linear regression in log-log space
5. **Uncertainty**: Bootstrap confidence intervals (1000 resamples)

## Results Interpretation

The fitted exponent α ≈ 1.00 confirms that:

1. The simulation correctly implements ballistic transport
2. Time scales linearly with distance
3. This establishes the lower bound for α in the RTM framework

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
