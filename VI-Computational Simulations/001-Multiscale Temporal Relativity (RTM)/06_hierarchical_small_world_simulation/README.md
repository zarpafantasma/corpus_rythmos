# RTM Simulation: Hierarchical Small-World Network (Baseline Neural-Like)

## Overview

This simulation verifies the **hierarchical modular small-world regime** in the RTM (Relatividad Temporal Multiescala) framework.

The network mimics cortical-type neural organization with:
- **Base modules**: Complete graphs (K8) representing local circuits
- **Tree hierarchy**: Modules connected via hub nodes
- **Bottlenecks**: Sparse inter-module connections that slow transport

The Mean First-Passage Time (MFPT) scales as:

$$T \propto L^\alpha \quad \text{where} \quad L = \sqrt{N} \quad \text{and} \quad \alpha \approx 2.5-2.6$$

## Expected Results

| Parameter | Expected Value | This Simulation |
|-----------|---------------|-----------------|
| α (exponent) | 2.5 – 2.6 | 2.6684 ± 0.0806 |
| Paper reported | 2.56 | — |
| R² | ~1.0 | 0.997273 |
| Status | — | ✓ CONFIRMED |

## Files

```
04_hierarchical_small_world/
├── hierarchical_small_world.py       # Main simulation script
├── hierarchical_small_world.ipynb    # Interactive Jupyter notebook
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container for reproducibility
├── README.md                         # This file
└── output/
    ├── hierarchical_small_world_data.csv           # Raw simulation data
    ├── hierarchical_small_world_fit_results.csv    # Fitting results
    ├── hierarchical_small_world_summary.txt        # Human-readable summary
    ├── hierarchical_small_world_results.png        # Visualization
    └── hierarchical_small_world_results.pdf        # Publication-quality figure
```

## Quick Start

### Option 1: Direct Python Execution

```bash
pip install -r requirements.txt
python hierarchical_small_world.py
```

### Option 2: Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook hierarchical_small_world.ipynb
```

### Option 3: Docker (Recommended)

```bash
docker build -t rtm-hierarchical-small-world .
docker run --rm -v $(pwd)/output:/app/output rtm-hierarchical-small-world
```

## Theory

### Network Structure

The hierarchical network is built recursively:

1. **Root module** (depth 0): Complete graph K8 with one designated hub
2. **Branching**: Each hub connects to 3 child modules
3. **Child modules**: Also K8 with their own hubs
4. **Inter-module edges**: Only between hub nodes

```
Level 0:     [K8]           (1 module, 8 nodes)
              |
Level 1:   [K8][K8][K8]     (3 modules, 24 nodes)
            |   |   |
Level 2:  ...  ...  ...     (9 modules, 72 nodes)
```

### Why Higher α?

Compared to flat small-world networks (α ≈ 2.1):

1. **Bottlenecks**: Inter-module connections only through hubs
2. **Hierarchy traversal**: Must climb up and down the tree
3. **Modular trapping**: Random walker gets "stuck" in modules

These factors increase the temporal cost of reaching distant targets.

### RTM Context

| Regime | α Value | Structure |
|--------|---------|-----------|
| Ballistic | ≈ 1.0 | Direct paths |
| Diffusive | ≈ 2.0 | Random walk |
| Flat Small-World | ≈ 2.0-2.1 | Shortcuts but no hierarchy |
| **Hierarchical SW** | **≈ 2.5-2.7** | **Modular bottlenecks** |
| Fractal | ≈ 2.5 | Self-similar structure |
| Quantum-confined | ≈ 3.5 | Coherence effects |

## Methodology

1. **Hierarchy depths**: 2, 3, 4, 5, 6 (excluding trivial depth=1)
2. **Module size**: 8 nodes (complete graph K8)
3. **Branching factor**: 3 children per hub
4. **Realizations per depth**: 8 independent networks
5. **Walks per network**: 30 random walks
6. **Target selection**: Single specific farthest node per walk

## Biological Relevance

This network models cortical organization:

- **Modules** ≈ Cortical columns or Brodmann areas
- **Hubs** ≈ Long-range projection neurons
- **K8 connectivity** ≈ Dense local recurrence
- **Sparse inter-module links** ≈ White matter tracts

The higher α explains why information processing across cortical areas takes longer than within areas.

## Comparison with Other Simulations

| Simulation | α | Comment |
|------------|---|---------|
| Ballistic 1-D | 1.00 | Lower bound |
| Diffusive 1-D | 1.97 | Random walk |
| Flat Small-World | 2.04 | Shortcuts |
| **Hierarchical SW** | **2.67** | **Bottlenecks (this sim)** |

The ~0.6 increase from flat (2.04) to hierarchical (2.67) quantifies the temporal cost of modular organization.

## Citation

```
RTM Corpus (2025). Temporal Relativity in Multiscale Systems.
License: CC BY 4.0
```

## License

CC BY 4.0 - Creative Commons Attribution 4.0 International
