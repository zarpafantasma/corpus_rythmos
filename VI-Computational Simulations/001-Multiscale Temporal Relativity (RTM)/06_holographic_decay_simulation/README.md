# RTM Simulation G: Holographic Decay Network P(r) ∝ r⁻³

## Multiscale Temporal Relativity (RTM) — Computational Validation

---

### Overview

This package contains the complete, reproducible simulation for the **Holographic Decay** regime of the RTM framework. The simulation validates the theoretical prediction that temporal scaling in holographic-inspired networks converges to **α → 3.0**, where the characteristic time *T* scales as *T ∝ L^α* with system size *L*.

### Theoretical Background

Holographic-inspired networks feature long-range connections with probability decaying as the inverse cube of distance: **P(r) ∝ r⁻³**. This decay law, motivated by holographic principles in theoretical physics, creates networks where transport time scales with the **volume** (L³) rather than the surface area or linear extent.

### Model Description

| Parameter | Value |
|---|---|
| Base lattice | 3D cubic grid of side L |
| Short-range connections | Standard 6-connectivity (±x, ±y, ±z) |
| Long-range links | 2 per node, P(r) ∝ r⁻³ |
| Observable | MFPT from origin (0,0,0) to farthest corner |
| Lattice sizes | L = 6, 8, 10, 12, 14, 16, 18, 20 |
| Nodes range | 216 to 8,000 |

### Results

```
Power-Law Fit: T = 2.19 × L^α
  α = 2.9499 ± 0.0683
  R² = 0.996791
  Bootstrap 95% CI: [2.8151, 3.0806]

Status: 95% CI INCLUDES theoretical value α = 3.0 ✓
```

**Comparison with previous simulation:**

| Metric | Previous (Paper) | Current |
|---|---|---|
| α | 3.1586 ± 0.2260 | 2.9499 ± 0.0683 |
| R² | 0.9799 | 0.9968 |
| 95% CI | [2.2568, 3.7273] | [2.8151, 3.0806] |
| Lattice range | L = 6–16 | L = 6–20 |
| Max N | 4,096 | 8,000 |

### Files

| File | Description |
|---|---|
| `holographic_decay_simulation.py` | Main simulation script |
| `holographic_decay_notebook.ipynb` | Interactive Jupyter notebook |
| `holographic_decay_results.csv` | Summary results by lattice size |
| `holographic_decay_walks.csv` | Individual walk data |
| `holographic_decay_fit_summary.csv` | Power-law fit parameters |
| `metadata.json` | Simulation metadata and configuration |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker container for reproducible execution |
| `figures/` | Publication-quality figures |

### Quick Start

#### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python holographic_decay_simulation.py --output-dir ./output

# Or with custom parameters
python holographic_decay_simulation.py \
  --lattice-sizes 6 8 10 12 14 16 18 20 \
  --realizations 5 \
  --walks 35 \
  --max-steps 1500000 \
  --seed 42
```

#### Docker Execution

```bash
# Build
docker build -t rtm-holographic-decay .

# Run simulation
docker run --rm -v $(pwd)/output:/app/output rtm-holographic-decay

# Run Jupyter notebook
docker run --rm -p 8888:8888 rtm-holographic-decay \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### License

CC BY 4.0
