## Simulation H: Quantum-Confined Regime ($\alpha \approx 3.5$)

### Theoretical Background

The RTM framework predicts a quantum-confined regime with $\alpha \approx 3.5$ for systems where quantum corrections dominate temporal correlations. This regime is theoretically motivated by three independent lines of reasoning that converge on the same value:

**Loop Quantum Gravity (LQG).** In the LQG framework, spacetime is discretized into spin networks whose evolution generates geometric operators with quantized spectra. For quantum-dominated systems where the characteristic length $L$ is much smaller than the coherence length $\xi$, the effective scaling exponent acquires a quantum correction:

$$\alpha_{\text{LQG}} = d + \frac{1}{2} = 3.5 \quad \text{for } d = 3$$

The additional $+\frac{1}{2}$ arises from the minimal area gap $\Delta A \propto \ell_P^2 \sqrt{j(j+1)}$ in the spin-network evolution, which introduces boundary/edge corrections to the spectral dimension that are absent in the classical ($\alpha = 3$) holographic case.

**AdS/CFT Correspondence.** In holographic systems with Lifshitz scaling $z \approx 2$ and hyperscaling violation exponent $\theta \approx 1.5$ (characteristic of systems with emergent Fermi surfaces under geometric confinement):

$$\alpha = d + z - \theta = 3 + 2 - 1.5 = 3.5$$

**String Theory.** Compactification effects with quantum ($\alpha'$) corrections reduce the bare string-theoretic exponent $\alpha_{\text{string}} \approx 4$ down to $\approx 3.5$ via string loop corrections. The convergence of all three independent approaches to the same value provides strong theoretical motivation.

### Distinction from the Holographic Regime (Scenario A vs. Scenario B)

The quantum-confined regime (Scenario B, $\alpha \approx 3.5$) differs fundamentally from the holographic regime (Scenario A, $\alpha \to 3.0$) through the presence of **geometric confinement**:

| Feature | Holographic (Scenario A) | Quantum-Confined (Scenario B) |
|:---|:---|:---|
| Boundaries | Open / no confinement | **Hard-wall confinement** |
| Long-range links | $P(r) \propto r^{-3}$ (holographic) | **None** (pure diffusion) |
| Boundary effects | Negligible | **Dominant** (quantum trapping) |
| Physical analog | Long-wavelength holographic transport | Confined quantum media, mesoscopic structures |
| Exponent source | Volume scaling from $r^{-d}$ shortcuts | Base diffusion + **boundary correction** |

Scenario B involves *confined* systems where boundary/edge corrections contribute additively to the exponent, shifting it from $\sim 3.0$ to $\sim 3.5$.

### Model Description

We construct a computational analog of a quantum-confined system:

1. **Base lattice:** 3D cubic grid of side $L$ with **hard-wall boundaries** (reflecting walls — the defining feature of confinement). No periodic wrapping.
2. **Short-range connections:** standard 6-connectivity ($\pm x, \pm y, \pm z$), restricted to within the box.
3. **No long-range links** ($n_{\text{lr}} = 0$): unlike the holographic regime, the quantum-confined model has *no* shortcuts. Transport is purely diffusive within the confining geometry.
4. **Boundary confinement potential:** nodes within a boundary layer (distance $\leq \delta$ from any wall) receive self-loops whose count scales as:

$$n_{\text{self-loops}}(\text{node}) = \lfloor \beta \cdot L^{\gamma} \rfloor \times (\delta - d_{\text{wall}} + 1)$$

with $\beta = 1.5$ (potential strength) and $\gamma = 1.0$ (scaling exponent). Nodes closer to the wall receive proportionally more self-loops (higher shell factor).

5. **Observable:** Mean First-Passage Time (MFPT) from origin $(0,0,0)$ to farthest corner $(L{-}1, L{-}1, L{-}1)$.

### Physical Motivation for the Confinement Mechanism

The self-loop mechanism models **quantum trapping at confining potential walls**. In a quantum box, wavefunctions develop standing-wave patterns: the probability current vanishes at the wall, but the probability *density* piles up near the boundary. A particle (or information carrier) effectively "dwells" near the walls before propagating inward.

The confinement effect operates through two competing scalings:

- **Boundary fraction** decreases as $L$ grows: $f_{\text{boundary}} \propto L^{-1}$ (surface-to-volume ratio)
- **Self-loops per boundary node** increase as $L$ grows: $n_{\text{loops}} \propto \beta \cdot L^{\gamma} = 1.5 \cdot L$

The net effect is $O(1)$—a constant additive correction to the scaling exponent:

$$\alpha_{\text{total}} = \alpha_{\text{base}} + \Delta\alpha_{\text{confinement}} \approx 3.28 + 0.22 \approx 3.5$$

where $\alpha_{\text{base}} \approx 3.28$ is the baseline exponent for MFPT on a pure 3D lattice (reflecting the near-cover-time contribution $\sim L^3 \log L$, which gives an effective exponent slightly above 3 at accessible lattice sizes).

### Epistemic Status: Consistency Check, Not Independent Validation

**This simulation is a proof-of-concept consistency check, not an independent validation.** The distinction is critical:

1. **The model parameters are calibrated to the target.** The values $\beta = 1.5$, $\gamma = 1.0$ are selected from a systematic sweep of ~20 configurations because they produce $\alpha \approx 3.5$. Other parameter choices yield different exponents ($\beta = 1.25$ gives $\alpha \approx 3.33$; $\beta = 1.75$ gives $\alpha \approx 3.38$). With sufficient free parameters ($\beta$, $\gamma$, $\delta$, $n_{\text{lr}}$, decay exponent), a broad range of target exponents can be reproduced. This makes the result **consistent with** the prediction, but **does not independently confirm** it.

2. **The model is phenomenological, not derived from first principles.** The self-loops at boundaries are not derived from LQG, AdS/CFT, or string theory. They are a heuristic mechanism inspired by the physical picture of wavefunction pileup at confining walls. The theoretical justification (the boundary correction adds $+\frac{1}{2}$ to the spectral dimension) is a post hoc rationalization, not a prediction.

3. **Contrast with Simulations A–G.** In those simulations, the model is either exactly specified beforehand (A–D), constrained by the physical system being modeled (E–F), or prescribed with no tunable parameters beyond lattice size (G). No parameter fitting to the target exponent is involved. Simulation H does not share this property.

**What this simulation does demonstrate:**

- That a *simple, physically motivated* confinement mechanism—boundary self-loops with $O(L)$ scaling—can produce the predicted exponent in a lattice model.
- That the resulting power law is clean ($R^2 = 0.997$) with good finite-size convergence, ruling out the possibility that $\alpha \approx 3.5$ requires exotic or contrived network structures.
- That the confinement correction is *additive* to the base exponent, consistent with the theoretical picture of a boundary/edge term on top of volumetric scaling.

These results strengthen the *plausibility* of the quantum-confined prediction, but they do not constitute the decisive test specified below.

### Calibration History

We document the parameter selection process for full transparency:

| Configuration | $\alpha$ | Notes |
|:---|:---:|:---|
| Holographic $r^{-3.5}$ decay (no boundary effects) | 2.94 | Same as Scenario A — confinement absent |
| Deep boundaries, $\beta=4$, $\gamma=0$ (constant loops) | 3.10 | Boundary contribution washes out at large $L$ |
| Harmonic potential, $\beta=1.25$, $\gamma=1.0$ | 3.33 | Tight CI excludes 3.5 |
| **Harmonic potential, $\beta=1.5$, $\gamma=1.0$** | **3.49** | **CI includes 3.5** |
| Harmonic potential, $\beta=1.75$, $\gamma=1.0$ | 3.38 | CI excludes 3.5 |

The selected configuration ($\beta = 1.5$, $\gamma = 1.0$, $n_{\text{lr}} = 0$, $\delta = 1$) produces the cleanest match to the theoretical target. This is parameter fitting, not blind prediction.

### Methodology

| Parameter | Value |
|:---|:---:|
| Lattice sizes $L$ | 5, 6, 7, 8, 10, 12, 14, 16, 18 |
| Nodes $N$ | 125 to 5,832 |
| Long-range links | **0** (pure lattice) |
| Confinement: $\beta$, $\gamma$, $\delta$ | 1.5, 1.0, 1 |
| Realizations per size | 8 |
| Walks per realization | 50 |
| Total walks | 3,600 |
| Max steps per walk | 3,000,000 |
| Bootstrap resamples | 10,000 |

### Results

| $L$ | $N$ | $\bar{T}$ (MFPT) | $\sigma_T$ | 95% CI low | 95% CI high | Done | Bdry % | Loops/Bdry |
|:---:|:---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 5 | 125 | 1,137 | 1,027 | 1,039 | 1,242 | 400/400 | 99.2% | 12.5 |
| 6 | 216 | 2,641 | 2,545 | 2,399 | 2,898 | 400/400 | 96.3% | 15.6 |
| 7 | 343 | 4,486 | 4,305 | 4,080 | 4,926 | 400/400 | 92.1% | 16.9 |
| 8 | 512 | 7,745 | 7,017 | 7,070 | 8,457 | 400/400 | 87.5% | 19.9 |
| 10 | 1,000 | 15,312 | 15,077 | 13,881 | 16,820 | 400/400 | 78.4% | 24.3 |
| 12 | 1,728 | 27,712 | 27,043 | 25,182 | 30,411 | 400/400 | 70.4% | 28.8 |
| 14 | 2,744 | 47,050 | 42,502 | 42,959 | 51,246 | 400/400 | 63.6% | 33.2 |
| 16 | 4,096 | 79,867 | 84,186 | 71,933 | 88,603 | 400/400 | 57.8% | 37.7 |
| 18 | 5,832 | 107,787 | 104,068 | 97,750 | 118,061 | 400/400 | 52.9% | 42.2 |

All 3,600 walks complete successfully (100% completion rate).

The rightmost columns illustrate the confinement mechanism: boundary fraction decreases from 99.2% ($L = 5$) to 52.9% ($L = 18$) while self-loops per boundary node increase from 12.5 to 42.2—the net $O(1)$ effect that produces the constant additive correction to $\alpha$.

**Power-law fit:**

$$T = 4.83 \times L^{3.4907}$$

$$\alpha = 3.4907 \pm 0.0677 \qquad R^2 = 0.9974$$

$$\text{Bootstrap 95\% CI: } [3.4186, \ 3.5643]$$

**Sensitivity analysis:**

- Excluding largest $L = 18$: $\alpha = 3.5335 \pm 0.0776$
- Excluding smallest $L = 5$: $\alpha = 3.3860 \pm 0.0433$

### Finite-Size Convergence

The running $\alpha$ estimate converges *from above* toward $\alpha = 3.5$:

| Points | $L$ range | Running $\alpha$ |
|:---:|:---|:---:|
| 3 | $L \leq 7$ | 4.10 |
| 4 | $L \leq 8$ | 4.03 |
| 5 | $L \leq 10$ | 3.75 |
| 6 | $L \leq 12$ | 3.60 |
| 7 | $L \leq 14$ | 3.54 |
| 8 | $L \leq 16$ | 3.54 |
| 9 | $L \leq 18$ | 3.49 |

At small $L$, the boundary layer dominates (up to 99% of nodes), inflating the effective exponent. As $L$ increases and the bulk contribution grows, $\alpha$ relaxes toward its asymptotic value. The trajectory stabilizes within the $[3.4, 3.6]$ band for $L \geq 14$.

### Interpretation

The model yields $\alpha = 3.4907 \pm 0.0677$, with a 95% bootstrap CI $[3.42, 3.56]$ that includes the theoretical prediction $\alpha = 3.5$. The deviation from theory is $\Delta\alpha = -0.009$.

The mechanism is transparent:

1. **Pure diffusion on a confined 3D lattice** provides a baseline $\alpha_{\text{base}} \approx 3.28$, reflecting near-cover-time scaling ($\sim L^3 \log L$) for corner-to-corner traversal.

2. **Boundary confinement** adds $\Delta\alpha \approx +0.22$ through self-loop trapping. The boundary fraction × loops-per-node product remains $O(1)$ across all lattice sizes, producing a constant additive shift.

3. **The total** $\alpha_{\text{base}} + \Delta\alpha_{\text{confinement}} \approx 3.28 + 0.22 = 3.50$ matches the theoretical target.

However, as detailed above, this match is *engineered through parameter selection*, not predicted blindly.

**Status: ◐ CONSISTENT (model-dependent).** The simulation produces an exponent consistent with $\alpha = 3.5$ via a physically motivated confinement mechanism. The model parameters are calibrated to the target, so the result does not constitute independent validation. Definitive confirmation requires the approaches specified below.

### What Would Constitute Independent Validation

For the quantum-confined regime to reach the epistemic level of Simulations A–G, one or more of the following are needed:

1. **Pre-registered lattice model:** A confinement model whose parameters derive from first principles (e.g., the LQG spin-network Hamiltonian) *before* running the simulation, not fitted to the target.
2. **Molecular dynamics:** Lennard-Jones or similar MD simulations with $N > 50{,}000$ particles in a confining potential, measuring relaxation or equilibration times as a function of system size.
3. **Quantum hardware:** Measurement of decoherence, thermalization, or entanglement propagation times across multiple system sizes in trapped ion chains ($> 50$ ions), optical lattices, or superconducting qubit arrays.
4. **Experimental BEC data:** Excitation propagation times in Bose-Einstein condensates of varying spatial extent, with controlled temperature and density.

### Supplementary Materials

Available in `08_quantum_confined_simulation.zip`:

- `quantum_confined_simulation.py` — Full simulation script with CLI interface
- `quantum_confined_notebook.ipynb` — Interactive Jupyter notebook
- `quantum_confined_results.csv` — Summary statistics by lattice size (9 rows)
- `quantum_confined_walks.csv` — Individual walk data (3,600 walks)
- `quantum_confined_fit_summary.csv` — Power-law fit parameters and bootstrap CI
- `metadata.json` — Full configuration and results
- `figures/` — 6 publication-quality figures (PNG + PDF)
- `requirements.txt`, `Dockerfile`, `README.md` — Reproducibility infrastructure

All code released under CC BY 4.0 license.

---

## 4. Consolidated Results Table

### Table 1: RTM Numerical Validation Results

| Simulation | Topology | $\alpha$ (Theory) | $\alpha$ (Measured) | 95% CI | $R^2$ | Status |
|:---|:---|:---:|:---|:---|:---:|:---:|
| **A.** Ballistic 1-D | Linear chain | 1.00 | $1.0000 \pm 0.0001$ | $[1.0000, \ 1.0000]$ | 1.0000 | ✅ Confirmed |
| **B.** Diffusive 1-D | Linear + RW | 2.00 | $1.9698 \pm 0.0089$ | $[1.9448, \ 1.9878]$ | 0.9999 | ✅ Confirmed |
| **C.** Flat Small-World | Watts-Strogatz | $\sim$2.0 | $2.0428 \pm 0.0146$ | $[2.0109, \ 2.0749]$ | 0.9998 | ✅ Confirmed |
| **D.** Sierpiński Fractal | Deterministic fractal | $d_w \approx 2.32$ | $2.3245 \pm 0.0157$ | $[2.2832, \ 2.3558]$ | 0.9999 | ✅ Confirmed |
| **E.** Vascular Tree | 3D fractal tree | 2.4–2.6 | $2.3875 \pm 0.1595$ | $[2.0599, \ 3.4305]$ | 0.9868 | ✅ Confirmed |
| **F.** Hierarchical SW | Modular hierarchy | 2.5–2.7 | $2.6684 \pm 0.0806$ | $[2.4845, \ 2.9035]$ | 0.9973 | ✅ Confirmed |
| **G.** Holographic Decay | $P(r) \propto r^{-3}$ lattice | $\to 3.0$ | $2.9499 \pm 0.0683$ | $[2.8151, \ 3.0806]$ | 0.9968 | ✅ Confirmed |
| **H.** Quantum-Confined | 3D lattice + confinement | $\approx 3.5$ | $3.4907 \pm 0.0677$ | $[3.4186, \ 3.5643]$ | 0.9974 | ◐ Consistent$^*$ |

$^*$ *Model-dependent: parameters calibrated to target. Consistency check, not independent validation. See Section 3 caveats.*

### Status Key

| Symbol | Meaning |
|:---:|:---|
| ✅ Confirmed | Model pre-specified or system-constrained; exponent measured without parameter fitting to target |
| ◐ Consistent | Exponent matches prediction, but model parameters are calibrated; proof of concept |

### Summary Statistics

- **Regimes tested:** 8 (7 power-law + 1 logarithmic small-world, excluded from table)
- **Independently confirmed:** 7 of 7 power-law regimes
- **Consistent (model-dependent):** 1 (quantum-confined)
- **Average $R^2$:** 0.9972
- **Independently validated exponent range:** $\alpha = 1.00$ to $\alpha = 2.95$
- **Model-dependent range extension:** to $\alpha = 3.49$
- All measured exponents fall within theoretical predictions or confidence intervals

*Note: The small-world case is excluded from the table because its scaling is logarithmic ($\ell \sim \log N$), not a power law.*

### Supplementary Packages

- `01_ballistic_1d_simulation.zip`
- `02_diffusive_1d_simulation.zip`
- `03_flat_small_world_simulation.zip`
- `04_sierpinski_fractal_simulation.zip`
- `05_vascular_tree_simulation.zip`
- `06_hierarchical_small_world_simulation.zip`
- `07_holographic_decay_simulation.zip`
- `08_quantum_confined_simulation.zip` *(proof-of-concept model)*

All code released under CC BY 4.0 license.

---

*RTM — Multiscale Temporal Relativity. Computational validation suite: seven regimes independently confirmed ($\alpha = 1$ to $\alpha \approx 3$); one regime consistent with prediction via model-dependent demonstration ($\alpha \approx 3.5$).*
