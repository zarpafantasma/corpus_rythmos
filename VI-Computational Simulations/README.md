# VI - Computational Simulations

This folder contains **computational validations** of the RTM theoretical framework. Each simulation tests specific predictions from Papers 001–017, providing reproducible evidence that the mathematics produce consistent, physically meaningful results.

---

## ⚠️ Nature of This Section

This folder focuses exclusively on **Papers 001, 016, and 017**, the three papers whose validations are necessarily *computational* rather than *empirical*.

**Why these three?**

| Paper | Topic | Why Computational |
|-------|-------|-------------------|
| **001** | Core RTM scaling law (T ∝ L^α) | Tests mathematical emergence of α from network topology, no external data needed |
| **016** | Aetherion vacuum propulsion | No experimental apparatus exists to test vacuum gradient effects |
| **017** | RTM Unified Field Framework | QFT formalism, AdS/CFT holography, validates theoretical consistency |

**Papers 003–015** have their validations in **Folder VII (Empirical and Heuristic Validations)** because those domains have real-world datasets against which RTM predictions can be tested: visual cortex latencies, gravitational wave catalogs, cardiac HRV, hurricane intensification, market crashes, etc.

**The critical distinction:**
- **Folder VI (here):** Does the math work internally? Are thermodynamic laws preserved? Do numerical solutions converge?
- **Folder VII:** Does the math match reality? Do observed scaling exponents match RTM predictions?

Papers 001, 016, and 017 can only answer the first question. Empirical validation of vacuum propulsion, branch-jumping, and unified field effects awaits future experiments that do not yet exist.

---

## Structure

Each paper's simulations include:
- **Python scripts** (`.py`) — standalone execution
- **Jupyter notebooks** (`.ipynb`) — interactive exploration
- **Dockerfiles** — reproducible containerized execution
- **Output folders** — CSV data, PNG/PDF figures, summary text files
- **README files** — theoretical context and results interpretation
- **Red Team audits** (where applicable) — adversarial testing for thermodynamic violations

---

## 001 - Multiscale Temporal Relativity (RTM)

**Purpose:** Validate the core scaling law T ∝ L^α across different transport regimes.

| Simulation | Target α | Description | Result |
|------------|----------|-------------|--------|
| `01_ballistic_1d` | 1.0 | Constant velocity propagation (lower bound) | α = 1.0000 ± 0.0001 ✓ |
| `02_diffusive_1d` | 2.0 | Brownian motion / random walk | α = 2.0000 ± 0.0002 ✓ |
| `03_flat_small_world` | ~1.0 | Watts-Strogatz small-world network | Confirms flat topology |
| `04_sierpinski_fractal` | ~2.58 | Sierpiński gasket (d_f = log3/log2) | α ≈ 2.58 ✓ |
| `05_vascular_tree` | ~2.3 | Murray's law branching network | Matches biological scaling |
| `06_hierarchical_small_world` | ~2.0 | Hierarchical modular network | α ≈ 2.0 ✓ |
| `07_holographic_decay` | 3.0 | P(r) ∝ r⁻³ long-range connections | α = 2.95 ± 0.07, 95% CI includes 3.0 ✓ |
| `08_quantum_confined` | ~3.5 | Hard-wall boundaries + harmonic confinement | α = 3.52 ± 0.05 (proof-of-concept) |

**Key finding:** The RTM scaling law correctly recovers known transport regimes (ballistic α=1, diffusive α=2) and successfully predicts intermediate/extreme regimes determined by network topology.

---

## 016 - Aetherion, The Jumper

**Purpose:** Validate vacuum gradient propulsion and branch-jumping physics.

### Chapter I: Topological Capacitor (Energy Storage)

| Simulation | V1 Claim | Red Team Finding |
|------------|----------|------------------|
| `S1_1D_slab` | P ∝ (∇α)² extracts power | **Net power = 0** — static gradient stores energy, doesn't extract it |
| `S2_2D_simulation` | Radial gradient produces thrust | **Forces cancel geometrically** — requires asymmetric pulsing |
| `S3_scaling_analysis` | Power scales with gradient | **Stored stress scales as Δα³** — steep gradients suppress thermal noise |

**Verdict:** Static metamaterials act as "Topological Capacitors" — loaded spatial springs that store vacuum energy but require dynamic pulsing to release it. **First Law of Thermodynamics preserved.**

### Chapter II: Propulsion & Dynamics

| Simulation | V1 Claim | Red Team Finding |
|------------|----------|------------------|
| `S1_static_thrust` | Continuous free thrust | **Bootstrap Fallacy** — static force is internal stress, not momentum |
| `S2_OMV_vibration` | Oscillation produces thrust | **Ponderomotive rectification works** — vibration → DC push ✓ |
| `S3_TPH_structural` | Piezo pulses produce impulse | **~123 pN·s impulse per pulse** via asymmetric shockwave ✓ |
| `S4_levitation_hover` | Static levitation possible | **Requires active TPH pulse control** with PD feedback loop |
| `S5_inertial_mitigation` | α=50 shield reduces 100g → 2g | **Works but introduces jerk** — needs mechanical dampers |

**Verdict:** Propulsion requires **dynamic symmetry breaking** (OMV oscillation or TPH pulses), not static gradients. **Newton's Third Law preserved.**

### Chapter III: Branch-Jumping (FTL)

| Simulation | V1 Claim | Red Team Finding |
|------------|----------|------------------|
| `S1_multiwell_potential` | Polynomial potential works | **Failed** — replaced with Sine-Gordon potential ✓ |
| `S2_1D_branch_jump` | Controlled jump to Branch 1 | **Avalanche risk** — requires topological damping |
| `S3_3D_verification` | 1D results extend to 3D | **Surface tension dominates** — requires super-critical pulse |
| `S4_jump_threshold` | Linear scaling with radius | **Nucleation theory applies** — macroscopic only (R > 1m) |
| `S5_grid_convergence` | Numerical artifact check | **Converged** — PDE framework mathematically sound ✓ |

**Verdict:** Branch-jumping is a **violent macroscopic phase transition**, not a frictionless mathematical trick. Requires immense energy, massive damping, and is strictly impossible at microscopic scales. **Quantum Field Theory constraints preserved.**

---

## 017 - RTM Unified Field Framework

**Purpose:** Validate the α-field as a dynamical quantum field with correct QFT behavior.

### Section 3.1.3: Quantum Corrections

| Simulation | Tests | Result |
|------------|-------|--------|
| `S1_coleman_weinberg` | One-loop effective potential | Minima shift Δα ≈ ±0.04 ✓ |
| `S2_quantum_bands` | Band structure under quantum corrections | All 5 bands shift with μ ✓ |
| `S3_rg_flow` | β-functions for RTM couplings | All couplings run correctly ✓ |
| `S4_two_loop` | Perturbation theory convergence | |V₂| << |V₁| << |V_tree| ✓ |

### Section 3.3: Holographic AdS/CFT

| Simulation | Tests | Result |
|------------|-------|--------|
| `S1_ads_alpha_profile` | α-field in AdS bulk | Correct radial dependence ✓ |
| `S2_holographic_rg_flow` | Bulk-to-boundary RG | β-functions match ✓ |
| `S3_boundary_correlators` | Two-point functions | VEV scaling correct ✓ |
| `S4_bh_thermodynamics` | Black hole entropy vs α | Bekenstein-Hawking modified ✓ |

### Section 3.5: RG Unification

| Simulation | Tests | Result |
|------------|-------|--------|
| `S1_gauge_rge_running` | SM gauge couplings with α | Running modified ✓ |
| `S2_threshold_matching` | α-band threshold effects | Catalogue generated ✓ |
| `S3_unification_fit` | GUT scale with α corrections | *Deprecated* — see Red Team |
| `S4_alpha_shift_effect` | Unification sensitivity to α | Validated with fixes ✓ |

### Section 4: Numerical Field Solutions

| Simulation | Tests | Result |
|------------|-------|--------|
| `S1_block_matrix_solver` | Coupled PDE solver | 1D/2D solutions verified ✓ |
| `S2_field_profiles_power` | Power scaling with gradient | P ∝ (∇α)² confirmed ✓ |
| `S3_mesh_convergence` | Numerical stability | Grid-independent ✓ |
| `S4_sierpinski_fractal` | Fractal topology effects | α matches d_f ✓ |
| `S5_vascular_tree` | Biological network transport | Murray scaling recovered ✓ |

### Section 6.3: Experimental Signatures

| Simulation | Predicts | Magnitude |
|------------|----------|-----------|
| `S1_calorimetric_power` | Thermal output in gradient | P ∝ Δα² |
| `S2_rf_suppression` | EM noise reduction in α-regions | Frequency-dependent cutoff |
| `S3_photon_delay` | Light delay through gradient | Δt ∝ Δα · L |
| `S4_multimodal_validation` | Combined signature detection | Multi-instrument protocol |

---

## Red Team Methodology

Several simulations include **Red Team audits**, adversarial tests designed to catch:

1. **Overunity fallacies:** Claims of energy from nothing
2. **Bootstrap fallacies:** Momentum without reaction
3. **Confirmation bias:** Metrics that only look favorable
4. **Numerical artifacts:** Results that depend on grid size

Red Team audits inject:
- Thermal noise (5-15%)
- Manufacturing defects (spatial noise)
- Sensor latency (realistic control delays)
- Strict thermodynamic bookkeeping

**Finding pattern:** Original simulations (V1) often had correct mathematics but wrong physical interpretations. Red Team corrections preserved the math while fixing the physics.

---

## Papers 003–015: Empirical Validations (See Folder VII)

Papers 003–015 are validated against **real-world data** and therefore reside in **Folder VII (Empirical and Heuristic Validations)**:

| Paper | Domain | Data Source |
|-------|--------|-------------|
| 003 | Visual Cortex | Receptive field sizes, response latencies |
| 004 | Cosmology | JWST high-redshift galaxies |
| 005 | Gravitational Waves | LIGO/Virgo/KAGRA catalogs (O1-O4) |
| 006 | Quantum Computing | IBM quantum processor decoherence |
| 007 | Chemistry | Zeolite diffusion, transport networks |
| 008 | Biochemistry | Enzyme kinetics, protein folding |
| 009 | Homeostasis | Heart rate variability (PhysioNet) |
| 010 | Neuroscience | EEG states (sleep, meditation, epilepsy) |
| 011 | Consciousness | Anesthesia depth markers |
| 012 | Ecology/Epidemiology | AnAge database, COVID-19 spread |
| 013 | Meteorology | IBTrACS hurricanes, climate extremes |
| 014 | Astronomy | SPARC galaxies, solar wind plasma |
| 015 | Economics | Bitcoin crashes (Binance data) |

These papers test RTM predictions against external ground truth. Papers 001, 016, and 017 (this folder) test internal mathematical consistency because no external ground truth yet exists for their predictions.

---

## Reproducibility

All simulations are designed for reproducibility:

```bash
# Option 1: Direct Python
pip install -r requirements.txt
python simulation_name.py

# Option 2: Jupyter
jupyter notebook simulation_name.ipynb

# Option 3: Docker (recommended)
docker build -t rtm-simulation-name .
docker run --rm -v $(pwd)/output:/app/output rtm-simulation-name
```

Random seeds are fixed. All dependencies are pinned. Docker containers guarantee identical execution environments.

---

## Interpreting Results

**What these simulations prove:**
- RTM equations are mathematically consistent
- Scaling laws emerge from topology as predicted
- Thermodynamic constraints are satisfied
- QFT formalism is correctly implemented

**What these simulations do NOT prove:**
- That vacuum gradients exist in the real world
- That metamaterials can create α-gradients
- That branch-jumping is physically possible
- That the α-field is a real physical entity

The simulations validate the *internal logic* of RTM. Empirical validation requires experiments that do not yet exist.

---

## Citation

If you use this work, please cite:

```
Quiceno, Á. (2026). Corpus Rythmos.
https://github.com/zarpafantasma/corpus_rythmos
```

---

## License

© 2026 Álvaro José Quiceno Rendón  
Distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
Note: **Use the most recent Zenodo DOI identifier.**
