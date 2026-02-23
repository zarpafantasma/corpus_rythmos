# 007-Rhythmic Chemistry: RTM Kinetics and Selectivity

## Paper: "Rhythmic Chemistry: An RTM Framework for Kinetics and Selectivity"

---

## Key Results

| Test | Result |
|------|--------|
| α recovery accuracy | **2.2% error** |
| Enhancement at 10nm (α=2.3) | **200×** |
| Microporous enhancement | **5467×** (2nm, α=2.2) |
| Selectivity enhancement | **6.3×** at 1nm |

---

## Core RTM Prediction

**k(L) = A₀ × (L/L_ref)^(−α) × exp(−E_a/RT)**

| Environment | α | Rate Effect |
|-------------|---|-------------|
| Bulk solution | — | Baseline |
| Diffusive | 2.0 | 100× at 10nm |
| Hierarchical | 2.2-2.5 | 200-300× at 10nm |

---

## Package Contents

```
007-Rhythmic_Chemistry/
│
├── S1_arrhenius_rtm/                      ← Classic vs RTM Arrhenius
│   ├── S1_arrhenius_rtm.py
│   └── output/
│       ├── S1_arrhenius_comparison.png
│       └── S1_alpha_extraction.png
│
├── S2_microreactors/                      ← Reactor Design
│   ├── S2_microreactors.py
│   └── output/
│       ├── S2_platform_comparison.png
│       ├── S2_optimization.png
│       └── S2_design_nomogram.png
│
├── S3_zeolite_selectivity/                ← Selectivity Prediction
│   ├── S3_zeolite_selectivity.py
│   └── output/
│       ├── S3_selectivity_scenarios.png
│       ├── S3_material_comparison.png
│       └── S3_design_space.png
│
└── requirements.txt
```

---

## Simulation Summaries

### S1: Arrhenius Classic vs RTM

Demonstrates that confinement adds a length-dependent factor:
- Same activation energy E_a
- Different apparent pre-exponential A_app ∝ L^(-α)
- α recoverable from isothermal data with 2% error

### S2: Microreactor Rate Predictions

Applies RTM to practical systems:
- **Microfluidic** (100μm): minimal enhancement
- **Mesoporous** (10nm): 158× enhancement
- **Microporous** (2nm): 5467× enhancement
- Includes diffusion limitation (Thiele modulus)

### S3: Zeolite/MOF Selectivity

Predicts selectivity tuning via pore size:
- Different α for competing reactions → tunable selectivity
- Database of zeolites and MOFs with pore sizes
- Design space map for selectivity engineering

---

## Running the Simulations

```bash
pip install numpy scipy pandas matplotlib

python S1_arrhenius_rtm/S1_arrhenius_rtm.py
python S2_microreactors/S2_microreactors.py
python S3_zeolite_selectivity/S3_zeolite_selectivity.py
```

---

## Falsification Tests

1. **Slope stability:** log(k) vs log(L) should give constant α
2. **Data collapse:** k × L^α should be constant
3. **Platform independence:** Same α from different confinement methods
4. **Temperature independence:** α should not vary with T

---

## Materials Database

### Zeolites
| Material | Pore (nm) | Topology |
|----------|-----------|----------|
| ZSM-5 | 0.55 | MFI |
| Mordenite | 0.70 | MOR |
| Beta | 0.76 | BEA |
| Y (Faujasite) | 0.74 | FAU |

### MOFs
| Material | Pore (nm) | Type |
|----------|-----------|------|
| UiO-66 | 0.75 | Zr-based |
| ZIF-8 | 1.16 | Zn-imidazolate |
| MOF-5 | 1.50 | Zn-carboxylate |
| MIL-101 | 3.4 | Cr-based |

### Mesoporous
| Material | Pore (nm) |
|----------|-----------|
| MCM-41 | 3.0 |
| SBA-15 | 8.0 |
