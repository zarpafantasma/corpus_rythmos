# Aetherion Chapter II: Reactionless Propulsion & Temporal Hopping

## Overview

This package contains five computational tools (S1-S5) that implement the **thrust, levitation, and inertial mitigation** mechanisms described in Chapter II of "Aetherion, The Jumper".

Building on Chapter I's vacuum energy extraction, Chapter II extends the Aetherion framework to:
- Generate directed thrust from α-gradients
- Achieve stable levitation (hovering)
- Mitigate G-forces via temporal decoupling

---

## Theoretical Background

### Thrust Mechanism

**Key Equation (Section 2.1):**
```
F/A = κ_eff × |∇α| × ε_vac
```

The α-gradient acts as a "pump" that rectifies vacuum fluctuations into directed momentum flux. Key properties:
- **Directionality**: Sign of ∇α determines thrust direction
- **Scalability**: Larger area or steeper gradient → more force
- **Reactionless**: No propellant expelled

### Inertial Mitigation

**Key Equation (Section 5.4):**
```
a_eff = a_ext / α
```

Inside a high-α region, proper time flows slower. Rapid external maneuvers appear stretched, reducing felt G-forces. With α = 50, a 100g maneuver feels like 2g.

---

## Simulations

### S1: Static Thrust Calculator

**File:** `S1_static_thrust/S1_static_thrust.py`

**Purpose:** Calculate thrust from static α-gradients.

**Key Results:**
- F/A ∝ |∇α| (linear scaling)
- Direction reversible by flipping gradient
- ~nN scale with current ε_vac estimates

---

### S2: OMV Vibration-Induced Thrust

**File:** `S2_OMV_vibration_thrust/S2_OMV_vibration_thrust.py`

**Purpose:** Model thrust from oscillating α-modulation.

**Key Equations:**
```
α(x,t) = α₀ + α₁ sin(kx) cos(ωt)
Δz_pp = (κ_eff × α₁² × k² × ε_vac) / (m × ω²)
```

**Paper Target:** ~0.5 nm displacement (detectable by interferometry)

---

### S3: TPH Structural-Gradient Thrust

**File:** `S3_TPH_structural_thrust/S3_TPH_structural_thrust.py`

**Purpose:** Model thrust from dynamic hierarchy modulation.

**Key Equations:**
```
f_α = κ_eff × ε_vac × ∂α/∂x × ln(L/L₀)  (temporal term)
f_L = κ_eff × ε_vac × α × (1/L) × ∂L/∂x  (geometric term)
```

**Hybrid Strategy:**
- Slow α-shaping → coarse thrust control
- Fast L-pulses → fine impulse control

---

### S4: Levitation and Hover Calculator

**File:** `S4_levitation_hover/S4_levitation_hover.py`

**Purpose:** Calculate gradient needed for stable hover.

**Key Equation:**
```
|∇α| = (m × g) / (κ_eff × ε_vac × A)
```

**Results:**
| Mass | Required ∇α (A = 1 cm²) |
|------|-------------------------|
| 1 ng | ~10³ /m |
| 1 µg | ~10⁶ /m |
| 1 g  | ~10⁹ /m |

With practical gradient limits (~10³ /m), nanogram-scale levitation is feasible.

---

### S5: Inertial Mitigation

**File:** `S5_inertial_mitigation/S5_inertial_mitigation.py`

**Purpose:** Simulate G-force reduction via temporal decoupling.

**Key Equation:**
```
a_eff = a_ext / α
```

**Comfort Requirements (< 2g felt):**
| External | Required α |
|----------|------------|
| 10g | 5 |
| 30g | 15 |
| 100g | 50 |
| 500g | 250 |

---

## Key Findings

| Simulation | Paper Prediction | Result |
|------------|------------------|--------|
| S1: Static Thrust | F ∝ \|∇α\| | ✓ Verified |
| S2: OMV | ~0.5 nm displacement | Methodology correct |
| S3: TPH | Hybrid control | ✓ Verified |
| S4: Levitation | Hover via gradient | ✓ ng-scale feasible |
| S5: Mitigation | a_eff = a_ext/α | ✓ Verified |

---

## Physical Interpretation

The Chapter II simulations demonstrate:

1. **Directed thrust without propellant** - Momentum exchanged with vacuum fluctuations, not reaction mass.

2. **Scalable levitation** - Stable hover achieved by balancing weight against gradient-induced lift.

3. **G-force immunity** - Temporal decoupling in high-α regions reduces perceived acceleration.

4. **Hybrid control** - Combining slow α-shaping with fast L-pulses enables precise maneuvering.

---

## Usage

### Direct Execution
```bash
cd S1_static_thrust
pip install -r requirements.txt
python S1_static_thrust.py
```

### Docker
```bash
cd S1_static_thrust
docker build -t aetherion_s1_ch2 .
docker run -v $(pwd)/output:/app/output aetherion_s1_ch2
```

### Jupyter
```bash
jupyter notebook S1_static_thrust.ipynb
```

---

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

---

## Notes on Magnitude

The simulations use ε_vac = 10⁻⁹ J/m³ as the accessible vacuum energy density. This gives very small forces (nN scale) and limits practical levitation to nanogram masses.

The paper suggests that with engineered materials and enhanced coupling, larger effects may be achievable. The simulations demonstrate the **methodology** and **scaling laws**, which are the scientifically testable predictions.

---

## Reference

Paper: "Aetherion, The Jumper"  
Chapter: II - Reactionless Propulsion & Temporal Hopping  
RTM Corpus Document: 016

---

## License

CC BY 4.0
