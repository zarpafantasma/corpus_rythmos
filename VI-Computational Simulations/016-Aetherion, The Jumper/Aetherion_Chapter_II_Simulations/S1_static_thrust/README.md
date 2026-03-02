# S1: Static Thrust Calculator

From "Aetherion, The Jumper" - Chapter II, Section 2.1

## Overview
Calculates thrust per unit area from static α-gradients.

## Key Equation
```
F/A = κ_eff × |∇α| × ε_vac
```

## Usage
```bash
python S1_static_thrust.py
```

## Key Results
- Thrust scales linearly with |∇α|
- Direction fixed by sign of ∇α
- ~nN scale forces with current ε_vac estimates
