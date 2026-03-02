# S1: Block-Matrix Solver for RTM-Aetherion Equations

From "RTM Unified Field Framework" - Section 4.1

## Overview
Implements finite-difference discretization and block-matrix solver for coupled field equations in 1D, 2D, and 3D.

## Key Equations
```
∇²φ - m²φ = -γ|∇α|²
M²∇²α = γ∇²φ
```

## Block System
```
[A_φ   -C  ] [φ]   [source]
[C    A_α ] [α] = [  0   ]
```

## Usage
```bash
python S1_block_matrix_solver.py
```
