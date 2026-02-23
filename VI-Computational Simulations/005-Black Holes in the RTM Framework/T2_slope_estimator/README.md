# T2: Slope-at-r Estimator

From "Black Holes in the RTM Framework"

## Overview
Estimates the RTM coherence exponent α from log-log regression of τ vs L.

## Method
1. Fit log(τ) vs log(L) at fixed radius
2. Slope = α(r), independent of GR redshift Z(r)
3. Bootstrap for confidence intervals

## Key Equation
log(τ_obs) = α × log(L) + intercept

## Two-Radius Test
Δα = α(r_inner) - α(r_outer)
If 95% CI excludes 0 → Activation detected
