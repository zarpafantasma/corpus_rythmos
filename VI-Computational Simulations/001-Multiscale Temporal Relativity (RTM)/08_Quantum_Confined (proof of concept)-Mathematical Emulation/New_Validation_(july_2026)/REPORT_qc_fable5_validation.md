# RTM Simulation H — Quantum-Confined Regime (α ≈ 3.5)
## Independent Re-Validation Report — Claude Fable 5

**Reference package:** `08_Quantum_Confined (proof of concept) — Mathematical Emulation` (executed with Claude Opus 4.5, seed 42).
**This validation:** independent re-implementation (numba CSR walker, code written from the model specification, not copied), independent RNG seed (2026), identical model specification and fitting protocol, plus two extension studies.
**License:** CC BY 4.0.

---

## Part 1 — Replication under the reference protocol

The model was rebuilt exactly as specified: a 3D cubic lattice of side L with hard walls, 6-connectivity, no long-range links, and boundary self-loops numbering ⌊β·L^γ⌋ × shell_factor with β = 1.5, γ = 1.0, δ = 1. The observable is the mean first-passage time from corner (0,0,0) to corner (L−1, L−1, L−1), with 400 walks per size over L ∈ {5, 6, 7, 8, 10, 12, 14, 16, 18}, a 3·10⁶-step cap, log–log OLS, and a 10,000-resample bootstrap.

| Quantity | Reference (Opus 4.5, seed 42) | This run (Fable 5, seed 2026) |
|---|---|---|
| α (OLS) | 3.4907 ± 0.0677 | **3.5039 ± 0.0443** |
| R² | 0.99738 | **0.99888** |
| Bootstrap 95% CI | [3.4186, 3.5643] | **[3.4307, 3.5752]** |
| CI includes 3.5 | yes | **yes** |
| Theil–Sen slope | — | 3.5071 |
| α excluding largest L | 3.5335 | 3.5183 |
| α excluding smallest L | 3.3860 | 3.4392 |
| Completion | 100% | 100% |

The two runs differ by Δα = +0.013, corresponding to z = 0.16 against the combined standard errors — statistically indistinguishable. The robust Theil–Sen estimate and both leave-one-edge-out fits agree with the OLS value. **The reference result is fully reproduced across an independent implementation, an independent random seed, and an independent model engine.** Per-size means agree with the reference within sampling error at every L (see `qc_fable5_results.csv`, column `T_mean_reference`).

## Part 2 — Finite-size extension (new analysis)

Doc 001 prescribes a finite-size sanity check: fits with and without the largest L, bootstrap intervals, and a convergence verdict, with systematic window drift to be treated as a finite-size artifact rather than a new regime. Applying that criterion required extending the size range. The lattice was extended to L ∈ {20, 22, 24} and the statistics at large L were increased fivefold (2,000 walks per size, standard errors ≈ 2%), together with a matched pure-lattice baseline (β = 0, identical protocol).

| Window | α confined (β=1.5, γ=1.0) | α pure lattice | Δα (confinement) |
|---|---|---|---|
| L = 5–18 (reference window) | 3.504 ± 0.044 | 3.228 ± 0.018 | **+0.28** |
| L = 12–24 | 3.202 ± 0.030 | 3.147 ± 0.029 | +0.06 |
| L = 16–24 | 3.159 ± 0.051 | 3.178 ± 0.033 | **≈ 0.00** |

The slowdown ratio T_confined / T_pure rises from 3.3 at L = 5 to ≈ 4.9 at L = 16 and then **saturates** (4.92, 4.89, 4.89, 4.92 for L = 16–24). A saturating multiplicative factor is, in the language of Doc 002, a **clock (gauge/intercept) effect, not a slope effect**: asymptotically the γ = 1.0 mechanism multiplies T by a constant and leaves α at the pure-lattice value.

This behavior is not an accident of the data; it follows from an exact identity. For a reversible random walk, Kac's commute-time identity gives E[τ] of order 2|E|·R_eff, where |E| is the total edge weight (self-loops included) and R_eff the effective resistance, which self-loops do not alter because no current flows through them. With per-node loops ∝ β·L^γ on a boundary of ~L² nodes, the self-loop budget scales as L^(2+γ) against a bulk budget of ~6L³. At γ = 1.0 the two scale identically — the correction is exactly marginal, producing the observed O(1) saturation and no asymptotic exponent shift.

**Conclusion of Part 2.** The α ≈ 3.49 of the reference configuration is a *pre-asymptotic, window-dependent effective exponent* produced by the still-growing portion of the slowdown ratio inside the L = 5–18 window. By the corpus's own finite-size criterion this should be classified as a finite-size crossover, not a stable α ≈ 3.5 band. This sharpens, but is consistent with, the package's declared epistemic status ("consistency check, not independent validation").

## Part 3 — A derived mechanism that does produce α = 3.5 (no calibration)

The commute-time analysis immediately specifies what a genuine α = 3.5 mechanism requires: the total self-loop budget must scale as L^3.5, i.e. **per-node boundary loops ∝ L^1.5 (γ = 1.5)**. Unlike the reference configuration, γ here is *derived* from the identity |E| ∝ L^(2+γ) ⇒ α_asym = max(d, 2+γ), not selected by sweeping toward the target. The strength β should affect only the intercept.

Both predictions were tested at L = 8–24 (800 walks/size) for two strength values:

| Window | α (γ=1.5, β=1.0) | α (γ=1.5, β=0.5) |
|---|---|---|
| L = 8–24 | 3.797 ± 0.054 | 3.754 ± 0.040 |
| L = 12–24 | 3.681 ± 0.088 | 3.679 ± 0.058 |
| L = 16–24 | 3.521 ± 0.187 | **3.545 ± 0.082** |

The effective exponent converges toward 3.5 from above (the excess at small L matches the pure-lattice logarithmic corrections, which decay), and the two β values converge to the same asymptote — confirming that β is gauge (intercept) and γ is structure (slope), precisely the slope/clock separation of Doc 002.

**Implication for the corpus.** Simulation H can be upgraded from a calibrated proof of concept (status ◐) to a **derived-mechanism demonstration**: the rule α = (d − 1) + γ for boundary self-loop confinement with γ > 1 yields the quantum-confined exponent α = 3.5 at γ = 1.5 with no parameter fitted to the target, grounded in an exact identity. The physical question then becomes sharper and more falsifiable: does real quantum confinement impose boundary dwell budgets growing as L^1.5 (super-extensive edge weighting), rather than the marginal L^1.0? This is a cleaner topological target for the experimental program of Doc 001 §5.4 than the original "≈ 0.5 lag" formulation.

## Limitations

The walk counts (400–2,000 per size) leave 2–4% standard errors on per-size means; the largest-window fits in Part 3 carry correspondingly wide intervals, and the asymptote is established by the identity plus convergence trend rather than by direct measurement at very large L. All results concern the lattice model class; nothing here validates α = 3.5 in physical quantum-confined systems, which remains an open experimental question (Doc 001 §5.4). The reference run's per-size means at the shared sizes were reproduced within sampling error, so no discrepancy in the original execution was found.

## Files

`qc_validation_fable5.py` (main script), `qc_highstat_large_L.py`, `qc_pure_lattice.py`, `qc_gamma15_test.py` (extension studies), `qc_fable5_results.csv` (per-size summary incl. reference comparison), `qc_fable5_walks.csv` (4,800 primary walks), `qc_fable5_summary.json` (full configuration and fits), `fig1_replication.png`, `fig2_residuals.png`, `fig3_finite_size_drift.png`, `fig4_gamma15_derived.png`, `fig5_bootstrap.png`.

---

*Validation executed by Claude Fable 5 (Anthropic), June 2026. Independent implementation; reference comparison data from the Opus 4.5 package. CC BY 4.0.*
