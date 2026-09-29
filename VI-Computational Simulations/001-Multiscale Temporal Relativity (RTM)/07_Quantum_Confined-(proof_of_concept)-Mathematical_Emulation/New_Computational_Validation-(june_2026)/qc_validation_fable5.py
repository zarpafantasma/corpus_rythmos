#!/usr/bin/env python3
"""
==============================================================================
RTM Simulation H — Quantum-Confined Regime (α ≈ 3.5)
INDEPENDENT RE-VALIDATION — Claude Fable 5 implementation
==============================================================================

Purpose
-------
Independent re-implementation of the quantum-confined proof-of-concept model
(reference package: 08_Quantum_Confined, executed with Claude Opus 4.5,
seed=42). This script:

  1. Re-implements the model from its written specification (NOT copied code):
     - 3D cubic lattice, side L, hard walls (no periodic wrapping)
     - 6-connectivity short-range edges
     - NO long-range links
     - Boundary self-loops: n = floor(beta * L^gamma) * shell_factor,
       beta=1.5, gamma=1.0, boundary_depth=1
       shell_factor = depth - d_wall + 1  (d_wall=0 -> 2, d_wall=1 -> 1)
     - Observable: first-passage time (FPT) corner (0,0,0) -> (L-1,L-1,L-1)

  2. Uses an INDEPENDENT RNG seed (2026) and an independent code path
     (numba-JIT CSR walker), so agreement with the reference cannot come
     from shared code or shared random draws.

  3. Reproduces the reference protocol exactly for the PRIMARY fit:
     L in {5,6,7,8,10,12,14,16,18}, 400 walks/size, max 3M steps,
     log-log OLS, 10k bootstrap.

  4. EXTENSION: adds L in {20, 22, 24} for a finite-size convergence
     analysis (excluded from the primary fit; reported separately).

Epistemic status (inherited from the reference document):
  This is a CONSISTENCY CHECK of a calibrated proof-of-concept model.
  Agreement confirms reproducibility and robustness of the claimed result;
  it does NOT constitute independent physical validation of alpha=3.5.

License: CC BY 4.0
"""

import json
import os
import time

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CONFIG = {
    "primary_sizes": [5, 6, 7, 8, 10, 12, 14, 16, 18],   # reference protocol
    "extension_sizes": [20, 22, 24],                       # convergence probe
    "n_walks_per_size": 400,        # 8 realizations x 50 walks in reference
    "max_steps": 3_000_000,
    "beta": 1.5,                    # potential strength
    "gamma": 1.0,                   # self-loop scaling exponent
    "boundary_depth": 1,
    "seed": 2026,                   # INDEPENDENT seed (reference used 42)
    "n_bootstrap": 10_000,
    "confidence": 0.95,
    "theory_target": 3.5,
    "output_dir": "./output",
}

# Reference results (Opus 4.5 run, seed=42) for comparison
REFERENCE = {
    "alpha": 3.4907, "alpha_se": 0.0677, "R2": 0.99738,
    "ci": (3.4186, 3.5643),
    "T_mean": {5: 1137.15, 6: 2640.82, 7: 4485.67, 8: 7744.50,
               10: 15312.33, 12: 27712.42, 14: 47050.15, 16: 79867.01,
               18: 107787.07},
}


# ----------------------------------------------------------------------------
# Network construction (CSR adjacency, self-loops as explicit entries)
# ----------------------------------------------------------------------------

def build_network_csr(L, beta, gamma, depth):
    """Build the quantum-confined lattice as CSR (indptr, indices)."""
    N = L ** 3
    base_loops = int(np.floor(beta * (L ** gamma)))

    # First pass: degree of each node
    deg = np.zeros(N, dtype=np.int64)
    coords = np.indices((L, L, L)).reshape(3, -1)  # x,y,z rows
    x, y, z = coords
    d_wall = np.minimum.reduce([x, y, z, L - 1 - x, L - 1 - y, L - 1 - z])

    # real neighbors
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        ok = ((x + dx >= 0) & (x + dx < L) &
              (y + dy >= 0) & (y + dy < L) &
              (z + dz >= 0) & (z + dz < L))
        deg += ok

    # self-loops at boundary layer
    is_b = d_wall <= depth
    shell = np.where(is_b, depth - d_wall + 1, 0)
    loops = base_loops * shell
    deg += loops

    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=np.int32)

    fill = indptr[:-1].copy()
    idx_of = (x * L * L + y * L + z).astype(np.int64)

    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        ok = ((nx >= 0) & (nx < L) & (ny >= 0) & (ny < L) &
              (nz >= 0) & (nz < L))
        src = idx_of[ok]
        dst = (nx[ok] * L * L + ny[ok] * L + nz[ok]).astype(np.int32)
        # place each edge
        pos = fill[src]
        indices[pos] = dst
        fill[src] += 1

    # self-loops
    b_nodes = idx_of[is_b]
    b_loops = loops[is_b]
    for node, k in zip(b_nodes, b_loops):
        indices[fill[node]:fill[node] + k] = node
        fill[node] += k

    assert np.array_equal(fill, indptr[1:]), "CSR fill mismatch"

    stats = {
        "N": int(N),
        "n_boundary": int(is_b.sum()),
        "boundary_pct": float(100.0 * is_b.sum() / N),
        "base_loops": int(base_loops),
        "self_loops_total": int(loops.sum()),
        "self_loops_per_boundary": float(loops.sum() / max(is_b.sum(), 1)),
        "mean_degree": float(deg.mean()),
    }
    return indptr, indices, stats


# ----------------------------------------------------------------------------
# Random walk (numba JIT) — independent code path
# ----------------------------------------------------------------------------

@njit(cache=True)
def _walk_batch(indptr, indices, source, target, n_walks, max_steps, seed):
    np.random.seed(seed)
    out = np.empty(n_walks, dtype=np.int64)
    for w in range(n_walks):
        cur = source
        fpt = -1
        for step in range(1, max_steps + 1):
            lo = indptr[cur]
            hi = indptr[cur + 1]
            cur = indices[lo + np.random.randint(0, hi - lo)]
            if cur == target:
                fpt = step
                break
        out[w] = fpt
    return out


def run_size(L, cfg, seed_offset):
    indptr, indices, stats = build_network_csr(
        L, cfg["beta"], cfg["gamma"], cfg["boundary_depth"])
    source = 0
    target = L ** 3 - 1  # (L-1,L-1,L-1) in x*L*L + y*L + z indexing
    t0 = time.time()
    fpts = _walk_batch(indptr, indices, source, target,
                       cfg["n_walks_per_size"], cfg["max_steps"],
                       cfg["seed"] + seed_offset)
    elapsed = time.time() - t0
    ok = fpts[fpts > 0]
    return {
        "L": L,
        "fpts": ok,
        "completed": int(ok.size),
        "failed": int(fpts.size - ok.size),
        "T_mean": float(ok.mean()),
        "T_std": float(ok.std(ddof=1)),
        "T_median": float(np.median(ok)),
        "stats": stats,
        "runtime_s": elapsed,
    }


# ----------------------------------------------------------------------------
# Fitting & bootstrap
# ----------------------------------------------------------------------------

def ols_loglog(Ls, Ts):
    lx = np.log10(np.asarray(Ls, dtype=float))
    ly = np.log10(np.asarray(Ts, dtype=float))
    n = len(lx)
    A = np.vstack([lx, np.ones(n)]).T
    (alpha, b), *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = alpha * lx + b
    res = ly - pred
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se = (np.sqrt(ss_res / max(n - 2, 1) /
                  np.sum((lx - lx.mean()) ** 2)) if n > 2 else np.nan)
    return {"alpha": float(alpha), "intercept": float(b),
            "C": float(10 ** b), "R2": float(r2), "alpha_se": float(se),
            "residuals": res.tolist()}


def theil_sen(Ls, Ts):
    lx = np.log10(np.asarray(Ls, dtype=float))
    ly = np.log10(np.asarray(Ts, dtype=float))
    slopes = []
    for i in range(len(lx)):
        for j in range(i + 1, len(lx)):
            slopes.append((ly[j] - ly[i]) / (lx[j] - lx[i]))
    return float(np.median(slopes))


def bootstrap_alpha(results, n_boot, conf, rng):
    Ls = [r["L"] for r in results]
    lx = np.log10(np.asarray(Ls, dtype=float))
    n = len(lx)
    sx = lx.sum()
    sxx = (lx ** 2).sum()
    denom = n * sxx - sx ** 2

    alphas = np.empty(n_boot)
    fpt_arrays = [r["fpts"] for r in results]
    for b in range(n_boot):
        ly = np.empty(n)
        for i, fp in enumerate(fpt_arrays):
            sample = fp[rng.integers(0, fp.size, fp.size)]
            ly[i] = np.log10(sample.mean())
        alphas[b] = (n * (lx * ly).sum() - sx * ly.sum()) / denom

    lo = float(np.percentile(alphas, 100 * (1 - conf) / 2))
    hi = float(np.percentile(alphas, 100 * (1 + conf) / 2))
    return {"bs_mean": float(alphas.mean()), "bs_std": float(alphas.std()),
            "ci_lo": lo, "ci_hi": hi, "alphas": alphas}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    all_sizes = cfg["primary_sizes"] + cfg["extension_sizes"]
    print(f"RTM Quantum-Confined — independent re-validation (Fable 5)")
    print(f"seed={cfg['seed']}  walks/size={cfg['n_walks_per_size']}")
    print("-" * 70)

    results = []
    for k, L in enumerate(all_sizes):
        r = run_size(L, cfg, seed_offset=k * 1000)
        results.append(r)
        print(f"L={L:3d}  N={r['stats']['N']:6d}  "
              f"T_mean={r['T_mean']:12.1f}  T_med={r['T_median']:10.1f}  "
              f"completed={r['completed']}/{cfg['n_walks_per_size']}  "
              f"({r['runtime_s']:.1f}s)")

    primary = [r for r in results if r["L"] in cfg["primary_sizes"]]
    extended = results  # all 12 sizes

    # --- Primary fit (reference protocol) ---
    fit = ols_loglog([r["L"] for r in primary],
                     [r["T_mean"] for r in primary])
    bs = bootstrap_alpha(primary, cfg["n_bootstrap"], cfg["confidence"], rng)
    ts_slope = theil_sen([r["L"] for r in primary],
                         [r["T_mean"] for r in primary])

    # Robustness: leave-one-out at the edges
    fit_excl_max = ols_loglog([r["L"] for r in primary[:-1]],
                              [r["T_mean"] for r in primary[:-1]])
    fit_excl_min = ols_loglog([r["L"] for r in primary[1:]],
                              [r["T_mean"] for r in primary[1:]])

    # --- Extended fit (all 12 sizes) ---
    fit_ext = ols_loglog([r["L"] for r in extended],
                         [r["T_mean"] for r in extended])
    bs_ext = bootstrap_alpha(extended, cfg["n_bootstrap"],
                             cfg["confidence"], rng)

    # Sliding 5-point window alphas (convergence diagnostic)
    win = []
    Ls_all = [r["L"] for r in extended]
    Ts_all = [r["T_mean"] for r in extended]
    for i in range(len(Ls_all) - 4):
        w = ols_loglog(Ls_all[i:i + 5], Ts_all[i:i + 5])
        win.append({"L_center": Ls_all[i + 2], "alpha": w["alpha"]})

    target = cfg["theory_target"]
    print("-" * 70)
    print(f"PRIMARY fit (9 sizes, reference protocol):")
    print(f"  alpha = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}   "
          f"R² = {fit['R2']:.6f}")
    print(f"  bootstrap 95% CI = [{bs['ci_lo']:.4f}, {bs['ci_hi']:.4f}]  "
          f"(includes 3.5: {bs['ci_lo'] <= target <= bs['ci_hi']})")
    print(f"  Theil–Sen = {ts_slope:.4f}   "
          f"excl-max = {fit_excl_max['alpha']:.4f}   "
          f"excl-min = {fit_excl_min['alpha']:.4f}")
    print(f"EXTENDED fit (12 sizes, L up to 24):")
    print(f"  alpha = {fit_ext['alpha']:.4f} ± {fit_ext['alpha_se']:.4f}   "
          f"R² = {fit_ext['R2']:.6f}   "
          f"CI = [{bs_ext['ci_lo']:.4f}, {bs_ext['ci_hi']:.4f}]")
    print(f"  Reference (Opus 4.5, seed 42): alpha = {REFERENCE['alpha']}  "
          f"CI = {REFERENCE['ci']}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    import pandas as pd

    rows = []
    for r in extended:
        rows.append({
            "L": r["L"], "N": r["stats"]["N"], "T_mean": r["T_mean"],
            "T_std": r["T_std"], "T_median": r["T_median"],
            "completed": r["completed"],
            "total_walks": cfg["n_walks_per_size"],
            "boundary_pct": r["stats"]["boundary_pct"],
            "self_loops_per_boundary": r["stats"]["self_loops_per_boundary"],
            "mean_degree": r["stats"]["mean_degree"],
            "in_primary": r["L"] in cfg["primary_sizes"],
            "T_mean_reference": REFERENCE["T_mean"].get(r["L"], np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["output_dir"], "qc_fable5_results.csv"),
              index=False)

    walks_rows = []
    for r in extended:
        for f in r["fpts"]:
            walks_rows.append({"L": r["L"], "fpt": int(f)})
    pd.DataFrame(walks_rows).to_csv(
        os.path.join(cfg["output_dir"], "qc_fable5_walks.csv"), index=False)

    summary = {
        "validation": "RTM Quantum-Confined — independent re-validation",
        "model_engine": "Claude Fable 5 (independent implementation, numba)",
        "reference_engine": "Claude Opus 4.5 (original package, seed 42)",
        "seed": cfg["seed"],
        "protocol": {
            "primary_sizes": cfg["primary_sizes"],
            "extension_sizes": cfg["extension_sizes"],
            "walks_per_size": cfg["n_walks_per_size"],
            "max_steps": cfg["max_steps"],
            "beta": cfg["beta"], "gamma": cfg["gamma"],
            "boundary_depth": cfg["boundary_depth"],
        },
        "primary_fit": {
            "alpha": fit["alpha"], "alpha_se": fit["alpha_se"],
            "R2": fit["R2"], "C": fit["C"],
            "bootstrap_ci_95": [bs["ci_lo"], bs["ci_hi"]],
            "includes_target": bool(bs["ci_lo"] <= target <= bs["ci_hi"]),
            "theil_sen": ts_slope,
            "alpha_excl_max": fit_excl_max["alpha"],
            "alpha_excl_min": fit_excl_min["alpha"],
        },
        "extended_fit": {
            "alpha": fit_ext["alpha"], "alpha_se": fit_ext["alpha_se"],
            "R2": fit_ext["R2"],
            "bootstrap_ci_95": [bs_ext["ci_lo"], bs_ext["ci_hi"]],
            "includes_target": bool(
                bs_ext["ci_lo"] <= target <= bs_ext["ci_hi"]),
        },
        "sliding_window_alphas": win,
        "reference_comparison": {
            "alpha_ref": REFERENCE["alpha"],
            "alpha_this_run": fit["alpha"],
            "delta_alpha": fit["alpha"] - REFERENCE["alpha"],
            "z_score_vs_ref": (fit["alpha"] - REFERENCE["alpha"]) /
                              np.hypot(fit["alpha_se"],
                                       REFERENCE["alpha_se"]),
        },
        "epistemic_status": (
            "Consistency check of a CALIBRATED proof-of-concept model "
            "(beta, gamma selected to hit the target in the reference run). "
            "This re-run confirms reproducibility across implementation, "
            "RNG seed, and model engine — it does not constitute "
            "independent physical validation of alpha = 3.5."),
    }
    with open(os.path.join(cfg["output_dir"], "qc_fable5_summary.json"),
              "w") as fh:
        json.dump(summary, fh, indent=2)

    np.save(os.path.join(cfg["output_dir"], "bootstrap_alphas.npy"),
            bs["alphas"])

    return extended, fit, bs, fit_ext, bs_ext, win, summary


if __name__ == "__main__":
    main()
