"""Test of the DERIVED mechanism: gamma = 1.5 should give asymptotic
alpha = 3.5 via the commute-time identity MFPT ~ 2|E| R_eff, R_eff = O(1) in 3D.
gamma is NOT tuned: it is fixed by requiring |E| ~ L^3.5.
beta only sets the intercept asymptotically (test two values)."""
import time
import numpy as np
from qc_validation_fable5 import build_network_csr, _walk_batch, ols_loglog

SIZES = [8, 10, 12, 14, 16, 18, 20, 22, 24]
N_WALKS = 800
MAX_STEPS = 30_000_000

for beta in (1.0, 0.5):
    print(f"\n=== gamma = 1.5, beta = {beta} (derived, not calibrated) ===")
    res = {}
    for k, L in enumerate(SIZES):
        indptr, indices, stats = build_network_csr(L, beta, 1.5, 1)
        t0 = time.time()
        f = _walk_batch(indptr, indices, 0, L**3 - 1, N_WALKS, MAX_STEPS, int(888000 + beta*100000 + k * 53))
        ok = f[f > 0]
        res[L] = ok
        se = ok.std(ddof=1)/np.sqrt(ok.size)
        print(f"L={L:3d} T_mean={ok.mean():13.1f} ± {100*se/ok.mean():.1f}%  n={ok.size}  ({time.time()-t0:.0f}s)")
    for sub in (SIZES, SIZES[2:], SIZES[4:]):
        fit = ols_loglog(sub, [res[L].mean() for L in sub])
        print(f"  window L={sub[0]}-{sub[-1]}: alpha = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}  R²={fit['R2']:.5f}")
    np.save(f"gamma15_beta{beta}_fpts.npy", {L: res[L] for L in SIZES}, allow_pickle=True)
