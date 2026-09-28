"""High-statistics follow-up: resolve the alpha drift at large L.
2000 walks/size for L in {12,14,16,18,20,22,24}, fresh seeds."""
import json, time
import numpy as np
from qc_validation_fable5 import build_network_csr, _walk_batch, ols_loglog, CONFIG

SIZES = [12, 14, 16, 18, 20, 22, 24]
N_WALKS = 2000
cfg = CONFIG

res = {}
for k, L in enumerate(SIZES):
    indptr, indices, stats = build_network_csr(L, cfg["beta"], cfg["gamma"], cfg["boundary_depth"])
    t0 = time.time()
    f = _walk_batch(indptr, indices, 0, L**3 - 1, N_WALKS, cfg["max_steps"], 777000 + k * 137)
    ok = f[f > 0]
    res[L] = ok
    se = ok.std(ddof=1)/np.sqrt(ok.size)
    print(f"L={L:3d} T_mean={ok.mean():12.1f} ± {se:8.1f} ({100*se/ok.mean():.1f}%) n={ok.size} ({time.time()-t0:.0f}s)")

# pairwise local slopes between consecutive sizes
print("\nLocal pairwise slopes alpha(L_i -> L_{i+1}):")
Ls = SIZES
for i in range(len(Ls)-1):
    a = (np.log10(res[Ls[i+1]].mean()) - np.log10(res[Ls[i]].mean())) / (np.log10(Ls[i+1]) - np.log10(Ls[i]))
    print(f"  {Ls[i]:2d} -> {Ls[i+1]:2d}: alpha_local = {a:.3f}")

# window fits
for lo in range(0, 3):
    sub = Ls[lo:]
    fit = ols_loglog(sub, [res[L].mean() for L in sub])
    print(f"OLS window L={sub}: alpha = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}  R²={fit['R2']:.5f}")

np.save("highstat_fpts.npy", {L: res[L] for L in SIZES}, allow_pickle=True)
