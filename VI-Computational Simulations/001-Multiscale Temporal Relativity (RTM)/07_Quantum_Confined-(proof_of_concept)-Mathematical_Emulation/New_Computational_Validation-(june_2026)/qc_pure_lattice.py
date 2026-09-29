"""Pure 3D lattice (no self-loops) baseline at matching sizes."""
import time
import numpy as np
from qc_validation_fable5 import build_network_csr, _walk_batch, ols_loglog, CONFIG

SIZES = [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24]
N_WALKS = 2000
cfg = CONFIG

res = {}
for k, L in enumerate(SIZES):
    # beta=0 -> no self-loops -> pure lattice
    indptr, indices, stats = build_network_csr(L, 0.0, 1.0, 1)
    t0 = time.time()
    f = _walk_batch(indptr, indices, 0, L**3 - 1, N_WALKS, cfg["max_steps"], 555000 + k * 91)
    ok = f[f > 0]
    res[L] = ok
    se = ok.std(ddof=1)/np.sqrt(ok.size)
    print(f"L={L:3d} T_mean={ok.mean():11.1f} ± {100*se/ok.mean():.1f}%  ({time.time()-t0:.0f}s)")

print("\nPure-lattice window fits:")
for sub in ([5,6,7,8,10,12,14,16,18], [12,14,16,18,20,22,24], [16,18,20,22,24]):
    fit = ols_loglog(sub, [res[L].mean() for L in sub])
    print(f"  L={sub}: alpha_base = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}  R²={fit['R2']:.5f}")

np.save("pure_lattice_fpts.npy", {L: res[L] for L in SIZES}, allow_pickle=True)
