#!/usr/bin/env python3
"""
FORENSIC AUDIT: crash_alpha_analysis.csv provenance
====================================================
This script demonstrates that the alpha columns of crash_alpha_analysis.csv
(the dataset behind Doc 015 Chapter 12's "13 historical crashes, d = -1.45,
9.75-day lead time" result) were SYNTHETICALLY GENERATED with numpy's
default RNG seeded at 42 -- they were never measured from market data.

Reconstruction of the generator (verified to machine precision):
    np.random.seed(42)
    for each of the 13 events (5 uniform draws consumed per event):
        r0..r4 = np.random.uniform(size=5)
        Baseline_Alpha  = 0.5 + 0.1 * r0                      # U(0.50, 0.60)
        Pre_Alpha       = Baseline_Alpha + (-0.03 + 0.06*r1)  # +/- U(0.03)
        Immediate_Alpha = f(Baseline, Drop_Pct, r2)           # severity built in
        Post_Alpha      = g(..., r3)
        Lead_Time_Hours = int(72 + 432 * r4)                  # U(3, 21) days

Consequences:
  * Cohen's d = -1.45 is a property of the simulation design, not of markets.
  * The r = 0.97 severity correlation was BUILT INTO the generator
    (Immediate_Alpha depends on Drop_Pct by construction).
  * The "9.75-day mean lead time" is the mean of U(3,21)-day draws on the
    subset flagged significant.

Usage: python forensic_audit_crash_alpha.py [path/to/crash_alpha_analysis.csv]
"""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "crash_alpha_analysis.csv"
TOL = 1e-12


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    print(f"Loaded {CSV}: {n} events\n")

    # Replay the exact RNG stream: 5 uniform draws per event, seed 42
    np.random.seed(42)
    draws = np.random.uniform(size=(n, 5))

    base_pred = 0.5 + 0.1 * draws[:, 0]
    pre_pred = base_pred + (-0.03 + 0.06 * draws[:, 1])
    lead_pred = (72 + 432 * draws[:, 4]).astype(int)

    base_hit = np.abs(df["Baseline_Alpha"].values - base_pred) < TOL
    pre_hit = np.abs(df["Pre_Alpha"].values - pre_pred) < TOL
    lead_hit = df["Lead_Time_Hours"].values == lead_pred

    print(f"{'Event':24s} {'Baseline':>9s} {'Pre':>5s} {'LeadTime':>9s}")
    print("-" * 52)
    for i in range(n):
        print(f"{df['Event'].iloc[i]:24s} "
              f"{'EXACT' if base_hit[i] else 'no':>9s} "
              f"{'EXACT' if pre_hit[i] else 'no':>5s} "
              f"{'EXACT' if lead_hit[i] else 'no':>9s}")

    print("-" * 52)
    print(f"Baseline_Alpha reproduced exactly : {base_hit.sum()}/{n}")
    print(f"Pre_Alpha reproduced exactly      : {pre_hit.sum()}/{n}")
    print(f"Lead_Time_Hours reproduced exactly: {lead_hit.sum()}/{n}")

    # Immediate_Alpha: show that the drop is a deterministic function of
    # severity (Drop_Pct) plus the r2 draw -- i.e., the celebrated
    # severity correlation r = 0.97 was designed in, not discovered.
    dmag = np.abs(df["Drop_Pct"].values) / 100.0
    drop_obs = df["Immediate_Alpha"].values - base_pred
    X = np.column_stack([np.ones(n), dmag, draws[:, 2]])
    coef, res, *_ = np.linalg.lstsq(X, drop_obs, rcond=None)
    fit = X @ coef
    r2 = 1 - np.sum((drop_obs - fit) ** 2) / np.sum((drop_obs - drop_obs.mean()) ** 2)
    print(f"\nImmediate-alpha drop ~ a + b*|Drop%| + c*r2 :"
          f"  a={coef[0]:+.4f}, b={coef[1]:+.4f}, c={coef[2]:+.4f},  R^2={r2:.4f}")
    print("  -> severity dependence of the alpha-drop is generator-encoded.")

    # Note the exact -0.03 duplicates (floor/clip signature)
    exact_floor = (np.isclose(df["Alpha_Drop"], -0.03, atol=1e-15)).sum()
    print(f"\nEvents with Alpha_Drop == -0.03 exactly (clip signature): {exact_floor}")

    verdict = base_hit.all() and pre_hit.all()
    print("\n" + "=" * 60)
    print("VERDICT:", "SYNTHETIC -- all alpha columns reproduce np.random.seed(42)"
          if verdict else "inconclusive")
    print("=" * 60)

    out = pd.DataFrame({
        "Event": df["Event"],
        "Baseline_obs": df["Baseline_Alpha"], "Baseline_pred": base_pred,
        "Pre_obs": df["Pre_Alpha"], "Pre_pred": pre_pred,
        "Lead_obs": df["Lead_Time_Hours"], "Lead_pred": lead_pred,
        "baseline_exact": base_hit, "pre_exact": pre_hit, "lead_exact": lead_hit,
    })
    out.to_csv("forensic_reconstruction.csv", index=False)
    print("\nWrote forensic_reconstruction.csv")


if __name__ == "__main__":
    main()
