# RTM Empirical Validation: Biochemistry 🧬

**Date:** February 2026  
**Analyses:** 2 (Protein Folding + Enzyme Kinetics)  
**Total Data Points:** 153 (84 + 69)

---

## Executive Summary

This analysis validates RTM predictions in biochemistry by demonstrating that **global** and **local** processes exhibit fundamentally different scaling behaviors:

| Process | System | n | α | R² | p-value |
|---------|--------|---|---|----|----- |
| **GLOBAL** | Protein Folding | 84 | **+7.2 ± 0.6** | 0.63 | < 10⁻¹⁸ |
| **LOCAL** | Enzyme Kinetics | 69 | **-0.9 ± 0.6** | 0.03 | 0.14 (NS) |

**Key RTM Insight:** The contrast between strong size dependence (folding) and no size dependence (catalysis) demonstrates that RTM can distinguish between process types.

---

## Quick Start

```bash
pip install -r requirements.txt
python analyze_biochemistry_rtm.py
```

Results are saved to the `output/` directory.

---

## Analysis 1: Protein Folding (Global Process)

### The Physics

Protein folding requires the **entire polypeptide chain** to rearrange from a disordered state to its native structure. This is a **global process** where every residue must find its correct position.

The "Levinthal Paradox" noted that random search would take astronomical time, but proteins fold in milliseconds to seconds. RTM explains this through the **folding funnel**: a directed, coherent process with α >> 1.

### Data

| Parameter | Value |
|-----------|-------|
| Proteins | 84 |
| Length range | 16 - 156 residues |
| Rate range | 0.2 - 10⁶ s⁻¹ |
| Two-state folders | 79 |
| Three-state folders | 5 |

### Sources

- Ivankov & Plaxco 2003 (original compilation)
- Maxwell et al. 2005 (two-state folders)
- Kubelka et al. 2004 (ultrafast folders)
- ACPro Database compilations

### Results

| Category | n | α | R² |
|----------|---|---|---|
| **All proteins** | 84 | **+7.2 ± 0.6** | 0.63 |
| Alpha-helical | 26 | +6.0 ± 0.9 | 0.65 |
| Beta-sheet | 29 | +7.5 ± 0.9 | 0.73 |
| Mixed α/β | 29 | +7.2 ± 0.7 | 0.78 |

### Interpretation

- **α ≈ 7**: Strong size dependence
- Each doubling of chain length → ~100× slower folding
- The "folding funnel" creates **coherent dynamics**
- This is NOT random search (which would be exponential)

---

## Analysis 2: Enzyme Kinetics (Local Process)

### The Physics

Enzyme catalysis occurs at the **active site**, a small region of ~10-20 residues. The rest of the protein provides structural support but doesn't directly participate in the chemical reaction.

This is a **local process** where overall protein size should NOT affect the turnover rate.

### Data

| Parameter | Value |
|-----------|-------|
| Enzymes | 69 |
| Length range | 124 - 3,300 residues |
| kcat range | 0.02 - 10⁶ s⁻¹ |
| EC classes | 6 (all major classes) |

### Sources

- BRENDA Database (comprehensive enzyme data)
- Bar-Even et al. 2011 (enzyme efficiency survey)
- Davidi et al. 2016 (natural enzyme turnover numbers)

### Results

| Category | n | α | R² | p |
|----------|---|---|---|---|
| **All enzymes** | 69 | **-0.9 ± 0.6** | 0.03 | 0.14 |

**The relationship is NOT statistically significant** (p > 0.05).

### By Enzyme Class

| EC Class | Name | n | α |
|----------|------|---|---|
| EC1 | Oxidoreductases | 16 | -2.9 |
| EC2 | Transferases | 16 | -0.8 |
| EC3 | Hydrolases | 18 | +4.2 |
| EC4 | Lyases | 11 | -2.3 |
| EC6 | Ligases | 5 | +0.1 |

The variability across classes and low overall R² confirms that **enzyme size does not predict catalytic speed**.

### Interpretation

- **α ≈ 0**: No significant size dependence
- Enzyme speed is determined by **active site chemistry**, not protein size
- Carbonic anhydrase (260 aa, kcat = 10⁶) vs RNA polymerase (3300 aa, kcat = 40)
- This confirms catalysis is a **local process**

---

## The Key RTM Insight

| Process Type | α | R² | What Matters |
|--------------|---|----|----|
| **GLOBAL** (Folding) | +7 | 0.6 | Geometry (entire chain) |
| **LOCAL** (Catalysis) | ~0 | ~0 | Chemistry (active site) |

### Why This Matters

RTM predicts that the scaling exponent α reflects the **nature of the process**:

1. **Global processes** (where the entire system must coordinate):
   - Strong size dependence
   - High α values
   - Example: Protein folding

2. **Local processes** (where only a small region is active):
   - No/weak size dependence  
   - α ≈ 0
   - Example: Enzyme catalysis

This distinction is a **testable RTM prediction** that the data confirm.

---

## Comparison to Original Analysis

| Metric | Original | This Analysis |
|--------|----------|---------------|
| Folding n | 41 | **84** |
| Folding α | 7.03 | **7.22 ± 0.62** |
| Enzyme analysis | Not included | **n=69, α≈0 (NS)** |
| Process comparison | No | **Yes** |

The expanded analysis:
1. Doubles the protein folding dataset
2. Adds enzyme kinetics as a contrasting analysis
3. Demonstrates the global vs local distinction

---

## Files Included

```
biochem_rtm_analysis/
├── analyze_biochemistry_rtm.py    # Main script (~400 lines)
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── output/                        # Generated results
    ├── biochemistry_rtm_analysis.png
    ├── biochemistry_rtm_analysis.pdf
    ├── protein_folding.csv        # 84 proteins
    └── enzyme_kinetics.csv        # 69 enzymes
```

---

## Extending the Analysis

### Add More Proteins

Edit `get_protein_folding_data()`:

```python
data = [
    # ... existing ...
    ("NewProtein", 75, 5.5, "Two-state", "Alpha", "NewRef 2024"),
    # (PDB, Length, ln_kf, Fold_Type, Structure, Reference)
]
```

### Add More Enzymes

Edit `get_enzyme_kinetics_data()`:

```python
data = [
    # ... existing ...
    ("New Enzyme", "1.1.1.999", 400, 150, "Substrate", "Organism", "Ref"),
    # (Name, EC, Length, kcat, Substrate, Organism, Reference)
]
```

---

## RTM Transport Classes

| α Range | Class | Biochemistry Example |
|---------|-------|---------------------|
| ~0 | Local/Chemistry-limited | Enzyme catalysis |
| 0.5 | Diffusive | Substrate binding (some cases) |
| 1-2 | Ballistic | Molecular motors |
| **5-10** | **Highly coherent** | **Protein folding** |

Protein folding's high α reflects the extreme cooperativity of the folding funnel.

---

## Key References

### Protein Folding
- Ivankov DN, Plaxco KW (2003) Contact order revisited. Protein Sci 12:1809
- Maxwell KL et al. (2005) Protein folding. Protein Sci 14:602
- Kubelka J et al. (2004) The protein folding speed limit. Curr Opin Struct Biol 14:76

### Enzyme Kinetics
- BRENDA: The Comprehensive Enzyme Information System. brenda-enzymes.org
- Bar-Even A et al. (2011) The moderately efficient enzyme. Biochemistry 50:4402
- Davidi D et al. (2016) A bird's-eye view of enzyme evolution. Nat Chem Biol 12:975

---

## Conclusion

**RTM successfully distinguishes two biochemical process types:**

1. **Protein Folding (Global):** α = +7.2, R² = 0.63
   - Strong size dependence
   - Entire chain must coordinate
   - "Folding funnel" creates coherent dynamics

2. **Enzyme Kinetics (Local):** α ≈ 0, R² ≈ 0
   - No size dependence
   - Only active site matters
   - Chemistry, not geometry, dominates

This contrast validates RTM's ability to characterize processes through scaling exponents.

---

## Citation

```bibtex
@misc{rtm_biochemistry_2026,
  author       = {RTM Research},
  title        = {RTM Biochemistry: Global vs Local Process Scaling},
  year         = {2026},
  note         = {Data: Literature compilation}
}
```

---

## License

CC BY 4.0. Data compiled from published literature.
