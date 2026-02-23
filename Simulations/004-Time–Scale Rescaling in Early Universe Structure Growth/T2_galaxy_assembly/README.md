# T2: Galaxy Assembly - Required Acceleration Calculator

From "Time–Scale Rescaling in Early Universe Structure Growth"

## Overview
Calculates the required RTM acceleration factor A to reach a target stellar mass at high redshift.

## Key Equation
M_star = f_b × M_halo × [1 - (1-ε)^(A×N_dyn)]

Required: A = ln[1 - M_star/(f_b×M_halo)] / [N_dyn × ln(1-ε)]

## Usage
```bash
python T2_galaxy_assembly.py
```

## Key Results
With ε ~ 2% and A ~ 30-60, RTM naturally explains "too-early/too-massive" galaxies at z > 10.

## License
CC BY 4.0
