import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
# File provided: 'jwst_galaxies_data.txt'
# Structure: GalaxyID, Redshift_z, Log_Stellar_Mass_Msun
FILE_PATH = "jwst_galaxies_data.txt"

def analyze_cosmo_rtm():
    print("Loading JWST High-z Galaxy dataset...")
    try:
        # Try tab then whitespace
        df = pd.read_csv(FILE_PATH, sep='\t')
        if df.shape[1] == 1:
             df = pd.read_csv(FILE_PATH, sep='\s+')
    except FileNotFoundError:
        print("Error: 'jwst_galaxies_data.txt' not found.")
        return

    # 2. Cosmology Setup (Planck 2018 simplified)
    # We need to calculate the standard available time t(z)
    # Approximation for Matter Dominated era (valid for z > 5)
    # Age(z) propto (1+z)^(-1.5)
    
    # We calibrate "Standard Limit" using a non-anomalous galaxy: JADES-GS-z13-0
    # Mass ~ 10^8 at z=13.2 is considered "normal" / "possible".
    base_z = 13.2
    base_mass = 10**8.0
    
    # Function for relative age factor
    def get_age_factor(z):
        return (1 + z)**(-1.5)

    base_age_factor = get_age_factor(base_z)

    # 3. Calculate "Impossible" Excess
    # Standard Limit Mass at z = Base Mass * (Time(z) / Time(base))
    # This assumes linear growth with time (constant SFR).
    df['Standard_Limit_Mass'] = base_mass * (get_age_factor(df['Redshift_z']) / base_age_factor)
    df['Observed_Mass'] = 10**df['Log_Stellar_Mass_Msun']
    
    # Acceleration Factor A = Obs / Limit
    df['Acceleration_Factor'] = df['Observed_Mass'] / df['Standard_Limit_Mass']
    
    # 4. Solve for Alpha
    # RTM Prediction: Effective Time ~ Standard Time * (1+z)^(1.5 * alpha)
    # So Acceleration A ~ (1+z)^(1.5 * alpha)
    # alpha = ln(A) / (1.5 * ln(1+z))
    
    df['Required_Alpha'] = np.log(df['Acceleration_Factor']) / (1.5 * np.log(1 + df['Redshift_z']))
    
    # Filter for the anomalies (A > 1)
    anomalies = df[df['Acceleration_Factor'] > 1.0].copy()
    
    print(f"\n--- RTM RESULTS ---")
    print(f"Mean Required Alpha: {anomalies['Required_Alpha'].mean():.4f}")
    print(f"Max Required Alpha: {anomalies['Required_Alpha'].max():.4f} ({anomalies.loc[anomalies['Required_Alpha'].idxmax(), 'GalaxyID']})")
    
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(anomalies['Redshift_z'], anomalies['Required_Alpha'], color='darkred', s=100, label='JWST Anomalies')
    
    # Mean line
    mean_val = anomalies['Required_Alpha'].mean()
    plt.axhline(mean_val, color='black', linestyle='--', label=f'Mean α = {mean_val:.2f}')
    
    # Theoretical Zone
    plt.axhspan(1.0, 1.5, color='orange', alpha=0.1, label='RTM Theoretical Range (1.0-1.5)')
    
    # Annotate
    for i, row in anomalies.iterrows():
        plt.annotate(row['GalaxyID'], (row['Redshift_z'], row['Required_Alpha']), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('RTM-Cosmology: The "Alpha" of Impossible Galaxies')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Required Coherence Exponent (Alpha)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('cosmo_rtm_validation.png')
    print("Chart saved to cosmo_rtm_validation.png")

if __name__ == "__main__":
    analyze_cosmo_rtm.py()