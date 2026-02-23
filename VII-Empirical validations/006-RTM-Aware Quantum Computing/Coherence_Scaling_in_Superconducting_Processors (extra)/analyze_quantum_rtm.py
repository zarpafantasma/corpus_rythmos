import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data
# File provided: 'ibm_quantum_data.txt'
FILE_PATH = "ibm_quantum_data.txt"

def analyze_quantum_rtm():
    print("Loading IBM Quantum dataset...")
    try:
        # Read file (skipping the 'Plaintext' header line if present)
        # Try tab separation first
        df = pd.read_csv(FILE_PATH, sep='\t', skiprows=1)
        if 'Qubits_N' not in df.columns:
             # Try whitespace separation
             df = pd.read_csv(FILE_PATH, sep='\s+', skiprows=1)
    except FileNotFoundError:
        print("Error: 'ibm_quantum_data.txt' not found.")
        return

    print(f"Loaded {len(df)} processors.")
    print(df)

    # 2. RTM Scaling Analysis
    # Hypothesis: T1 ~ N^alpha
    # Log-Log Transform
    df['ln_N'] = np.log(df['Qubits_N'])
    df['ln_T1'] = np.log(df['Avg_T1_us'])
    
    # 3. Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['ln_N'], df['ln_T1'])
    alpha = slope
    
    print(f"\n--- RTM RESULTS ---")
    print(f"Slope (Alpha): {alpha:.4f}")
    print(f"Correlation (r): {r_value:.4f}")
    print("Interpretation: Positive alpha means Coherence improves with Scale (Good Engineering).")
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ln_N'], df['ln_T1'], color='tab:purple', s=100, label='IBM Processors (2017-2023)')
    
    # Annotate points
    for i, row in df.iterrows():
        plt.annotate(row['Processor_Family'], (row['ln_N'], row['ln_T1']), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot Fit Line
    x_range = np.linspace(df['ln_N'].min(), df['ln_N'].max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(x_range, y_pred, color='tab:orange', linewidth=2, linestyle='--', label=f'RTM Scaling (α={alpha:.2f})')
    
    plt.title('RTM-Quantum: The Scaling of Coherence Time')
    plt.xlabel('Log Qubits (ln N)')
    plt.ylabel('Log Avg T1 Time (ln T1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('quantum_rtm_validation.png')
    print("Chart saved to quantum_rtm_validation.png")

if __name__ == "__main__":
    analyze_quantum_rtm()