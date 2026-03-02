import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
# File provided: 'hrv_aging_data.txt.txt'
FILE_PATH = "hrv_aging_data.txt.txt"

def analyze_homeostasis_rtm():
    print("Loading HRV Aging dataset...")
    try:
        # Check delimiter
        df = pd.read_csv(FILE_PATH, sep='\t')
        if df.shape[1] == 1:
            df = pd.read_csv(FILE_PATH, sep='\s+')
    except FileNotFoundError:
        print("Error: 'hrv_aging_data.txt.txt' not found.")
        return

    print(f"Loaded {len(df)} subjects.")
    print(df.groupby('Group')['DFA_Alpha_Coherence'].describe())

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    # Define order for logical progression
    order = ['Young_Healthy', 'Elderly_Healthy', 'Heart_Failure']
    palette = {"Young_Healthy": "tab:green", "Elderly_Healthy": "tab:orange", "Heart_Failure": "tab:red"}
    
    # Boxplot with Swarmplot overlay
    sns.boxplot(x='Group', y='DFA_Alpha_Coherence', data=df, order=order, palette=palette, showfliers=False)
    sns.swarmplot(x='Group', y='DFA_Alpha_Coherence', data=df, order=order, color=".25", size=8)
    
    # Add Reference Lines
    plt.axhline(1.0, color='green', linestyle='--', label='Optimal 1/f (Pink Noise)')
    plt.axhline(0.5, color='red', linestyle='--', label='Uncorrelated (White Noise)')
    
    plt.title('RTM-Homeostasis: The Loss of Complexity with Age & Disease')
    plt.ylabel('Fractal Coherence (DFA Alpha)')
    plt.xlabel('Health Cohort')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    
    plt.savefig('homeostasis_rtm_validation.png')
    print("Chart saved to homeostasis_rtm_validation.png")

if __name__ == "__main__":
    analyze_homeostasis_rtm()