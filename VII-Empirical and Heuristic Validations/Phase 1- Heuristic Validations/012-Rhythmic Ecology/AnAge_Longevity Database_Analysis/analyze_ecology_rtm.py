import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data
# Download from: https://genomics.senescence.info/species/dataset.zip
FILE_PATH = "anage_data.txt"

def analyze_ecology_rtm():
    print("Loading AnAge database...")
    try:
        df = pd.read_csv(FILE_PATH, sep='\t', encoding='latin-1')
    except:
        df = pd.read_csv(FILE_PATH, sep='\t', encoding='utf-8')
    
    # 2. Filter Valid Data
    # We need Mass (g) and Longevity (yrs)
    # Filter for realistic values > 0
    df_clean = df.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)'])
    df_clean = df_clean[(df_clean['Body mass (g)'] > 0) & (df_clean['Maximum longevity (yrs)'] > 0)]
    
    # Log Transform
    df_clean['log_Mass'] = np.log10(df_clean['Body mass (g)'])
    df_clean['log_Longevity'] = np.log10(df_clean['Maximum longevity (yrs)'])
    
    # 3. Analyze by Class
    classes = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']
    colors = {'Mammalia': 'tab:red', 'Aves': 'tab:blue', 'Reptilia': 'tab:green', 'Amphibia': 'tab:orange'}
    
    plt.figure(figsize=(10, 7))
    
    # Plot background points
    plt.scatter(df_clean['log_Mass'], df_clean['log_Longevity'], alpha=0.1, color='gray', label='All Data')
    
    results = []
    
    for cls in classes:
        subset = df_clean[df_clean['Class'] == cls]
        if len(subset) > 10:
            slope, intercept, r, p, err = stats.linregress(subset['log_Mass'], subset['log_Longevity'])
            results.append({'Class': cls, 'Alpha': slope, 'N': len(subset)})
            
            # Plot Line
            x_range = np.linspace(subset['log_Mass'].min(), subset['log_Mass'].max(), 100)
            y_pred = slope * x_range + intercept
            label = f"{cls} (α={slope:.2f})"
            plt.plot(x_range, y_pred, color=colors.get(cls, 'black'), linewidth=2, label=label)
    
    plt.xlabel('Log10 Body Mass (g)')
    plt.ylabel('Log10 Max Longevity (yrs)')
    plt.title('Rhythmic Ecology: The Universal Clock (T ~ L^α)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('ecology_rtm_validation.png')
    print("Analysis saved to ecology_rtm_validation.png")
    
    # Print Table
    res_df = pd.DataFrame(results)
    print("\n--- RTM SCALING RESULTS ---")
    print(res_df)

if __name__ == "__main__":
    analyze_ecology_rtm()