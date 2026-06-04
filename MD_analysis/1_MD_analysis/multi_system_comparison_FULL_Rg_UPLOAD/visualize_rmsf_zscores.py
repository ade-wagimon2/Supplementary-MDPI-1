import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Arial', 'Arial', 'DejaVu Sans'],
    'font.size'         : 18,
    'axes.labelsize'    : 20,
    'axes.titlesize'    : 20,
    'axes.titleweight'  : 'bold',
    'xtick.labelsize'   : 18,
    'ytick.labelsize'   : 18,
    'legend.fontsize'   : 18,
    'axes.linewidth'    : 1.0,
    'grid.linestyle'    : '--',
    'grid.linewidth'    : 0.5,
    'grid.alpha'        : 0.3,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight'
})

SYSTEMS = [
    "PSY-FSPP",
    "PSY-GGPP",
    "CrtM-FSPP",
    "CrtM-GGPP"
]

RMSF_FILES = ["MD_rmsf.csv", "MD2_rmsf.csv", "MD3_rmsf.csv"]

def load_and_aggregate_rmsf(system_dir):
    """Loads all RMSF replicas and calculates the mean per residue."""
    dfs = []
    for f in RMSF_FILES:
        path = os.path.join(system_dir, f)
        if os.path.exists(path):
            try:
                dfs.append(pd.read_csv(path))
            except Exception as e:
                print(f"  ⚠️ Error loading {path}: {e}")
    
    if not dfs:
        return None
        
    combined = pd.concat(dfs)
    # Group by Residue and take the mean
    aggregated = combined.groupby('Residue')['RMSF_CA'].mean().reset_index()
    return aggregated

def calculate_zscores(df):
    """Calculates Z-scores for RMSF values within a system."""
    rmsf = df['RMSF_CA']
    mean_val = rmsf.mean()
    std_val = rmsf.std()
    df['Z_Score'] = (rmsf - mean_val) / std_val
    return df

def generate_rmsf_zscore_plot():
    print("Generating Residue Flexibility Z-Score Plot...")
    
    fig, axes = plt.subplots(len(SYSTEMS), 1, figsize=(8, 14), sharex=True, constrained_layout=True)
    
    # Storage for legend handles
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='$Z = +2$ (High Flexibility)'),
        Line2D([0], [0], color='blue', linestyle='--', label='$Z = -2$ (Low Flexibility)'),
        Patch(facecolor='red', edgecolor='none', label='Flexible Residue'),
        Patch(facecolor='grey', edgecolor='none', label='Normal Range'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.0, label='$Z = 0$ (Baseline)')
    ]

    for i, system in enumerate(SYSTEMS):
        ax = axes[i]
        print(f"  Processing {system}...")
        
        # Load data
        data = load_and_aggregate_rmsf(system)
        if data is None:
            ax.text(0.5, 0.5, f"No data for {system}", transform=ax.transAxes, ha='center')
            continue
            
        # Calculate Z-scores
        data = calculate_zscores(data)
        
        # Plot bars
        # Color based on Z-score threshold
        colors = ['red' if z > 2 else 'grey' for z in data['Z_Score']]
        ax.bar(data['Residue'], data['Z_Score'], color=colors, width=1.0, alpha=0.8)
        
        # Add horizontal lines
        ax.axhline(0, color='black', linestyle='-', linewidth=1.0, zorder=1)
        ax.axhline(2, color='red', linestyle='--', linewidth=1.5, zorder=2)
        ax.axhline(-2, color='blue', linestyle='--', linewidth=1.5, zorder=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}$"))
        
        # Formatting
        ax.set_title(f"{system}", fontweight='bold')
        ax.set_ylabel("RMSF Z-Score", fontweight='bold')
        ax.set_ylim(-4, 6)
        ax.grid(True, axis='y')
        
        # X-axis label only for bottom plot
        if i == len(SYSTEMS) - 1:
            ax.set_xlabel("Residue Number", fontweight='bold')

    # Add shared legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.55, -0.06), 
               ncol=3, frameon=True, fontsize=12)

    # Save
    output_name = "rmsf_zscore_comparison"
    for ext in ['.png', '.pdf']:
        fig.savefig(output_name + ext, bbox_inches='tight')
    
    print(f"Plot saved to {output_name}.png and .pdf")
    plt.close()

if __name__ == "__main__":
    generate_rmsf_zscore_plot()
