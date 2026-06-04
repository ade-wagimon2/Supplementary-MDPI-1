import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ============================================================================
# STYLE SETTINGS (Nature-style publication quality)
# ============================================================================
def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.titlesize': 22,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.frameon': True,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

SYSTEM_COLORS = {
    'PSY': '#2E86AB',    # Professional Blue
    'CrtM': '#F18F01',   # Professional Orange
    'GGPP': '#34495E',   # Muted Navy
    'FSPP': '#A23B72'    # Muted Burgundy/Magenta
}

INTERACTION_COLORS = {
    'Hydrophobic': '#45B39D',
    'HBAcceptor': '#7D3C98',
    'Anionic': '#AF601A',
    'VdWContact': '#7B7D7D',
    'HBDonor': '#117A65'
}

# ============================================================================
# AMINO ACID CONVERSION
# ============================================================================
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HIE': 'H', 'HID': 'H', 'HIP': 'H', 'CYX': 'C',  # common variants
}

INTERACTION_SHORT = {
    'Hydrophobic': 'Hph',
    'HBAcceptor': 'HBA',
    'HBDonor': 'HBD',
    'Anionic': 'Ani',
    'Cationic': 'Cat',
    'VdWContact': 'VdW',
    'PiStacking': 'PiS',
    'PiCation': 'PiC',
}

def convert_to_one_letter(res_str):
    """Convert three-letter residue label (e.g., ALA140) to one-letter (e.g., A140)."""
    match = re.match(r'([A-Za-z]+)(\d+)', str(res_str))
    if match:
        name, num = match.group(1).upper(), match.group(2)
        return AA_3TO1.get(name, name) + num
    return str(res_str)

def shorten_interaction(interaction):
    """Shorten interaction type name."""
    return INTERACTION_SHORT.get(interaction, interaction)

# ============================================================================
# DATA HANDLING
# ============================================================================

def get_res_num(res_str):
    """Extract residue number from residue string."""
    match = re.search(r'\d+', str(res_str))
    return int(match.group()) if match else 0

def make_label(protein_col, interaction_col):
    """Create compact residue-interaction label using one-letter AA codes."""
    return protein_col.apply(convert_to_one_letter) + ' | ' + interaction_col.apply(shorten_interaction)

def plot_fingerprint(data_path, system_name, output_path):
    """
    Generate a fingerprint interaction plot for a given system comparing FSPP and GGPP.
    """
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    # Rescale from 0-200 to 0-100%
    df['GGPP'] = df['GGPP'] / 2.0
    df['FSPP'] = df['FSPP'] / 2.0
    
    # Create a unique identifier for each residue-interaction pair (one-letter AA)
    df['Res-Int'] = make_label(df['protein'], df['interaction'])
    
    # Sort by residue number if possible
    df['res_num'] = df['protein'].apply(get_res_num)
    df = df.sort_values(by=['res_num', 'interaction'])
    
    # Melt the dataframe for seaborn plotting
    df_melted = df.melt(id_vars=['Res-Int', 'interaction', 'res_num'], 
                        value_vars=['GGPP', 'FSPP'], 
                        var_name='Ligand', 
                        value_name='Frequency (%)')

    # Setup figure
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Create grouped bar plot
    sns.barplot(
        data=df_melted, 
        x='Res-Int', 
        y='Frequency (%)', 
        hue='Ligand', 
        palette={'GGPP': SYSTEM_COLORS['GGPP'], 'FSPP': SYSTEM_COLORS['FSPP']},
        alpha=0.9,
        edgecolor='black',
        linewidth=0.8,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(f'Binding Site Interaction Fingerprint: {system_name}', pad=20)
    ax.set_xlabel('Residue | Interaction', fontsize=16, labelpad=12)
    ax.set_ylabel('Interaction Frequency (%)', fontsize=16, labelpad=12)
    ax.set_ylim(0, 105)
    
    # FIX: Better label rotation and selective display
    n_labels = len(ax.get_xticklabels())
    if n_labels > 20:
        # Show only every nth label if too many
        step = max(1, n_labels // 15)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
    
    plt.xticks(rotation=90, ha='right', fontsize=16)
    
    # Add grid lines for clarity
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    
    # Legend
    ax.legend(title='Ligand', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Final layout adjustments
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_interaction_panel(base_path, system_name, output_path):
    """
    Generate a 3-panel figure: Common, Unique to GGPP, and Unique to FSPP interactions.
    Optimized for publication (Vertical stacking).
    """
    setup_style()
    # FIX: Increased figure size for better spacing
    fig, axes = plt.subplots(3, 1, figsize=(10, 16), sharex=False)
    
    files = {
        'Common Interactions': 'common_interactions.csv',
        'Unique to GGPP': 'unique_to_GGPP.csv',
        'Unique to FSPP': 'unique_to_FSPP.csv'
    }

    for i, (title, filename) in enumerate(files.items()):
        ax = axes[i]
        csv_path = os.path.join(base_path, filename)
        
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'File not found:\n{filename}', ha='center', va='center')
            ax.set_title(title)
            continue
            
        df = pd.read_csv(csv_path)
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            continue

        # Rescale from 0-200 to 0-100%
        if 'GGPP' in df.columns:
            df['GGPP'] = df['GGPP'] / 2.0
        if 'FSPP' in df.columns:
            df['FSPP'] = df['FSPP'] / 2.0

        df['Res-Int'] = make_label(df['protein'], df['interaction'])
        df['res_num'] = df['protein'].apply(get_res_num)
        df = df.sort_values(by=['res_num', 'interaction'])
        
        if title == 'Common Interactions':
            df_melted = df.melt(id_vars=['Res-Int'], value_vars=['GGPP', 'FSPP'], 
                                var_name='Ligand', value_name='Frequency (%)')
            sns.barplot(data=df_melted, x='Res-Int', y='Frequency (%)', hue='Ligand', 
                        palette={'GGPP': SYSTEM_COLORS['GGPP'], 'FSPP': SYSTEM_COLORS['FSPP']},
                        alpha=0.9, edgecolor='black', linewidth=0.8, ax=ax)
            ax.legend(title='Ligand', fontsize=16, title_fontsize=16, loc='upper right')
        else:
            ligand = 'GGPP' if 'GGPP' in title else 'FSPP'
            sns.barplot(data=df, x='Res-Int', y=ligand, color=SYSTEM_COLORS[ligand],
                        alpha=0.9, edgecolor='black', linewidth=0.8, ax=ax)
            ax.set_title(f"{title} ({ligand})", fontsize=16)

        ax.set_title(f"{i+1}. {title}", loc='left', fontsize=16, fontweight='bold')
        ax.set_xlabel('Residue | Interaction', fontsize=16, labelpad=10)
        ax.set_ylabel('Interaction Frequency (%)', fontsize=16, labelpad=10)
            
        ax.set_ylim(0, 105)
        # FIX: Increased label rotation and padding
        ax.tick_params(axis='x', rotation=90, labelsize=16, pad=10)
        ax.yaxis.grid(True, linestyle='--', alpha=.25)

    plt.suptitle(f'Comparative Interaction Analysis: {system_name}', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Vertical Panel Plot: {output_path}")

def plot_combined_interaction_panel(base_dir, output_path):
    """
    Generate a 3x2 figure: 
    Rows: Common, Unique to GGPP, Unique to FSPP
    Columns: PSY, CrtM
    Nature Publication Ready with A, B, C labels.
    """
    setup_style()
    # Nature style settings
    plt.rcParams.update({
        'font.size': 18, 
        'axes.titlesize': 18, 
        'axes.labelsize': 18,
        'legend.fontsize': 18
    })
    
    # FIX 1: Increased figure width significantly for more label space
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharey=True)
    
    systems = ['PSY', 'CrtM']
    categories = [
        ('Common Interactions', 'common_interactions.csv'),
        ('Unique to GGPP', 'unique_to_GGPP.csv'),
        ('Unique to FSPP', 'unique_to_FSPP.csv')
    ]
    
    row_labels = ['A', 'B', 'C']

    for col_idx, system in enumerate(systems):
        system_dir = os.path.join(base_dir, system)
        
        for row_idx, (cat_name, filename) in enumerate(categories):
            ax = axes[row_idx, col_idx]
            csv_path = os.path.join(system_dir, filename)
            
            # Add Panel Labels (A, B, C) with better positioning
            if col_idx == 0:
                ax.text(-0.15, 1.15, row_labels[row_idx], transform=ax.transAxes, 
                        fontsize=18, fontweight='bold', va='top', ha='right')
            
            if not os.path.exists(csv_path):
                ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=9)
                continue
                
            df = pd.read_csv(csv_path)
            if df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=9)
                continue

            # Rescale from 0-200 to 0-100%
            if 'GGPP' in df.columns:
                df['GGPP'] = df['GGPP'] / 2.0
            if 'FSPP' in df.columns:
                df['FSPP'] = df['FSPP'] / 2.0

            # Filtering: Only show high frequency interactions (>50%) for Common
            if 'Common' in cat_name:
                df = df[(df['GGPP'] >= 25.0) | (df['FSPP'] >= 25.0)]

            # FIX 2: Limit number of labels shown to prevent overcrowding
            if len(df) > 15:
                sort_col = 'GGPP' if 'GGPP' in cat_name else 'FSPP' if 'FSPP' in cat_name else 'GGPP'
                df = df.nlargest(15, sort_col)

            df['Res-Int'] = make_label(df['protein'], df['interaction'])
            df['res_num'] = df['protein'].apply(get_res_num)
            df = df.sort_values(by=['res_num', 'interaction'])
            
            if 'Common' in cat_name:
                df_melted = df.melt(id_vars=['Res-Int'], value_vars=['GGPP', 'FSPP'], 
                                    var_name='Ligand', value_name='Frequency (%)')
                sns.barplot(data=df_melted, x='Res-Int', y='Frequency (%)', hue='Ligand', 
                            palette={'GGPP': SYSTEM_COLORS['GGPP'], 'FSPP': SYSTEM_COLORS['FSPP']},
                            alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax)
                
                if col_idx == 0 and row_idx == 0:
                    ax.legend(title='', fontsize=10, loc='upper left', bbox_to_anchor=(0.0, 1.25), 
                              ncol=2, frameon=False)
                else:
                    legend = ax.get_legend()
                    if legend: legend.remove()
            else:
                ligand = 'GGPP' if 'GGPP' in cat_name else 'FSPP'
                sns.barplot(data=df, x='Res-Int', y=ligand, color=SYSTEM_COLORS[ligand],
                            alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax)
            
            # Refined titles with more pad
            title_text = f"{cat_name}" if row_idx > 0 else f"{system}: {cat_name}"
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Residue | Interaction', fontsize=16, labelpad=10)
            
            if col_idx == 0:
                ax.set_ylabel('Interaction Frequency (%)', fontsize=16, labelpad=10)
            else:
                ax.set_ylabel('')
                
            ax.set_ylim(0, 105)
            
            # FIX 3: Better label rotation and spacing
            ax.tick_params(axis='x', rotation=90, labelsize=16, pad=5)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.3)
            sns.despine(ax=ax)
            
            # FIX 4: Align labels to right for better readability
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
                label.set_rotation_mode('anchor')

    plt.suptitle('Global Comparative Interaction Profile: PSY vs CrtM', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # FIX 5: Increased spacing to prevent all overlaps
    plt.subplots_adjust(hspace=0.8, wspace=0.3, top=0.92, bottom=0.08)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=600) 
    plt.close()
    print(f"Saved Refined Nature Figure: {output_path}")

def run_analysis():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    systems = ['PSY', 'CrtM']
    
    # Generate individual plots
    for sys in systems:
        sys_dir = os.path.join(base_dir, sys)
        
        # Original Fingerprint Plot
        high_freq_csv = os.path.join(sys_dir, "ligand_comparison_high_freq.csv")
        fingerprint_out = os.path.join(base_dir, f"{sys.lower()}_interaction_fingerprint.png")
        plot_fingerprint(high_freq_csv, sys, fingerprint_out)
        
        # New Panel Plot (Vertical)
        panel_out = os.path.join(base_dir, f"{sys.lower()}_interaction_panel.png")
        plot_interaction_panel(sys_dir, sys, panel_out)
    
    # Generate Combined Figure (Nature Ready)
    nature_out = os.path.join(base_dir, "nature_interaction_analysis.png")
    plot_combined_interaction_panel(base_dir, nature_out)

if __name__ == "__main__":
    run_analysis()