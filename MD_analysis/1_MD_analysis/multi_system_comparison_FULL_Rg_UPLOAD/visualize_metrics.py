import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
from matplotlib.patches import Patch
import matplotlib.ticker as ticker


# Nature-style colors
publication_colors = {
    'psy': '#48C9B0',          # PSY-FSPP color (light blue-green)
    'psy_dark': '#2980b9',     # PSY-GGPP color (darker blue)
    'crtm': '#F18F01',         # CrtM-FSPP color (orange)
    'crtm_dark': '#d35400',    # CrtM-GGPP color (darker orange)
}

# Apply user-requested Matplotlib styling
plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'         : 14,
    'axes.labelsize'    : 16,
    'axes.titlesize'    : 16,
    'axes.titleweight'  : 'bold',
    'xtick.labelsize'   : 12,
    'ytick.labelsize'   : 12,
    'legend.fontsize'   : 12,
    'legend.framealpha' : 0.92,
    'legend.edgecolor'  : '0.75',
    'axes.linewidth'    : 1.0,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'xtick.direction'   : 'out',
    'ytick.direction'   : 'out',
    'grid.linestyle'    : ':',
    'grid.linewidth'    : 0.6,
    'grid.alpha'        : 0.50,
    'grid.color'        : '#aaaaaa',
    'lines.linewidth'   : 1.8,
    'patch.linewidth'   : 0.8,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype'      : 42,
    'ps.fonttype'       : 42
})

def save_publication_figure(fig, base_name):
    """Saves figure in multiple formats (PNG and PDF)."""
    for ext in ['.png', '.pdf']:
        path = base_name + ext
        fig.savefig(path)

def process_directory(directory, palette):
    """Processes a single directory and returns normalized aggregated data."""
    print(f"Processing directory: {directory}")
    
    patterns = {
        "Replica 1": os.path.join(directory, "MD_main_metrics.csv"),
        "Replica 2": os.path.join(directory, "MD2_main_metrics.csv"),
        "Replica 3": os.path.join(directory, "MD3_main_metrics.csv")
    }
    
    dfs = []
    labels = []
    for label, path in patterns.items():
        if os.path.exists(path):
            try:
                dfs.append(pd.read_csv(path))
                labels.append(label)
            except Exception:
                pass
    
    if not dfs:
        return None

    # Use LaTeX for Angstrom symbol
    angstrom = r"($\mathrm{\AA}$)"
    
    columns = dfs[0].columns
    ligand_columns = [c for c in columns if '_Lig' in c and 'RMSD' in c]
    
    metrics_to_plot = [
        ('Protein_RMSD', f'Protein RMSD {angstrom}'), 
        ('Rg', f'Radius of Gyration {angstrom}')
    ]
    for lig_col in ligand_columns:
        metrics_to_plot.append((lig_col, f'{lig_col.replace("_", " ")} {angstrom}'))

    # Generate individual system multi-panel figure
    num_panels = len(metrics_to_plot)
    cols = 2
    rows = (num_panels + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows), constrained_layout=True)
    axes = axes.flatten() if num_panels > 1 else [axes]
    
    colors = sns.color_palette("muted", len(dfs))
    for i, (col_name, ylabel) in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.grid(True)
        
        for df, label, color in zip(dfs, labels, colors):
            if col_name in df.columns:
                window = 50
                ax.plot(df['Frame'], df[col_name], color=color, alpha=0.15, linewidth=0.5, label='_nolegend_')
                ax.plot(df['Frame'], df[col_name].rolling(window=window, min_periods=1).mean(), 
                        color=color, alpha=0.95, label=label)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel('Frame', fontweight='bold')
        # Place panel label outside the grid with consistent positioning
        ax.annotate(string.ascii_uppercase[i], xy=(0, 1), xycoords='axes fraction',
                   xytext=(-25, 10), textcoords='offset points',
                   fontsize=16, fontweight='bold', va='baseline', ha='right')
        # Center the descriptive title
        ax.set_title(ylabel.split(" (")[0].strip(), loc='center', fontweight='bold')
        if i == 0: ax.legend(loc='upper left', frameon=True)

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    
    save_publication_figure(fig, os.path.join(directory, 'summary_metrics_publication'))
    plt.close()

    # Normalize timeseries data for global summary
    system_name = os.path.basename(directory)
    
    # Load and aggregate RMSF data
    rmsf_patterns = {
        "Replica 1": os.path.join(directory, "MD_rmsf.csv"),
        "Replica 2": os.path.join(directory, "MD2_rmsf.csv"),
        "Replica 3": os.path.join(directory, "MD3_rmsf.csv")
    }
    
    rmsf_dfs = []
    for path in rmsf_patterns.values():
        if os.path.exists(path):
            try:
                rmsf_dfs.append(pd.read_csv(path))
            except Exception:
                pass
                
    aggregated_rmsf = None
    if rmsf_dfs:
        rmsf_combined = pd.concat(rmsf_dfs)
        rmsf_grouped = rmsf_combined.groupby('Residue')
        aggregated_rmsf = rmsf_grouped.mean().reset_index()
        aggregated_rmsf['RMSF_std'] = rmsf_grouped.std().reset_index()['RMSF_CA']
        aggregated_rmsf['System'] = system_name

    norm_dfs = []
    for df in dfs:
        temp_df = df.copy()
        rename_map = {}
        for col in temp_df.columns:
            if '_Lig1' in col and 'RMSD' in col: rename_map[col] = 'Lig1_RMSD'
            if '_Lig2' in col and 'RMSD' in col: rename_map[col] = 'Lig2_RMSD'
        temp_df.rename(columns=rename_map, inplace=True)
        cols_to_keep = ['Frame', 'Protein_RMSD', 'Rg', 'Lig1_RMSD', 'Lig2_RMSD']
        norm_dfs.append(temp_df[[c for c in cols_to_keep if c in temp_df.columns]])

    combined = pd.concat(norm_dfs)
    grouped = combined.groupby('Frame')
    system_summary = grouped.mean().reset_index()
    system_summary = system_summary.merge(grouped.std().reset_index(), on='Frame', suffixes=('', '_std'))
    system_summary['System'] = system_name
    
    return {
        'timeseries': system_summary,
        'rmsf': aggregated_rmsf
    }

def generate_global_summary(all_timeseries, all_rmsf):
    """Generates a professional multi-panel cross-system figure including RMSF."""
    print("Generating global summary figure with custom styling...")
    df_ts = pd.concat(all_timeseries)
    
    angstrom = r"($\mathrm{\AA}$)"
    metrics = [
        ('Protein_RMSD', f'Protein RMSD {angstrom}'),
        ('Rg', f'Radius of Gyration {angstrom}'),
        ('Lig1_RMSD', f'Ligand 1 RMSD {angstrom}'),
        ('Lig2_RMSD', f'Ligand 2 RMSD {angstrom}')
    ]
    
    available_metrics = [(m, l) for m, l in metrics if m in df_ts.columns]
    num_ts_panels = len(available_metrics)
    
    # We want to add RMSF as an additional panel
    has_rmsf = len(all_rmsf) > 0
    total_panels = num_ts_panels + (1 if has_rmsf else 0)
    
    cols = 2
    rows = num_ts_panels // cols + (1 if has_rmsf else 0)
    
    fig = plt.figure(figsize=(10, 3.5 * rows), constrained_layout=True)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(rows, cols, figure=fig)
    
    systems = df_ts['System'].unique()
    window = 50

    # Plot Timeseries Metrics (A-D)
    for i, (col_name, ylabel) in enumerate(available_metrics):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.grid(True)
        std_col = f'{col_name}_std'
        
        for system in systems:
            # Determine color based on system name (GGPP systems use darker shades)
            is_ggpp = 'GGPP' in system
            if 'PSY' in system:
                color = publication_colors['psy_dark'] if is_ggpp else publication_colors['psy']
            else:
                color = publication_colors['crtm_dark'] if is_ggpp else publication_colors['crtm']
            
            sys_data = df_ts[df_ts['System'] == system].sort_values('Frame')
            if col_name in sys_data.columns and not sys_data[col_name].isna().all():
                y_mean = sys_data[col_name].rolling(window=window, min_periods=1).mean()
                ax.plot(sys_data['Frame'], y_mean, label=system, color=color, alpha=1.0)
                
                if std_col in sys_data.columns:
                    y_std = sys_data[std_col].rolling(window=window, min_periods=1).mean()
                    ax.fill_between(sys_data['Frame'], y_mean - y_std, y_mean + y_std, 
                                   color=color, alpha=0.15, edgecolor='none')
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel('Frame', fontweight='bold')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        # Place panel label outside the grid with consistent positioning
        ax.annotate(string.ascii_uppercase[i], xy=(0, 1), xycoords='axes fraction',
                   xytext=(-25, 10), textcoords='offset points',
                   fontsize=16, fontweight='bold', va='baseline', ha='right')
        # Center the descriptive title
        ax.set_title(ylabel.split(" (")[0].strip(), loc='center', fontweight='bold')

    # Plot RMSF Panel (E) - Spans two columns
    if has_rmsf:
        idx = num_ts_panels
        ax = fig.add_subplot(gs[rows - 1, :])
        ax.grid(True)
        df_rmsf = pd.concat(all_rmsf)
        
        for system in systems:
            is_ggpp = 'GGPP' in system
            color = publication_colors['psy_dark'] if 'PSY' in system else publication_colors['crtm_dark']
            if not is_ggpp:
                color = publication_colors['psy'] if 'PSY' in system else publication_colors['crtm']
            
            sys_rmsf = df_rmsf[df_rmsf['System'] == system].sort_values('Residue')
            if not sys_rmsf.empty:
                ax.plot(sys_rmsf['Residue'], sys_rmsf['RMSF_CA'], color=color, label=system, alpha=0.9)
                if 'RMSF_std' in sys_rmsf.columns:
                    ax.fill_between(sys_rmsf['Residue'], 
                                   sys_rmsf['RMSF_CA'] - sys_rmsf['RMSF_std'],
                                   sys_rmsf['RMSF_CA'] + sys_rmsf['RMSF_std'],
                                   color=color, alpha=0.15, edgecolor='none')
        
        ax.set_ylabel(f'RMSF {angstrom}', fontweight='bold')
        ax.set_xlabel('Residue Number', fontweight='bold')
        # Place panel label outside the grid with consistent positioning
        ax.annotate(string.ascii_uppercase[idx], xy=(0, 1), xycoords='axes fraction',
                   xytext=(-25, 10), textcoords='offset points',
                   fontsize=16, fontweight='bold', va='baseline', ha='right')
        ax.set_title('RMSF Profile', loc='center', fontweight='bold')

    # Place a shared legend (use the first axis to get handles)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    
    # Add shadow legend entry
    shadow_patch = Patch(color='gray', alpha=0.3, label='Std. Dev.')
    handles.append(shadow_patch)
    labels.append('Std. Dev.')
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05 / rows * 3.5), 
               ncol=min(len(systems) + 1, 5), frameon=True, fontsize=14, edgecolor='0.75')

    # Remove any empty subplots
    # Not strictly needed with GridSpec if we assign explicitly, but kept for robustness
    # (Though fig.axes might include deleted ones if we subplotted differently)

    save_publication_figure(fig, 'global_systems_timeseries')
    plt.close()
    print("  Generated final publication-quality figures.")

def main():
    base_dir = "."
    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    palette = sns.color_palette("colorblind")
    
    all_ts = []
    all_rmsf = []
    for subdir in subdirs:
        if subdir.startswith('.') or subdir.startswith('_'): continue
        res = process_directory(os.path.join(base_dir, subdir), palette)
        if res is not None:
            all_ts.append(res['timeseries'])
            if res['rmsf'] is not None:
                all_rmsf.append(res['rmsf'])
    
    if all_ts:
        generate_global_summary(all_ts, all_rmsf)

if __name__ == "__main__":
    main()
