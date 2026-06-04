"""
Hydrophobicity Analysis and Visualization Script
Analyzes protein sequences from FASTA files using multiple hydrophobicity scales
and generates comprehensive visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HYDROPHOBICITY SCALES
# ============================================================================

# Kyte-Doolittle scale (most commonly used)
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    '-': 0.0, 'X': 0.0, 'B': -3.5, 'Z': -3.5, 'U': 2.5
}

# Hopp-Woods scale (for surface accessibility)
HOPP_WOODS = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
    'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
    'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
    'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5,
    '-': 0.0, 'X': 0.0, 'B': 1.6, 'Z': 1.6, 'U': -1.0
}

# Eisenberg scale (consensus scale)
EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
    '-': 0.0, 'X': 0.0, 'B': -0.84, 'Z': -0.80, 'U': 0.29
}

# Rose scale (based on mean fractional area loss)
ROSE = {
    'A': 0.74, 'R': 0.64, 'N': 0.63, 'D': 0.62, 'C': 0.91,
    'Q': 0.62, 'E': 0.62, 'G': 0.72, 'H': 0.78, 'I': 0.88,
    'L': 0.85, 'K': 0.52, 'M': 0.85, 'F': 0.88, 'P': 0.64,
    'S': 0.66, 'T': 0.70, 'W': 0.85, 'Y': 0.76, 'V': 0.86,
    '-': 0.0, 'X': 0.0, 'B': 0.63, 'Z': 0.62, 'U': 0.91
}

# Janin scale (interior vs. surface)
JANIN = {
    'A': 0.74, 'R': -1.01, 'N': -0.82, 'D': -1.05, 'C': 0.91,
    'Q': -0.85, 'E': -1.14, 'G': 0.72, 'H': 0.17, 'I': 1.39,
    'L': 1.30, 'K': -1.62, 'M': 0.91, 'F': 1.38, 'P': -0.36,
    'S': -0.04, 'T': -0.04, 'W': 1.00, 'Y': 0.42, 'V': 1.22,
    '-': 0.0, 'X': 0.0, 'B': -0.94, 'Z': -1.00, 'U': 0.91
}

# Engelman (GES) scale - for membrane proteins
ENGELMAN_GES = {
    'A': 1.6, 'R': -12.3, 'N': -4.8, 'D': -9.2, 'C': 2.0,
    'Q': -4.1, 'E': -8.2, 'G': 1.0, 'H': -3.0, 'I': 3.1,
    'L': 2.8, 'K': -8.8, 'M': 3.4, 'F': 3.7, 'P': -0.2,
    'S': 0.6, 'T': 1.2, 'W': 1.9, 'Y': -0.7, 'V': 2.6,
    '-': 0.0, 'X': 0.0, 'B': -7.0, 'Z': -6.2, 'U': 2.0
}

# Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)
COWAN_WHITTAKER = {
    'A': 0.35, 'R': -1.50, 'N': -0.99, 'D': -2.15, 'C': 0.76,
    'Q': -0.93, 'E': -1.95, 'G': 0.00, 'H': -0.65, 'I': 1.83,
    'L': 1.80, 'K': -1.54, 'M': 1.10, 'F': 1.69, 'P': 0.84,
    'S': -0.22, 'T': 0.05, 'W': 1.35, 'Y': 0.60, 'V': 1.40,
    '-': 0.0, 'X': 0.0, 'B': -1.57, 'Z': -1.44, 'U': 0.76
}

SCALES = {
    'Kyte-Doolittle': KYTE_DOOLITTLE,
    'Hopp-Woods': HOPP_WOODS,
    'Eisenberg': EISENBERG,
    'Rose': ROSE,
    'Janin': JANIN,
    'Engelman (GES)': ENGELMAN_GES,
    'Cowan-Whittaker': COWAN_WHITTAKER
}

# ============================================================================
# FASTA PARSING
# ============================================================================

def parse_fasta(filepath: str) -> Dict[str, str]:
    """Parse a FASTA file and return a dictionary of sequences."""
    sequences = {}
    current_header = None
    current_sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                current_header = line[1:].split()[0]  # Take first part as ID
                current_sequence = []
            else:
                current_sequence.append(line.replace('\r', ''))
        
        if current_header:
            sequences[current_header] = ''.join(current_sequence)
    
    return sequences

# ============================================================================
# HYDROPHOBICITY CALCULATIONS
# ============================================================================

def calculate_hydrophobicity(sequence: str, scale: Dict[str, float]) -> List[float]:
    """Calculate hydrophobicity values for each position in a sequence."""
    return [scale.get(aa.upper(), 0.0) for aa in sequence]

def sliding_window_average(values: List[float], window_size: int = 9) -> List[float]:
    """Calculate sliding window average of hydrophobicity values."""
    if len(values) < window_size:
        return values
    
    result = []
    half_window = window_size // 2
    
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        window_values = [v for v in values[start:end] if v != 0.0]  # Exclude gaps
        if window_values:
            result.append(np.mean(window_values))
        else:
            result.append(0.0)
    
    return result

def calculate_consensus_hydrophobicity(sequences: Dict[str, str], scale: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """Calculate mean and std of hydrophobicity at each position across all sequences."""
    if not sequences:
        return [], []
    
    # Get alignment length
    seq_list = list(sequences.values())
    alignment_length = max(len(s) for s in seq_list)
    
    position_values = [[] for _ in range(alignment_length)]
    
    for seq in seq_list:
        hydro_values = calculate_hydrophobicity(seq, scale)
        for i, val in enumerate(hydro_values):
            if seq[i] != '-':  # Only include non-gap positions
                position_values[i].append(val)
    
    means = []
    stds = []
    
    for pos_vals in position_values:
        if pos_vals:
            means.append(np.mean(pos_vals))
            stds.append(np.std(pos_vals))
        else:
            means.append(0.0)
            stds.append(0.0)
    
    return means, stds

def calculate_hydrophobic_moments_single(sequence: str, scale: Dict[str, float], window_size: int = 11, angle: int = 100) -> List[float]:
    """Calculate hydrophobic moment for a single sequence using helical structure.
    
    Args:
        sequence: Amino acid sequence string
        scale: Hydrophobicity scale dictionary
        window_size: Window size for moment calculation
        angle: Rotation angle per residue (100 for alpha-helix, 160 for beta-sheet)
    """
    sequence = sequence.replace('-', '')
    if len(sequence) < window_size:
        return [0.0] * len(sequence)
    
    moments = []
    angle_rad = np.radians(angle)
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        sin_sum = 0.0
        cos_sum = 0.0
        
        for j, aa in enumerate(window):
            h = scale.get(aa.upper(), 0.0)
            sin_sum += h * np.sin(j * angle_rad)
            cos_sum += h * np.cos(j * angle_rad)
        
        moment = np.sqrt(sin_sum**2 + cos_sum**2) / window_size
        moments.append(moment)
    
    # Pad to match original length
    padding = (len(sequence) - len(moments)) // 2
    return [0.0] * padding + moments + [0.0] * (len(sequence) - len(moments) - padding)

def calculate_consensus_hydrophobic_moments(sequences: Dict[str, str], scale: Dict[str, float], 
                                             window_size: int = 11, angle: int = 100) -> Tuple[List[float], List[float]]:
    """Calculate mean and std of hydrophobic moments at each position across all sequences.
    
    Args:
        sequences: Dictionary of sequence IDs to sequences
        scale: Hydrophobicity scale dictionary
        window_size: Window size for moment calculation
        angle: Rotation angle per residue (100 for alpha-helix)
    
    Returns:
        Tuple of (mean moments, std moments) at each position
    """
    if not sequences:
        return [], []
    
    # Calculate moments for each sequence
    all_moments = []
    for seq in sequences.values():
        moments = calculate_hydrophobic_moments_single(seq, scale, window_size, angle)
        all_moments.append(moments)
    
    # Find the minimum length (since sequences may have different lengths after gap removal)
    min_len = min(len(m) for m in all_moments) if all_moments else 0
    max_len = max(len(m) for m in all_moments) if all_moments else 0
    
    # Calculate mean and std at each position
    means = []
    stds = []
    
    for i in range(max_len):
        pos_vals = [m[i] for m in all_moments if i < len(m)]
        if pos_vals:
            means.append(np.mean(pos_vals))
            stds.append(np.std(pos_vals))
        else:
            means.append(0.0)
            stds.append(0.0)
    
    return means, stds

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================

# Publication DPI (300 for print, 150 for screen)
PUBLICATION_DPI = 300

# Figure sizes optimized for journal columns (in inches)
FIGURE_SIZES = {
    'single_column': (3.5, 3.0),
    'double_column': (7.0, 5.0),
    'full_page': (7.5, 10.0),
    'wide': (10.0, 6.0),
    'square': (6.0, 6.0),
    'landscape': (12.0, 7.0),
    'portrait': (8.0, 10.0),
    'heatmap': (14.0, 8.0),
    'profile': (12.0, 6.0),
}

# Publication-quality color palette (colorblind-friendly)
PUBLICATION_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta  
    'tertiary': '#F18F01',     # Orange
    'accent1': '#00d9ff',      # Cyan
    'accent2': '#ff6b6b',      # Coral
    'neutral': '#6C757D',      # Gray
    'positive': '#3A7D44',     # Green
}

# Specific system colors
SYSTEM_COLORS = {
    'psy': '#3498db',          # PSY color (blue)
    'crtm': '#F18F01',         # CrtM color (orange)
}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def setup_style():
    """Configure matplotlib for Nature publication-quality figures."""
    plt.rcParams.update({
        'font.family'       : 'sans-serif',
        'font.sans-serif'   : ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size'         : 16,
        'axes.labelsize'    : 16,
        'axes.titlesize'    : 16,
        'axes.titleweight'  : 'bold',
        'xtick.labelsize'   : 16,
        'ytick.labelsize'   : 16,
        'legend.fontsize'   : 16,
        'legend.frameon'    : False,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.0,
        'axes.linewidth'    : 0.5,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'xtick.direction'   : 'out',
        'ytick.direction'   : 'out',
        'xtick.major.size'  : 3,
        'xtick.major.width' : 0.5,
        'ytick.major.size'  : 3,
        'ytick.major.width' : 0.5,
        'grid.linestyle'    : '-',
        'grid.linewidth'    : 0.3,
        'grid.alpha'        : 0.3,
        'grid.color'        : '#cccccc',
        'lines.linewidth'   : 1.0,
        'patch.linewidth'   : 0.5,
        'figure.dpi'        : 150,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
        'savefig.pad_inches': 0.02,
    })

def create_hydropathy_profile(sequences: Dict[str, str], scale_name: str, 
                               output_path: str, window_size: int = 9,
                               title_suffix: str = ""):
    """Create hydropathy profile plot with sliding window."""
    setup_style()
    
    scale = SCALES[scale_name]
    means, stds = calculate_consensus_hydrophobicity(sequences, scale)
    smoothed = sliding_window_average(means, window_size)
    smoothed_stds = sliding_window_average(stds, window_size)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=PUBLICATION_DPI)
    
    positions = range(1, len(means) + 1)
    smoothed_arr = np.array(smoothed)
    smoothed_stds_arr = np.array(smoothed_stds)
    
    # Plot STD shading (±1 SD)
    ax.fill_between(positions, smoothed_arr - smoothed_stds_arr, smoothed_arr + smoothed_stds_arr, 
                    alpha=0.2, color='#00d9ff', label='±1 SD')
    
    # Plot smoothed profile
    ax.fill_between(positions, smoothed, alpha=0.3, color='#00d9ff')
    ax.plot(positions, smoothed, color='#00d9ff', linewidth=2, label='Mean')
    
    # Add threshold line at 0
    ax.axhline(y=0, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.7)
    
    # Highlight hydrophobic regions
    smoothed_arr = np.array(smoothed)
    hydrophobic = smoothed_arr > 0
    for i, (pos, is_hydro) in enumerate(zip(positions, hydrophobic)):
        if is_hydro:
            ax.axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color='#ff9f43', zorder=0)
    
    ax.set_xlabel('Position in Alignment', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Hydrophobicity ({scale_name})', fontsize=12, fontweight='bold')
    ax.set_title(f'Hydropathy Profile - {scale_name} Scale{title_suffix}\n(Window size: {window_size})',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', facecolor='white', edgecolor='#333333')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_multi_scale_comparison(sequences: Dict[str, str], output_path: str,
                                   window_size: int = 9, title_suffix: str = ""):
    """Compare hydrophobicity profiles across multiple scales."""
    setup_style()
    
    n_scales = len(SCALES)
    n_cols = 2
    n_rows = int(np.ceil(n_scales / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), dpi=PUBLICATION_DPI)
    axes = axes.flatten()
    
    colors = ['#ff6b6b', '#00d9ff', '#ffd93d', '#6bcb77', '#a66ded', '#ff9f43', '#e056fd']
    
    for idx, (scale_name, scale) in enumerate(SCALES.items()):
        ax = axes[idx]
        
        means, stds = calculate_consensus_hydrophobicity(sequences, scale)
        smoothed = sliding_window_average(means, window_size)
        positions = range(1, len(means) + 1)
        
        ax.fill_between(positions, smoothed, alpha=0.3, color=colors[idx % len(colors)])
        ax.plot(positions, smoothed, color=colors[idx % len(colors)], linewidth=1.5)
        ax.axhline(y=0, color='#333333', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_title(scale_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Position', fontsize=9)
        ax.set_ylabel('Hydrophobicity', fontsize=9)
    
    # Hide any unused subplots
    for idx in range(n_scales, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Multi-Scale Hydrophobicity Comparison{title_suffix}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_hydrophobic_moments_plot(sequences: Dict[str, str], output_path: str,
                                     window_size: int = 11, angle: int = 100,
                                     title_suffix: str = ""):
    """Create hydrophobic moments plot using Cowan-Whittaker scale with mean ± STD.
    
    Hydrophobic moment measures the amphipathicity of a helix - high values indicate
    one face is hydrophobic and the other hydrophilic (common in membrane-interacting helices).
    
    Args:
        sequences: Dictionary of sequence IDs to sequences
        output_path: Path to save the output plot
        window_size: Window size for moment calculation (default 11 for alpha-helix)
        angle: Rotation angle per residue (100° for alpha-helix, 160° for beta-sheet)
        title_suffix: Optional suffix for plot title
    """
    setup_style()
    
    # Use Cowan-Whittaker scale as requested
    scale = COWAN_WHITTAKER
    scale_name = "Cowan-Whittaker (HPLC pH 7.5)"
    
    # Calculate consensus hydrophobic moments
    means, stds = calculate_consensus_hydrophobic_moments(sequences, scale, window_size, angle)
    
    if not means:
        print(f"  Warning: No hydrophobic moments calculated for {output_path}")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=PUBLICATION_DPI)
    
    positions = range(1, len(means) + 1)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    
    # --- Panel 1: Hydrophobic Moment Profile with STD ---
    ax1 = axes[0]
    
    # Plot STD shading (±1 SD)
    ax1.fill_between(positions, means_arr - stds_arr, means_arr + stds_arr, 
                     alpha=0.25, color='#a66ded', label='±1 SD')
    
    # Plot mean hydrophobic moment
    ax1.fill_between(positions, means_arr, alpha=0.4, color='#a66ded')
    ax1.plot(positions, means_arr, color='#6b21a8', linewidth=2, label='Mean Moment')
    
    # Add threshold line for significant amphipathicity
    threshold = np.mean(means_arr) + np.std(means_arr)
    ax1.axhline(y=threshold, color='#ff6b6b', linestyle='--', linewidth=1.5, 
                label=f'High Moment Threshold ({threshold:.2f})')
    ax1.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Highlight regions with high hydrophobic moment (amphipathic helices)
    high_moment = means_arr > threshold
    for i, (pos, is_high) in enumerate(zip(positions, high_moment)):
        if is_high:
            ax1.axvspan(pos - 0.5, pos + 0.5, alpha=0.15, color='#ff9f43', zorder=0)
    
    ax1.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hydrophobic Moment (μH)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Hydrophobic Moment Profile{title_suffix}\n{scale_name} Scale (Window: {window_size}, Angle: {angle}°)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='white', edgecolor='#333333')
    ax1.set_xlim(1, len(means))
    
    # --- Panel 2: Comparison - Mean Hydrophobicity vs Hydrophobic Moment ---
    ax2 = axes[1]
    
    # Calculate mean hydrophobicity for comparison
    hydro_means, hydro_stds = calculate_consensus_hydrophobicity(sequences, scale)
    smoothed_hydro = sliding_window_average(hydro_means, window_size)
    
    # Normalize both to [0, 1] for comparison
    hydro_norm = (np.array(smoothed_hydro[:len(means)]) - min(smoothed_hydro[:len(means)])) / \
                 (max(smoothed_hydro[:len(means)]) - min(smoothed_hydro[:len(means)]) + 1e-10)
    moment_norm = (means_arr - min(means_arr)) / (max(means_arr) - min(means_arr) + 1e-10)
    
    ax2.plot(positions, hydro_norm, color='#00d9ff', linewidth=2, label='Hydrophobicity (normalized)')
    ax2.plot(positions, moment_norm, color='#6b21a8', linewidth=2, label='Hydrophobic Moment (normalized)')
    ax2.fill_between(positions, hydro_norm, alpha=0.2, color='#00d9ff')
    ax2.fill_between(positions, moment_norm, alpha=0.2, color='#a66ded')
    
    ax2.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
    ax2.set_title('Hydrophobicity vs Amphipathicity Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='white', edgecolor='#333333')
    ax2.set_xlim(1, len(means))
    ax2.set_ylim(-0.05, 1.05)
    
    fig.suptitle(f'Hydrophobic Moment Analysis - Amphipathic Helix Detection{title_suffix}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_hydrophobicity_heatmap(sequences: Dict[str, str], output_path: str,
                                   max_sequences: int = 50, title_suffix: str = ""):
    """Create QR code-style hydrophobicity visualization using Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)."""
    setup_style()
    
    # Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)
    scale = COWAN_WHITTAKER
    scale_name = "Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)"
    
    # Calculate consensus hydrophobicity
    means, stds = calculate_consensus_hydrophobicity(sequences, scale)
    
    # Create custom colormap (blue-white-red for hydrophilicity-neutral-hydrophobicity)
    colors_cmap = ['#2980b9', '#ffffff', '#c0392b']
    cmap = LinearSegmentedColormap.from_list('hydro', colors_cmap, N=256)
    
    # QR code style: reshape data into a 2D grid
    # Calculate optimal grid dimensions
    n_positions = len(means)
    grid_width = int(np.ceil(np.sqrt(n_positions)))
    grid_height = int(np.ceil(n_positions / grid_width))
    
    # Pad means to fill the grid
    padded_means = means + [0.0] * (grid_width * grid_height - n_positions)
    qr_data = np.array(padded_means).reshape(grid_height, grid_width)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=PUBLICATION_DPI)
    
    # 1. QR code-style 2D grid visualization
    ax1 = axes[0]
    
    # Determine color scale range based on Cowan-Whittaker values
    vmax = max(abs(min(means)), abs(max(means)), 2.15)
    
    im = ax1.imshow(qr_data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    
    # Add grid lines for QR code effect
    ax1.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax1.grid(which='minor', color='#333333', linestyle='-', linewidth=0.3)
    ax1.tick_params(which='minor', size=0)
    
    # Remove major ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add position labels on edges
    ax1.set_xlabel(f'Position (1-{n_positions}, read left→right, top→bottom)', fontsize=10)
    ax1.set_title(f'QR Code-Style Hydrophobicity Map{title_suffix}\n{scale_name}', 
                 fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', shrink=0.8,
                        label='Hydrophobicity')
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.tick_params(colors='black')
    
    # 2. Traditional barcode for comparison
    ax2 = axes[1]
    
    # Barcode style
    barcode_data = np.array(means).reshape(1, -1)
    im2 = ax2.imshow(barcode_data, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                    extent=[0.5, len(means) + 0.5, 0, 1])
    
    ax2.set_yticks([])
    ax2.set_xlabel('Position in Alignment', fontsize=10)
    ax2.set_title(f'Linear Barcode View\n{scale_name}', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.15, 
                         label='Hydrophobicity', shrink=0.8)
    cbar2.ax.xaxis.label.set_color('black')
    cbar2.ax.tick_params(colors='black')
    
    # Add annotation explaining the reading order
    fig.text(0.25, 0.02, 
             f'Grid: {grid_height}×{grid_width} = {grid_height*grid_width} cells | Positions: {n_positions}',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_hydrophobicity_distribution(sequences: Dict[str, str], output_path: str,
                                        title_suffix: str = ""):
    """Create distribution plot of hydrophobicity values."""
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PUBLICATION_DPI)
    
    # 1. Overall amino acid hydrophobicity distribution
    ax1 = axes[0, 0]
    all_aa = []
    for seq in sequences.values():
        all_aa.extend([aa.upper() for aa in seq if aa != '-'])
    
    aa_hydro = {aa: KYTE_DOOLITTLE.get(aa, 0) for aa in set(all_aa)}
    aa_counts = defaultdict(int)
    for aa in all_aa:
        aa_counts[aa] += 1
    
    sorted_aa = sorted(aa_hydro.keys(), key=lambda x: aa_hydro[x])
    colors = ['#ff6b6b' if aa_hydro[aa] > 0 else '#4a69bd' for aa in sorted_aa]
    counts = [aa_counts[aa] for aa in sorted_aa]
    
    ax1.barh(sorted_aa, [aa_hydro[aa] for aa in sorted_aa], color=colors, alpha=0.8)
    ax1.axvline(x=0, color='#333333', linestyle='--', linewidth=1)
    ax1.set_xlabel('Hydrophobicity (Kyte-Doolittle)', fontsize=10)
    ax1.set_ylabel('Amino Acid', fontsize=10)
    ax1.set_title('Amino Acid Hydrophobicity', fontsize=11, fontweight='bold')
    
    # 2. Histogram of position-wise mean hydrophobicity
    ax2 = axes[0, 1]
    means, _ = calculate_consensus_hydrophobicity(sequences, KYTE_DOOLITTLE)
    non_zero_means = [m for m in means if m != 0]
    
    ax2.hist(non_zero_means, bins=50, color='#00d9ff', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(x=0, color='#ff6b6b', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(non_zero_means), color='#ffd93d', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(non_zero_means):.2f}')
    ax2.set_xlabel('Hydrophobicity', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Distribution of Position-wise Mean Hydrophobicity', fontsize=11, fontweight='bold')
    ax2.legend(facecolor='white', edgecolor='#333333')
    
    # 3. Amino acid frequency
    ax3 = axes[1, 0]
    sorted_by_count = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
    aa_names = [aa for aa, _ in sorted_by_count]
    aa_freq = [count for _, count in sorted_by_count]
    colors = ['#ff6b6b' if KYTE_DOOLITTLE.get(aa, 0) > 0 else '#4a69bd' for aa in aa_names]
    
    ax3.bar(aa_names, aa_freq, color=colors, alpha=0.8)
    ax3.set_xlabel('Amino Acid', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Amino Acid Frequency (Red=Hydrophobic, Blue=Hydrophilic)', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=0)
    
    # 4. Box plot comparing hydrophobicity by amino acid type
    ax4 = axes[1, 1]
    hydrophobic_vals = [KYTE_DOOLITTLE.get(aa, 0) for aa in all_aa if KYTE_DOOLITTLE.get(aa.upper(), 0) > 0]
    hydrophilic_vals = [KYTE_DOOLITTLE.get(aa, 0) for aa in all_aa if KYTE_DOOLITTLE.get(aa.upper(), 0) <= 0]
    
    bp = ax4.boxplot([hydrophilic_vals, hydrophobic_vals], 
                     labels=['Hydrophilic', 'Hydrophobic'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#4a69bd', '#ff6b6b']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    
    ax4.set_ylabel('Hydrophobicity', fontsize=10)
    ax4.set_title('Hydrophobicity Distribution by Type', fontsize=11, fontweight='bold')
    
    fig.suptitle(f'Hydrophobicity Distribution Analysis{title_suffix}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_comparison_plot(sequences1: Dict[str, str], sequences2: Dict[str, str],
                            name1: str, name2: str, output_path: str,
                            window_size: int = 9):
    """Compare hydrophobicity profiles between two sequence sets."""
    setup_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=PUBLICATION_DPI)
    
    # Calculate profiles
    means1, stds1 = calculate_consensus_hydrophobicity(sequences1, KYTE_DOOLITTLE)
    means2, stds2 = calculate_consensus_hydrophobicity(sequences2, KYTE_DOOLITTLE)
    
    smoothed1 = sliding_window_average(means1, window_size)
    smoothed2 = sliding_window_average(means2, window_size)
    
    # 1. Overlay plot
    ax1 = axes[0]
    pos1 = range(1, len(smoothed1) + 1)
    pos2 = range(1, len(smoothed2) + 1)
    
    ax1.fill_between(pos1, smoothed1, alpha=0.3, color=SYSTEM_COLORS['psy'], label=name1)
    ax1.plot(pos1, smoothed1, color=SYSTEM_COLORS['psy'], linewidth=2)
    ax1.fill_between(pos2, smoothed2, alpha=0.3, color=SYSTEM_COLORS['crtm'], label=name2)
    ax1.plot(pos2, smoothed2, color=SYSTEM_COLORS['crtm'], linewidth=2)
    ax1.axhline(y=0, color='#333333', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Hydrophobicity', fontsize=11)
    ax1.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='white', edgecolor='#333333')
    
    # 2. Side-by-side comparison
    ax2 = axes[1]
    
    # Normalize to same length for comparison
    max_len = max(len(smoothed1), len(smoothed2))
    s1_padded = smoothed1 + [0.0] * (max_len - len(smoothed1))
    s2_padded = smoothed2 + [0.0] * (max_len - len(smoothed2))
    
    width = 0.4
    positions = np.arange(0, max_len, 10)  # Show every 10th position
    
    vals1 = [s1_padded[i] if i < len(s1_padded) else 0 for i in positions]
    vals2 = [s2_padded[i] if i < len(s2_padded) else 0 for i in positions]
    
    ax2.bar(positions - width/2, vals1, width, label=name1, color=SYSTEM_COLORS['psy'], alpha=0.8)
    ax2.bar(positions + width/2, vals2, width, label=name2, color=SYSTEM_COLORS['crtm'], alpha=0.8)
    ax2.axhline(y=0, color='#333333', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Hydrophobicity', fontsize=11)
    ax2.set_title('Bar Comparison (Every 10th Position)', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='white', edgecolor='#333333')
    
    # 3. Difference plot
    ax3 = axes[2]
    min_len = min(len(smoothed1), len(smoothed2))
    diff = [smoothed1[i] - smoothed2[i] for i in range(min_len)]
    positions = range(1, min_len + 1)
    
    colors = [SYSTEM_COLORS['psy'] if d > 0 else SYSTEM_COLORS['crtm'] for d in diff]
    ax3.bar(positions, diff, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    
    ax3.set_xlabel('Position', fontsize=11)
    ax3.set_ylabel(f'Difference ({name1} - {name2})', fontsize=11)
    ax3.set_title('Hydrophobicity Difference', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Hydrophobicity Comparison: {name1} vs {name2}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_hydrophobic_region_map(sequences: Dict[str, str], output_path: str,
                                   threshold: float = 1.0, min_length: int = 5,
                                   title_suffix: str = ""):
    """Identify and visualize hydrophobic regions (potential transmembrane segments)."""
    setup_style()
    
    means, _ = calculate_consensus_hydrophobicity(sequences, KYTE_DOOLITTLE)
    smoothed = sliding_window_average(means, 19)  # Use larger window for TM detection
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=PUBLICATION_DPI)
    
    # 1. Profile with highlighted regions
    ax1 = axes[0]
    positions = range(1, len(smoothed) + 1)
    
    ax1.fill_between(positions, smoothed, alpha=0.3, color='#00d9ff')
    ax1.plot(positions, smoothed, color='#00d9ff', linewidth=2)
    ax1.axhline(y=threshold, color='#ff6b6b', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({threshold})')
    ax1.axhline(y=0, color='#333333', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Find and highlight hydrophobic regions
    regions = []
    in_region = False
    start = 0
    
    for i, val in enumerate(smoothed):
        if val >= threshold and not in_region:
            in_region = True
            start = i
        elif val < threshold and in_region:
            if i - start >= min_length:
                regions.append((start, i))
            in_region = False
    
    if in_region and len(smoothed) - start >= min_length:
        regions.append((start, len(smoothed)))
    
    for start, end in regions:
        ax1.axvspan(start + 1, end + 1, alpha=0.3, color='#ff9f43', 
                   label='Hydrophobic Region' if start == regions[0][0] else '')
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Hydrophobicity', fontsize=11)
    ax1.set_title(f'Hydrophobic Region Detection{title_suffix}', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='white', edgecolor='#333333')
    
    # 2. Region map (simplified view)
    ax2 = axes[1]
    
    # Create a bar showing regions
    region_map = np.zeros(len(smoothed))
    for start, end in regions:
        region_map[start:end] = 1
    
    ax2.imshow([region_map], aspect='auto', cmap='RdYlBu_r', extent=[0, len(smoothed), 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_title(f'Hydrophobic Region Map ({len(regions)} regions detected)', 
                 fontsize=12, fontweight='bold')
    
    # Add region annotations
    for i, (start, end) in enumerate(regions):
        mid = (start + end) / 2
        ax2.text(mid, 0.5, f'{i+1}\n({end-start})', ha='center', va='center',
                fontsize=9, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return regions

def create_summary_statistics(sequences: Dict[str, str], output_path: str,
                               title_suffix: str = ""):
    """Create summary statistics visualization."""
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PUBLICATION_DPI)
    
    # Calculate various statistics
    means, stds = calculate_consensus_hydrophobicity(sequences, KYTE_DOOLITTLE)
    smoothed = sliding_window_average(means, 9)
    
    # 1. Statistics summary
    ax1 = axes[0, 0]
    stats = {
        'Mean Hydrophobicity': np.mean([m for m in means if m != 0]),
        'Std Hydrophobicity': np.std([m for m in means if m != 0]),
        'Max Hydrophobicity': max(means),
        'Min Hydrophobicity': min([m for m in means if m != 0]),
        'Hydrophobic %': sum(1 for m in means if m > 0) / len([m for m in means if m != 0]) * 100,
        'Hydrophilic %': sum(1 for m in means if m < 0) / len([m for m in means if m != 0]) * 100,
    }
    
    stat_names = list(stats.keys())
    stat_values = list(stats.values())
    colors = ['#00d9ff'] * len(stats)
    
    bars = ax1.barh(stat_names, stat_values, color=colors, alpha=0.8)
    ax1.set_xlabel('Value', fontsize=10)
    ax1.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, stat_values):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontsize=9, color='black')
    
    # 2. Cumulative hydrophobicity
    ax2 = axes[0, 1]
    cumsum = np.cumsum([m for m in means if m != 0])
    ax2.plot(range(len(cumsum)), cumsum, color='#ffd93d', linewidth=2)
    ax2.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='#ffd93d')
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_ylabel('Cumulative Hydrophobicity', fontsize=10)
    ax2.set_title('Cumulative Hydrophobicity Profile', fontsize=11, fontweight='bold')
    
    # 3. Scale comparison for overall mean
    ax3 = axes[1, 0]
    scale_means = {}
    for name, scale in SCALES.items():
        m, _ = calculate_consensus_hydrophobicity(sequences, scale)
        scale_means[name] = np.mean([v for v in m if v != 0])
    
    ax3.bar(scale_means.keys(), scale_means.values(), 
            color=['#ff6b6b', '#00d9ff', '#ffd93d', '#6bcb77', '#a66ded', '#ff9f43'],
            alpha=0.8)
    ax3.axhline(y=0, color='#333333', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Mean Hydrophobicity', fontsize=10)
    ax3.set_title('Mean Hydrophobicity by Scale', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Sequence length distribution
    ax4 = axes[1, 1]
    seq_lengths = [len(seq.replace('-', '')) for seq in sequences.values()]
    ax4.hist(seq_lengths, bins=30, color='#6bcb77', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax4.axvline(x=np.mean(seq_lengths), color='#ff6b6b', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(seq_lengths):.0f}')
    ax4.set_xlabel('Sequence Length (without gaps)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Sequence Length Distribution', fontsize=11, fontweight='bold')
    ax4.legend(facecolor='white', edgecolor='#333333')
    
    fig.suptitle(f'Hydrophobicity Analysis Summary{title_suffix}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_hydrophobicity(fasta_file: str, output_dir: str, name: str = None):
    """Run complete hydrophobicity analysis on a FASTA file."""
    
    if name is None:
        name = os.path.splitext(os.path.basename(fasta_file))[0]
    
    # Standardize names for plotting
    name_map = {'finalpsy': 'PSY', 'finalcrtm': 'CrtM'}
    name = name_map.get(name.lower(), name)
    
    print(f"\n{'='*60}")
    print(f"Hydrophobicity Analysis: {name}")
    print(f"{'='*60}")
    
    # Parse sequences
    print(f"\nParsing {fasta_file}...")
    sequences = parse_fasta(fasta_file)
    print(f"  Found {len(sequences)} sequences")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    title_suffix = f" - {name}"
    
    print(f"\nGenerating visualizations...")
    
    # 1. Kyte-Doolittle profile
    create_hydropathy_profile(
        sequences, 'Kyte-Doolittle',
        os.path.join(output_dir, f'{name}_kyte_doolittle_profile.png'),
        title_suffix=title_suffix
    )
    
    # 2. Multi-scale comparison
    create_multi_scale_comparison(
        sequences,
        os.path.join(output_dir, f'{name}_multi_scale_comparison.png'),
        title_suffix=title_suffix
    )
    
    # 3. Hydrophobicity heatmap
    create_hydrophobicity_heatmap(
        sequences,
        os.path.join(output_dir, f'{name}_hydrophobicity_heatmap.png'),
        title_suffix=title_suffix
    )
    
    # 4. Distribution analysis
    create_hydrophobicity_distribution(
        sequences,
        os.path.join(output_dir, f'{name}_distribution_analysis.png'),
        title_suffix=title_suffix
    )
    
    # 5. Hydrophobic region map
    regions = create_hydrophobic_region_map(
        sequences,
        os.path.join(output_dir, f'{name}_hydrophobic_regions.png'),
        title_suffix=title_suffix
    )
    
    # 6. Summary statistics
    create_summary_statistics(
        sequences,
        os.path.join(output_dir, f'{name}_summary_statistics.png'),
        title_suffix=title_suffix
    )
    
    # 7. Hydrophobic moments plot (amphipathic helix detection)
    create_hydrophobic_moments_plot(
        sequences,
        os.path.join(output_dir, f'{name}_hydrophobic_moments.png'),
        title_suffix=title_suffix
    )
    
    print(f"\n  Detected {len(regions)} hydrophobic regions")
    for i, (start, end) in enumerate(regions):
        print(f"    Region {i+1}: positions {start+1}-{end+1} (length: {end-start})")
    
    return sequences, regions

def random_sample_sequences(sequences: Dict[str, str], n: int, seed: int = None) -> Dict[str, str]:
    """Randomly sample n sequences from the input dictionary."""
    import random
    if seed is not None:
        random.seed(seed)
    
    if len(sequences) <= n:
        return sequences
    
    keys = list(sequences.keys())
    sampled_keys = random.sample(keys, n)
    return {k: sequences[k] for k in sampled_keys}

def main():
    """Main function to run hydrophobicity analysis on PSY and CrtM sequences."""
    import random
    
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    psy_file = os.path.join(base_dir, 'finalpsy.fasta')
    crtm_file = os.path.join(base_dir, 'finalcrtm.fasta')
    output_dir = os.path.join(base_dir, 'hydrophobicity_analysis_output')
    
    print("\n" + "="*70)
    print("HYDROPHOBICITY ANALYSIS OF PSY AND CrtM SEQUENCES")
    print("="*70)
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse all sequences first
    print(f"\nParsing sequences...")
    all_psy_sequences = parse_fasta(psy_file)
    crtm_sequences_full = parse_fasta(crtm_file)
    
    n_crtm = len(crtm_sequences_full)
    n_psy = len(all_psy_sequences)
    print(f"  PSY: {n_psy} sequences")
    print(f"  CrtM: {n_crtm} sequences")
    
    # Number of random sampling iterations
    n_samples = 5
    print(f"\nPerforming {n_samples} random samplings of PSY (n={n_crtm} each)...")
    
    # Analyze CrtM sequences (full set)
    crtm_output = os.path.join(output_dir, 'CrtM')
    crtm_sequences, crtm_regions = analyze_hydrophobicity(crtm_file, crtm_output, 'CrtM')
    
    # Create directory for PSY samples
    psy_samples_dir = os.path.join(output_dir, 'PSY_Sampled')
    os.makedirs(psy_samples_dir, exist_ok=True)
    
    # Store all sampled PSY data for aggregate analysis
    all_psy_sampled = []
    
    for i in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Random Sampling {i+1}/{n_samples}")
        print(f"{'='*60}")
        
        # Random sample PSY sequences
        seed = 42 + i  # Reproducible but different seeds
        psy_sampled = random_sample_sequences(all_psy_sequences, n_crtm, seed=seed)
        all_psy_sampled.append(psy_sampled)
        
        print(f"  Sampled {len(psy_sampled)} PSY sequences (seed={seed})")
        
        # Create output directory for this sample
        sample_output = os.path.join(psy_samples_dir, f'Sample_{i+1}')
        os.makedirs(sample_output, exist_ok=True)
        
        # Generate visualizations for sampled PSY
        title_suffix = f" - PSY Sample {i+1} (n={n_crtm})"
        print(f"\n  Generating visualizations...")
        
        create_hydropathy_profile(
            psy_sampled, 'Kyte-Doolittle',
            os.path.join(sample_output, f'PSY_sample{i+1}_kyte_doolittle_profile.png'),
            title_suffix=title_suffix
        )
        
        create_multi_scale_comparison(
            psy_sampled,
            os.path.join(sample_output, f'PSY_sample{i+1}_multi_scale_comparison.png'),
            title_suffix=title_suffix
        )
        
        create_hydrophobicity_distribution(
            psy_sampled,
            os.path.join(sample_output, f'PSY_sample{i+1}_distribution_analysis.png'),
            title_suffix=title_suffix
        )
        
        create_hydrophobic_region_map(
            psy_sampled,
            os.path.join(sample_output, f'PSY_sample{i+1}_hydrophobic_regions.png'),
            title_suffix=title_suffix
        )
        
        create_summary_statistics(
            psy_sampled,
            os.path.join(sample_output, f'PSY_sample{i+1}_summary_statistics.png'),
            title_suffix=title_suffix
        )
        
        # Create comparison plot for this sample vs CrtM
        create_comparison_plot(
            psy_sampled, crtm_sequences,
            f'PSY (Sample {i+1})', 'CrtM',
            os.path.join(sample_output, f'PSY_sample{i+1}_vs_CrtM_comparison.png')
        )
    
    # Also analyze full PSY for reference
    print(f"\n{'='*60}")
    print("Analyzing Full PSY Dataset (for reference)")
    print(f"{'='*60}")
    psy_output = os.path.join(output_dir, 'PSY_Full')
    psy_sequences, psy_regions = analyze_hydrophobicity(psy_file, psy_output, 'PSY_Full')
    
    # Create aggregate comparison using all samples
    print(f"\n{'='*60}")
    print("Generating Aggregate Comparison Plots...")
    print(f"{'='*60}")
    
    # Create aggregate visualization showing all samples
    create_aggregate_comparison(all_psy_sampled, crtm_sequences, n_samples,
                                os.path.join(output_dir, 'PSY_samples_vs_CrtM_aggregate.png'))
    
    # Map hydrophobicity to PDB structure
    pdb_file = os.path.join(base_dir, '3W7F_A.pdb')
    
    if os.path.exists(pdb_file):
        print(f"\n{'='*70}")
        print("MAPPING HYDROPHOBICITY TO PDB STRUCTURE")
        print(f"{'='*70}")
        
        # Map CrtM hydrophobicity to structure (3W7F is a CrtM structure)
        hydro_mapping = map_hydrophobicity_to_structure(
            sequences=crtm_sequences_full,
            msa_file=crtm_file,
            pdb_file=pdb_file,
            output_dir=os.path.join(output_dir, 'CrtM'),
            scale=KYTE_DOOLITTLE,
            scale_name="Kyte-Doolittle",
            chain='A',
            hydrophobic_threshold=0.5,
            window_size=9
        )
    else:
        print(f"\nWarning: PDB file not found: {pdb_file}")
        print("Skipping structure mapping.")
        hydro_mapping = None
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print("  CrtM/                     - Full CrtM analysis")
    print("  PSY_Full/                 - Full PSY analysis (all sequences)")
    print("  PSY_Sampled/              - Random sampling analyses")
    for i in range(n_samples):
        print(f"    Sample_{i+1}/            - Sample {i+1} visualizations")
    print("  PSY_samples_vs_CrtM_aggregate.png")
    
    if hydro_mapping:
        print("\nStructure mapping files (in CrtM/):")
        print("  - hydrophobicity_pdb_mapping.csv")
        print("  - hydrophobic_regions_pdb.csv")
        print("  - visualize_hydrophobicity.pml (PyMOL script)")
        print("  - visualize_hydrophobic_regions.pml (PyMOL script)")
        print("  - visualize_hydrophobicity.cxc (ChimeraX script)")
        print("  - visualize_hydrophobic_regions.cxc (ChimeraX script)")


def create_aggregate_comparison(psy_samples: list, crtm_sequences: Dict[str, str], 
                                 n_samples: int, output_path: str):
    """Create aggregate comparison plot showing all PSY samples vs CrtM using Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)."""
    setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=PUBLICATION_DPI)
    
    # Use COWAN_WHITTAKER
    scale = COWAN_WHITTAKER
    scale_name = " Cowan-Whittaker scale (HPLC pH 7.5) - Cowan R., Whittaker R.G. (1990)"
    
    # Calculate CrtM profile
    crtm_means, _ = calculate_consensus_hydrophobicity(crtm_sequences, scale)
    crtm_smoothed = sliding_window_average(crtm_means, 9)
    
    # Calculate all PSY sample profiles
    psy_profiles = []
    for psy_sample in psy_samples:
        means, _ = calculate_consensus_hydrophobicity(psy_sample, scale)
        smoothed = sliding_window_average(means, 9)
        psy_profiles.append(smoothed)
    
    # 1. Overlay all profiles
    ax1 = axes[0]
    
    # Plot CrtM
    crtm_pos = range(1, len(crtm_smoothed) + 1)
    ax1.plot(crtm_pos, crtm_smoothed, color=SYSTEM_COLORS['crtm'], linewidth=2.5, label='CrtM', zorder=10)
    
    # Plot all PSY samples with transparency
    # Using Blues colormap for PSY samples to match PSY standard
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, n_samples))
    for i, profile in enumerate(psy_profiles):
        pos = range(1, len(profile) + 1)
        ax1.plot(pos, profile, color=colors[i], linewidth=1, alpha=0.5, 
                label=f'PSY Sample {i+1}' if i == 0 else None)
    
    # Calculate and plot PSY mean ± std
    min_len = min(len(p) for p in psy_profiles)
    psy_matrix = np.array([p[:min_len] for p in psy_profiles])
    psy_mean = np.mean(psy_matrix, axis=0)
    psy_std = np.std(psy_matrix, axis=0)
    
    psy_pos = range(1, min_len + 1)
    ax1.plot(psy_pos, psy_mean, color=SYSTEM_COLORS['psy'], linewidth=2.5, label='PSY Mean', zorder=9)
    ax1.fill_between(psy_pos, psy_mean - psy_std, psy_mean + psy_std, 
                    alpha=0.3, color=SYSTEM_COLORS['psy'], label='PSY ±1 SD')
    
    ax1.axhline(y=0, color='#333333', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Residue Alignment Position', fontsize=16)
    ax1.set_ylabel('Hydrophobicity', fontsize=16)
    ax1.set_title(f'Aggregate Comparison:  PSY Samples vs CrtM', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='white', edgecolor='#333333', fontsize=12)
    
    # 2. Difference plot (mean PSY - CrtM)
    ax2 = axes[1]
    
    min_len_all = min(min_len, len(crtm_smoothed))
    diff = psy_mean[:min_len_all] - np.array(crtm_smoothed[:min_len_all])
    diff_std = psy_std[:min_len_all]
    positions = range(1, min_len_all + 1)
    
    colors_diff = [SYSTEM_COLORS['psy'] if d > 0 else SYSTEM_COLORS['crtm'] for d in diff]
    ax2.bar(positions, diff, color=colors_diff, alpha=0.7)
    ax2.fill_between(positions, diff - diff_std, diff + diff_std, alpha=0.2, color='gray')
    ax2.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Residue Alignment Position', fontsize=16)
    ax2.set_ylabel('Difference (PSY Mean - CrtM)', fontsize=16)
    ax2.set_title('Hydrophobicity Difference with Sampling Uncertainty', fontsize=16, fontweight='bold')
    
    # Add descriptive legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SYSTEM_COLORS['psy'], alpha=0.7, label='PSY-enriched'),
        Patch(facecolor=SYSTEM_COLORS['crtm'], alpha=0.7, label='CrtM-enriched'),
        Patch(facecolor='gray', alpha=0.2, label='Sampling Uncertainty (±1 SD)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', facecolor='white', edgecolor='#333333', fontsize=12)
    
    fig.suptitle(f'Hydrophobicity Comparison - {scale_name}\n(Random Sampling n={len(crtm_sequences)} per sample)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PUBLICATION_DPI, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# STRUCTURE MAPPING FUNCTIONS
# =============================================================================

def map_hydrophobicity_to_structure(
    sequences: Dict[str, str],
    msa_file: str,
    pdb_file: str,
    output_dir: str,
    scale: Dict[str, float] = None,
    scale_name: str = "Kyte-Doolittle",
    chain: str = 'A',
    hydrophobic_threshold: float = 0.5,
    window_size: int = 9
) -> Dict:
    """
    Map hydrophobicity analysis to PDB structure residue numbers.
    
    Args:
        sequences: Dictionary of MSA sequences
        msa_file: Path to the MSA FASTA file
        pdb_file: Path to PDB structure file
        output_dir: Directory for output files
        scale: Hydrophobicity scale dictionary (default: Kyte-Doolittle)
        scale_name: Name of the scale for labeling
        chain: PDB chain identifier
        hydrophobic_threshold: Threshold for hydrophobic classification
        window_size: Window size for smoothing
    
    Returns:
        Dictionary with mapping results and output file paths
    """
    from structure_mapper import (
        parse_fasta as sm_parse_fasta,
        get_pdb_info,
        get_representative_sequence,
        simple_align,
        get_alignment_stats,
        create_msa_position_map,
        generate_hydrophobicity_pymol_script,
        generate_hydrophobicity_chimerax_script,
        generate_chimerax_regions_script
    )
    
    if scale is None:
        scale = KYTE_DOOLITTLE
    
    print(f"\n{'='*60}")
    print("MAPPING HYDROPHOBICITY TO STRUCTURE")
    print(f"{'='*60}")
    
    # Parse MSA and PDB
    print(f"\nParsing MSA: {msa_file}")
    msa_sequences = sm_parse_fasta(msa_file)
    print(f"  Loaded {len(msa_sequences)} sequences")
    
    print(f"\nParsing PDB: {pdb_file}")
    pdb_info = get_pdb_info(pdb_file, chain)
    print(f"  Chain {chain}: {pdb_info['length']} residues")
    print(f"  Residue range: {pdb_info['first_residue']}-{pdb_info['last_residue']}")
    
    # Get representative sequence and align
    header, rep_seq = get_representative_sequence(msa_sequences)
    first_aligned_seq = next(iter(msa_sequences.values()))
    msa_position_map = create_msa_position_map(first_aligned_seq)
    
    print(f"\nAligning MSA representative to PDB sequence...")
    aligned_msa, aligned_pdb, position_pairs = simple_align(rep_seq, pdb_info['sequence'])
    stats = get_alignment_stats(aligned_msa, aligned_pdb)
    print(f"  Identity: {stats['identity']:.1f}%")
    print(f"  Coverage: {stats['coverage']:.1f}%")
    
    # Create ungapped to PDB residue number mapping
    ungapped_to_pdb = {}
    for msa_pos, pdb_pos in position_pairs:
        if pdb_pos <= len(pdb_info['residue_numbers']):
            ungapped_to_pdb[msa_pos] = pdb_info['residue_numbers'][pdb_pos - 1]
    
    # Calculate consensus hydrophobicity at each MSA position
    print(f"\nCalculating position-wise hydrophobicity...")
    means, stds = calculate_consensus_hydrophobicity(sequences, scale)
    smoothed = sliding_window_average(means, window_size)
    
    # Map MSA positions to PDB residue numbers with hydrophobicity values
    residue_hydrophobicity = {}
    all_mappings = []
    
    for msa_pos in range(1, len(smoothed) + 1):
        hydro_val = smoothed[msa_pos - 1] if msa_pos <= len(smoothed) else 0
        
        mapping = {
            'msa_position': msa_pos,
            'ungapped_position': None,
            'pdb_residue_number': None,
            'mapped': False,
            'hydrophobicity': hydro_val,
            'classification': 'hydrophobic' if hydro_val > hydrophobic_threshold else (
                'hydrophilic' if hydro_val < -hydrophobic_threshold else 'neutral'
            )
        }
        
        if msa_pos in msa_position_map:
            ungapped_pos = msa_position_map[msa_pos]
            mapping['ungapped_position'] = ungapped_pos
            
            if ungapped_pos in ungapped_to_pdb:
                pdb_res = ungapped_to_pdb[ungapped_pos]
                mapping['pdb_residue_number'] = pdb_res
                mapping['mapped'] = True
                residue_hydrophobicity[pdb_res] = hydro_val
        
        all_mappings.append(mapping)
    
    # Identify hydrophobic regions on structure
    hydrophobic_regions = []
    in_region = False
    region_start = None
    region_residues = []
    
    # Sort mapped residues by PDB number
    sorted_residues = sorted(residue_hydrophobicity.items())
    
    for pdb_res, hydro_val in sorted_residues:
        if hydro_val > hydrophobic_threshold:
            if not in_region:
                in_region = True
                region_start = pdb_res
                region_residues = [pdb_res]
            else:
                region_residues.append(pdb_res)
        else:
            if in_region and len(region_residues) >= 5:  # Min region length
                hydrophobic_regions.append({
                    'start': region_start,
                    'end': region_residues[-1],
                    'residues': region_residues.copy(),
                    'length': len(region_residues),
                    'mean_hydrophobicity': np.mean([residue_hydrophobicity[r] for r in region_residues])
                })
            in_region = False
            region_residues = []
    
    # Handle last region
    if in_region and len(region_residues) >= 5:
        hydrophobic_regions.append({
            'start': region_start,
            'end': region_residues[-1],
            'residues': region_residues.copy(),
            'length': len(region_residues),
            'mean_hydrophobicity': np.mean([residue_hydrophobicity[r] for r in region_residues])
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("MAPPING SUMMARY")
    print(f"{'='*60}")
    mapped_count = sum(1 for m in all_mappings if m['mapped'])
    print(f"  Total MSA positions: {len(all_mappings)}")
    print(f"  Mapped to PDB: {mapped_count}")
    print(f"  Hydrophobic residues (>{hydrophobic_threshold}): {sum(1 for v in residue_hydrophobicity.values() if v > hydrophobic_threshold)}")
    print(f"  Hydrophobic regions detected: {len(hydrophobic_regions)}")
    
    for i, region in enumerate(hydrophobic_regions):
        print(f"    Region {i+1}: residues {region['start']}-{region['end']} ({region['length']} aa, mean={region['mean_hydrophobicity']:.2f})")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'hydrophobicity_pdb_mapping.csv')
    _save_hydrophobicity_mapping_csv(all_mappings, csv_path)
    
    # Save regions CSV
    regions_csv_path = os.path.join(output_dir, 'hydrophobic_regions_pdb.csv')
    _save_hydrophobic_regions_csv(hydrophobic_regions, regions_csv_path)
    
    # Generate PyMOL script
    pml_path = os.path.join(output_dir, 'visualize_hydrophobicity.pml')
    generate_hydrophobicity_pymol_script(pdb_file, residue_hydrophobicity, pml_path, scale_name)
    
    # Generate region highlighting script
    region_pml_path = os.path.join(output_dir, 'visualize_hydrophobic_regions.pml')
    _generate_region_pymol_script(pdb_file, hydrophobic_regions, region_pml_path)
    
    # Generate ChimeraX scripts
    cxc_path = os.path.join(output_dir, 'visualize_hydrophobicity.cxc')
    generate_hydrophobicity_chimerax_script(pdb_file, residue_hydrophobicity, cxc_path, scale_name)
    
    region_cxc_path = os.path.join(output_dir, 'visualize_hydrophobic_regions.cxc')
    generate_chimerax_regions_script(pdb_file, hydrophobic_regions, region_cxc_path)
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  Hydrophobicity mapping CSV: {csv_path}")
    print(f"  Hydrophobic regions CSV: {regions_csv_path}")
    print(f"  PyMOL hydrophobicity script: {pml_path}")
    print(f"  PyMOL regions script: {region_pml_path}")
    print(f"  ChimeraX hydrophobicity script: {cxc_path}")
    print(f"  ChimeraX regions script: {region_cxc_path}")
    print(f"\nTo visualize, run:")
    print(f"  PyMOL: pymol {pml_path}")
    print(f"  ChimeraX: chimerax {cxc_path}")
    
    return {
        'mappings': all_mappings,
        'residue_hydrophobicity': residue_hydrophobicity,
        'hydrophobic_regions': hydrophobic_regions,
        'pdb_info': pdb_info,
        'alignment_stats': stats,
        'csv_path': csv_path,
        'pml_path': pml_path,
        'region_pml_path': region_pml_path,
        'cxc_path': cxc_path,
        'region_cxc_path': region_cxc_path
    }


def _save_hydrophobicity_mapping_csv(mappings: List, output_path: str) -> str:
    """Save hydrophobicity mappings to CSV."""
    import csv
    
    columns = ['msa_position', 'ungapped_position', 'pdb_residue_number', 
               'mapped', 'hydrophobicity', 'classification']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mappings)
    
    print(f"  Hydrophobicity mapping CSV saved: {output_path}")
    return output_path


def _save_hydrophobic_regions_csv(regions: List[Dict], output_path: str) -> str:
    """Save hydrophobic regions to CSV."""
    import csv
    
    if not regions:
        print(f"  No hydrophobic regions to save")
        return output_path
    
    columns = ['region_id', 'start_residue', 'end_residue', 'length', 
               'mean_hydrophobicity', 'residues']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        for i, region in enumerate(regions):
            writer.writerow([
                i + 1,
                region['start'],
                region['end'],
                region['length'],
                f"{region['mean_hydrophobicity']:.3f}",
                '+'.join(str(r) for r in region['residues'])
            ])
    
    print(f"  Hydrophobic regions CSV saved: {output_path}")
    return output_path


def _generate_region_pymol_script(
    pdb_file: str,
    regions: List[Dict],
    output_path: str
) -> str:
    """Generate PyMOL script to highlight hydrophobic regions."""
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    script_lines = [
        "# PyMOL Hydrophobic Regions Visualization Script",
        f"# Generated for: {pdb_basename}",
        "",
        "reinitialize",
        "",
        f"load {rel_pdb_path}, {pdb_name}",
        "",
        "# Base visualization",
        "hide all",
        f"show cartoon, {pdb_name}",
        f"color gray80, {pdb_name}",
        "",
        f"# Highlight {len(regions)} hydrophobic regions",
    ]
    
    # Color palette for regions
    colors = ['firebrick', 'orange', 'yellow', 'forest', 'marine', 'violet', 'hotpink']
    
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        res_str = '+'.join(str(r) for r in region['residues'])
        selection_name = f"hydro_region_{i+1}"
        
        script_lines.extend([
            f"",
            f"# Region {i+1}: residues {region['start']}-{region['end']} (mean hydro: {region['mean_hydrophobicity']:.2f})",
            f"select {selection_name}, {pdb_name} and resi {res_str}",
            f"color {color}, {selection_name}",
            f"show surface, {selection_name}",
            f"set transparency, 0.5, {selection_name}",
        ])
    
    script_lines.extend([
        "",
        "# Final settings",
        "deselect",
        "orient",
        "zoom vis",
        "bg_color white",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"  PyMOL regions script saved: {output_path}")
    return output_path


if __name__ == '__main__':
    main()

