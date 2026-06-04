"""
Conserved Amino Acid Analysis Tool
Extracts conserved amino acids from MSA FASTA files and performs comparative
overlay analysis between two protein families (PSY and CrtM).
"""
import os
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from sklearn.manifold import TSNE
# =============================================================================
# PUBLICATION-QUALITY PLOT SETTINGS
# =============================================================================
# Publication DPI (300 for print, 150 for screen)
PUBLICATION_DPI = 300
# Figure sizes optimized for journal columns (in inches)
# Single column: ~3.5", Double column: ~7", Full page width: ~7.5"
FIGURE_SIZES = {
    'single_column': (3.5, 3.0),
    'double_column': (7.0, 5.0),
    'full_page': (7.5, 10.0),
    'wide': (10.0, 6.0),
    'square': (6.0, 6.0),
    'landscape': (12.0, 7.0),
    'portrait': (8.0, 10.0),
}

def setup_publication_style():
    """Configure matplotlib for Nature publication-quality figures."""
    plt.rcParams.update({
        'font.family'       : 'sans-serif',
        'font.sans-serif'   : ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size'         : 8,
        'axes.labelsize'    : 8,
        'axes.titlesize'    : 8,
        'axes.titleweight'  : 'bold',
        'xtick.labelsize'   : 7,
        'ytick.labelsize'   : 7,
        'legend.fontsize'   : 7,
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
        'savefig.pad_inches': 0.02
    })

# Color palette for publication (colorblind-friendly)
PUBLICATION_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'success': '#3A7D44',      # Green
    'neutral': '#6C757D',      # Gray
    'psy': '#3498db',          # PSY color (blue)
    'crtm': '#F18F01',         # CrtM color (orange)
    'shared': '#27ae60',       # Shared (green)
    'highlight': '#C73E1D',    # Binding site highlight (Red)
}

# Mapped from 3W7F_A.pdb S. aureus CrtM
BINDING_SITE_ALIGNMENT_POSITIONS = [
    33, 38, 39, 40, 41, 42, 46, 58, 62, 72, 73, 76, 77, 212, 261, 262, 265, 
    266, 269, 270, 274, 303, 306, 307, 310, 311, 314, 315, 317, 318, 321, 
    322, 328, 398, 415, 425, 428, 445
]

def parse_fasta(filepath: str) -> Dict[str, str]:
    """Parse a FASTA file and return a dictionary of sequences."""
    sequences = {}
    current_header = None
    current_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_header:
        sequences[current_header] = ''.join(current_seq)
    return sequences

def get_alignment_length(sequences: Dict[str, str]) -> int:
    """Get the alignment length from sequences."""
    if not sequences:
        return 0
    return len(next(iter(sequences.values())))

def calculate_conservation(
    sequences: Dict[str, str],
    threshold: float = 0.8,
    min_coverage: float = 0.5
) -> List[Dict]:
    """
    Calculate conserved positions in a multiple sequence alignment.
    Args:
        sequences: Dictionary of sequence name -> aligned sequence
        threshold: Minimum fraction of sequences that must have the same amino acid
                   for a position to be considered conserved (default: 80%)
        min_coverage: Minimum fraction of total sequences that must have data
                      (non-gap) at a position to be considered significant (default: 50%)
    Returns:
        List of dictionaries containing conserved position info
    """
    if not sequences:
        return []
    alignment_length = get_alignment_length(sequences)
    num_sequences = len(sequences)
    min_count = int(num_sequences * min_coverage)  # Convert percentage to absolute count
    conserved_positions = []

    for pos in range(alignment_length):
        column = []
        for seq in sequences.values():
            if pos < len(seq):
                aa = seq[pos]
                if aa != '-':  # Exclude gaps
                    column.append(aa)

        if not column:
            continue

        # Skip positions with insufficient coverage (too many gaps)
        total_non_gap = len(column)
        if total_non_gap < min_count:
            continue

        # Count amino acids at this position
        aa_counts = Counter(column)
        most_common_aa, count = aa_counts.most_common(1)[0]

        # Calculate conservation (excluding gaps)
        conservation_ratio = count / total_non_gap if total_non_gap > 0 else 0

        if conservation_ratio >= threshold:
            conserved_positions.append({
                'position': pos + 1,  # 1-indexed
                'amino_acid': most_common_aa,
                'conservation': conservation_ratio,
                'count': count,
                'total': total_non_gap,
                'gap_count': num_sequences - total_non_gap,
                'coverage': total_non_gap / num_sequences
            })

    return conserved_positions

def analyze_fasta_file(
    filepath: str,
    threshold: float = 0.8,
    min_coverage: float = 0.5
) -> Tuple[Dict[str, str], List[Dict]]:
    """Analyze a FASTA file and return sequences and conserved positions."""
    sequences = parse_fasta(filepath)
    conserved = calculate_conservation(sequences, threshold, min_coverage)
    return sequences, conserved

def compare_conserved_positions(
    conserved1: List[Dict],
    conserved2: List[Dict],
    name1: str = "PSY",
    name2: str = "CrtM"
) -> Dict:
    """
    Compare conserved positions between two alignments.
    Returns analysis of:
    - Positions conserved in both
    - Positions conserved only in first
    - Positions conserved only in second
    - Same amino acid at same positions
    - Different amino acids at same positions
    """
    # Create dictionaries keyed by position
    pos1 = {c['position']: c for c in conserved1}
    pos2 = {c['position']: c for c in conserved2}
    positions1 = set(pos1.keys())
    positions2 = set(pos2.keys())

    # Find overlapping and unique positions
    shared_positions = positions1 & positions2
    only_in_1 = positions1 - positions2
    only_in_2 = positions2 - positions1

    # For shared positions, check if same amino acid
    same_aa_positions = []
    different_aa_positions = []
    for pos in shared_positions:
        aa1 = pos1[pos]['amino_acid']
        aa2 = pos2[pos]['amino_acid']
        if aa1 == aa2:
            same_aa_positions.append({
                'position': pos,
                'amino_acid': aa1,
                f'{name1}_conservation': pos1[pos]['conservation'],
                f'{name2}_conservation': pos2[pos]['conservation'],
                f'{name1}_count': pos1[pos]['count'],
                f'{name2}_count': pos1[pos]['count'],
                f'{name1}_total': pos1[pos]['total'],
                f'{name2}_total': pos2[pos]['total']
            })
        else:
            different_aa_positions.append({
                'position': pos,
                f'{name1}_aa': aa1,
                f'{name2}_aa': aa2,
                f'{name1}_conservation': pos1[pos]['conservation'],
                f'{name2}_conservation': pos2[pos]['conservation'],
                f'{name1}_count': pos1[pos]['count'],
                f'{name2}_count': pos1[pos]['count'],
                f'{name1}_total': pos1[pos]['total'],
                f'{name2}_total': pos2[pos]['total']
            })

    return {
        'shared_same_aa': sorted(same_aa_positions, key=lambda x: x['position']),
        'shared_different_aa': sorted(different_aa_positions, key=lambda x: x['position']),
        'only_in_1': sorted([pos1[p] for p in only_in_1], key=lambda x: x['position']),
        'only_in_2': sorted([pos2[p] for p in only_in_2], key=lambda x: x['position']),
        'stats': {
            'total_conserved_1': len(conserved1),
            'total_conserved_2': len(conserved2),
            'shared_positions': len(shared_positions),
            'same_amino_acid': len(same_aa_positions),
            'different_amino_acid': len(different_aa_positions),
            'unique_to_1': len(only_in_1),
            'unique_to_2': len(only_in_2),
            'overlap_percentage': (len(shared_positions) / max(len(positions1), len(positions2)) * 100) if positions1 or positions2 else 0
        },
        'conserved1': conserved1,
        'conserved2': conserved2
    }

def create_tsne_plot(
    sequences1: Dict[str, str],
    sequences2: Dict[str, str],
    comparison: Dict,
    name1: str,
    name2: str,
    output_dir: str,
    max_sequences: int = 137
):
    """
    Create t-SNE visualization of sequences based on amino acid patterns
    at conserved positions.
    Args:
        sequences1: Dictionary of sequences from first file
        sequences2: Dictionary of sequences from second file
        comparison: Comparison results dictionary
        name1: Name of first protein family
        name2: Name of second protein family
        output_dir: Output directory for saving plot
        max_sequences: Maximum sequences to sample from each group
    """
    print("Generating t-SNE visualization...")
    # Apply publication-quality styling
    setup_publication_style()

    # Get all conserved positions (from both analyses)
    conserved1 = comparison['conserved1']
    conserved2 = comparison['conserved2']

    # Combine all conserved positions
    all_positions = set([c['position'] for c in conserved1] + [c['position'] for c in conserved2])
    all_positions = sorted(list(all_positions))

    if len(all_positions) < 2:
        print("  Not enough conserved positions for t-SNE (need at least 2)")
        return

    # Amino acid to numeric encoding
    aa_list = 'ACDEFGHIKLMNPQRSTVWY-'
    aa_to_num = {aa: i for i, aa in enumerate(aa_list)}

    def encode_sequence(seq, positions):
        """Encode amino acids at given positions as numeric vector."""
        vector = []
        for pos in positions:
            idx = pos - 1  # Convert to 0-indexed
            if idx < len(seq):
                aa = seq[idx].upper()
                vector.append(aa_to_num.get(aa, aa_to_num['-']))
            else:
                vector.append(aa_to_num['-'])
        return vector

    # Sample sequences if too many
    seq_list1 = list(sequences1.items())
    seq_list2 = list(sequences2.items())
    if len(seq_list1) > max_sequences:
        np.random.seed(42)
        indices = np.random.choice(len(seq_list1), max_sequences, replace=False)
        seq_list1 = [seq_list1[i] for i in indices]
    if len(seq_list2) > max_sequences:
        np.random.seed(42)
        indices = np.random.choice(len(seq_list2), max_sequences, replace=False)
        seq_list2 = [seq_list2[i] for i in indices]

    # Encode all sequences
    X = []
    labels = []
    for name, seq in seq_list1:
        X.append(encode_sequence(seq, all_positions))
        labels.append(name1)
    for name, seq in seq_list2:
        X.append(encode_sequence(seq, all_positions))
        labels.append(name2)

    X = np.array(X)
    if len(X) < 5:
        print("  Not enough sequences for t-SNE (need at least 5)")
        return

    # Apply t-SNE
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X)

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['square'])

    # Use publication colors
    color1 = PUBLICATION_COLORS['psy']
    color2 = PUBLICATION_COLORS['crtm']

    mask1 = np.array(labels) == name1
    mask2 = np.array(labels) == name2

    ax.scatter(X_tsne[mask1, 0], X_tsne[mask1, 1], c=color1, alpha=0.7,
               s=60, label=f'{name1} (n={mask1.sum()})', edgecolors='white', linewidth=0.5)
    ax.scatter(X_tsne[mask2, 0], X_tsne[mask2, 1], c=color2, alpha=0.7,
               s=60, label=f'{name2} (n={mask2.sum()})', edgecolors='white', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f't-SNE Clustering of Sequences by Conserved Position Patterns\n'
                 f'(Based on {len(all_positions)} conserved positions)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='#CCCCCC')

    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_clustering.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  t-SNE plot saved with {len(X)} sequences and {len(all_positions)} features")

def create_visualizations(
    comparison: Dict,
    sequences1: Dict[str, str],
    sequences2: Dict[str, str],
    name1: str,
    name2: str,
    output_dir: str
):
    """Create publication-quality visualization plots for the conservation analysis."""
    # Apply publication-quality styling
    setup_publication_style()

    # Use publication colors
    color1 = PUBLICATION_COLORS['psy']
    color2 = PUBLICATION_COLORS['crtm']

    # Pre-calculate arrays and lengths for reuse (removes redundancy)
    conserved1 = comparison['conserved1']
    conserved2 = comparison['conserved2']
    max_len = max(
        max([c['position'] for c in conserved1]) if conserved1 else 0,
        max([c['position'] for c in conserved2]) if conserved2 else 0
    )
    cons_array1 = np.zeros(max_len)
    cons_array2 = np.zeros(max_len)
    for c in conserved1:
        cons_array1[c['position'] - 1] = c['conservation'] * 100
    for c in conserved2:
        cons_array2[c['position'] - 1] = c['conservation'] * 100


    # 1. Summary Bar Chart
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['wide'])
    stats = comparison['stats']
    categories = ['Total\nConserved', 'Shared\nSame AA', 'Shared\nDiff AA', 'Unique']
    x = np.arange(len(categories))
    width = 0.35
    values1 = [stats['total_conserved_1'], stats['same_amino_acid'],
               stats['different_amino_acid'], stats['unique_to_1']]
    values2 = [stats['total_conserved_2'], stats['same_amino_acid'],
               stats['different_amino_acid'], stats['unique_to_2']]

    bars1 = ax.bar(x - width/2, values1, width, label=name1, color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, values2, width, label=name2, color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Number of Positions', fontsize=12)
    ax.set_title('Conservation Analysis Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, padding=3, fontsize=9, fontweight='bold')
    ax.bar_label(bars2, padding=3, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 2. Venn-style Overlap Diagram
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a simple Venn-like representation
    circle1 = plt.Circle((0.35, 0.5), 0.3, color='#3498db', alpha=0.5, label=name1)
    circle2 = plt.Circle((0.65, 0.5), 0.3, color='#e74c3c', alpha=0.5, label=name2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Add labels
    ax.text(0.2, 0.5, f"Unique to {name1}\n{stats['unique_to_1']}",
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.8, 0.5, f"Unique to {name2}\n{stats['unique_to_2']}",
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, f"Shared\n{stats['shared_positions']}\n(Same: {stats['same_amino_acid']}\nDiff: {stats['different_amino_acid']})",
            ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Conservation Position Overlap', fontsize=14, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [mpatches.Patch(facecolor='#3498db', alpha=0.5, label=f'{name1} ({stats["total_conserved_1"]} positions)'),
                       mpatches.Patch(facecolor='#e74c3c', alpha=0.5, label=f'{name2} ({stats["total_conserved_2"]} positions)')]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_diagram.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 3. Conservation Level Distribution (Histogram)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cons_values1 = [c['conservation'] * 100 for c in conserved1]
    cons_values2 = [c['conservation'] * 100 for c in conserved2]

    axes[0].hist(cons_values1, bins=10, range=(80, 100), color='#3498db',
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Conservation Level (%)', fontsize=11)
    axes[0].set_ylabel('Number of Positions', fontsize=11)
    axes[0].set_title(f'{name1} Conservation Distribution', fontsize=12, fontweight='bold')

    axes[1].hist(cons_values2, bins=10, range=(80, 100), color='#e74c3c',
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Conservation Level (%)', fontsize=11)
    axes[1].set_ylabel('Number of Positions', fontsize=11)
    axes[1].set_title(f'{name2} Conservation Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conservation_distribution.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 4. Position-wise conservation comparison (for shared positions)
    if comparison['shared_same_aa'] or comparison['shared_different_aa']:
        fig, ax = plt.subplots(figsize=(14, 6))
        all_shared = comparison['shared_same_aa'] + comparison['shared_different_aa']
        all_shared = sorted(all_shared, key=lambda x: x['position'])

        if all_shared:
            positions = [item['position'] for item in all_shared]
            cons1 = [item[f'{name1}_conservation'] * 100 for item in all_shared]
            cons2 = [item[f'{name2}_conservation'] * 100 for item in all_shared]

            x = np.arange(len(positions))
            width = 0.35

            ax.bar(x - width/2, cons1, width, label=name1, color='#3498db', edgecolor='black')
            ax.bar(x + width/2, cons2, width, label=name2, color='#e74c3c', edgecolor='black')

            ax.set_xlabel('Position', fontsize=11)
            ax.set_ylabel('Conservation (%)', fontsize=11)
            ax.set_title('Conservation Comparison at Shared Positions', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(positions, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=11, framealpha=0.9)
            ax.set_ylim(75, 105)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shared_positions_comparison.png'), dpi=PUBLICATION_DPI,
                        facecolor='white', edgecolor='none', bbox_inches='tight')
            plt.close()


    # 5. Amino Acid Composition at Conserved Positions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    aa_counts1 = Counter([c['amino_acid'] for c in conserved1])
    aa_counts2 = Counter([c['amino_acid'] for c in conserved2])

    # Sort by frequency
    sorted_aa1 = sorted(aa_counts1.items(), key=lambda x: -x[1])
    sorted_aa2 = sorted(aa_counts2.items(), key=lambda x: -x[1])

    if sorted_aa1:
        aas1, counts1 = zip(*sorted_aa1)
        axes[0].bar(aas1, counts1, color='#3498db', edgecolor='black')
        axes[0].set_xlabel('Amino Acid', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'{name1} Conserved Amino Acid Distribution', fontsize=12, fontweight='bold')

    if sorted_aa2:
        aas2, counts2 = zip(*sorted_aa2)
        axes[1].bar(aas2, counts2, color='#e74c3c', edgecolor='black')
        axes[1].set_xlabel('Amino Acid', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title(f'{name2} Conserved Amino Acid Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amino_acid_distribution.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 6. Position Heatmap along sequence
    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

    # Plot using pre-calculated arrays
    im1 = axes[0].imshow([cons_array1], aspect='auto', cmap='Blues', vmin=0, vmax=100)
    axes[0].set_ylabel(name1, fontsize=12)
    axes[0].set_yticks([])
    im2 = axes[1].imshow([cons_array2], aspect='auto', cmap='Reds', vmin=0, vmax=100)
    axes[1].set_ylabel(name2, fontsize=12)
    axes[1].set_yticks([])
    axes[1].set_xlabel('Alignment Position', fontsize=11)

    fig.suptitle('Conservation Landscape Across Alignment', fontsize=14, fontweight='bold')

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, aspect=20)
    cbar1.set_label('Conservation %', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, aspect=20)
    cbar2.set_label('Conservation %', fontsize=10)

    # Highlight Binding Site on heatmap
    highlight_color = PUBLICATION_COLORS['highlight']
    for pos in BINDING_SITE_ALIGNMENT_POSITIONS:
        # Drawing small markers on top and bottom of each series
        for ax_idx in [0, 1]:
            axes[ax_idx].axvline(x=pos-1, color=highlight_color, alpha=0.4, 
                                 linewidth=1.2, linestyle='-', zorder=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conservation_landscape.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 7. 2D NMR-style Conservation Plot with marginal profiles
    # Layout: PSY on left (Y-axis), CrtM on bottom (X-axis), main panel shows connections
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 5],
                          height_ratios=[5, 1],
                          wspace=0.02, hspace=0.02)

    ax_main = fig.add_subplot(gs[0, 1])      # Main 2D panel (upper right)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)  # Bottom marginal (CrtM)
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)    # Left marginal (PSY)
    ax_empty = fig.add_subplot(gs[1, 0])     # Empty corner (bottom left)
    ax_empty.axis('off')

    positions = np.arange(max_len)  # 0-indexed for plotting
    conservation_threshold = 80
    
    # Find positions where both are conserved (above threshold)
    both_conserved_mask = (cons_array1 >= conservation_threshold) & (cons_array2 >= conservation_threshold)
    both_conserved_indices = np.where(both_conserved_mask)[0]
    
    # Main panel: Draw connection lines for conserved positions
    ax_main.set_facecolor('#fafafa')
    
    # Draw diagonal reference line
    ax_main.plot([0, max_len-1], [0, max_len-1], 'k--', alpha=0.2, linewidth=1.0)
    
    # For each conserved position, draw cross-peak style markers
    for idx in both_conserved_indices:
        # Draw vertical line from bottom marginal up to the diagonal point
        ax_main.plot([idx, idx], [0, idx], color=PUBLICATION_COLORS['shared'], 
                     alpha=0.5, linewidth=1.2)
        # Draw horizontal line from left marginal to the diagonal point
        ax_main.plot([0, idx], [idx, idx], color=PUBLICATION_COLORS['shared'], 
                     alpha=0.5, linewidth=1.2)
        # Draw marker at the diagonal intersection point
        ax_main.scatter([idx], [idx], color=PUBLICATION_COLORS['shared'], 
                        s=50, zorder=5, edgecolors='black', linewidths=0.5)
    
    # Bottom marginal: CrtM conservation profile (X-axis)
    ax_bottom.fill_between(positions, 0, cons_array2, color=PUBLICATION_COLORS['crtm'], alpha=0.7)
    ax_bottom.plot(positions, cons_array2, color=PUBLICATION_COLORS['crtm'], linewidth=1.0)
    ax_bottom.axhline(y=conservation_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_bottom.set_ylim(105, 0)  # Inverted so peaks point up toward main panel
    ax_bottom.set_xlabel(f'{name2} - Alignment Position', fontsize=16, fontweight='bold')
    ax_bottom.set_ylabel('Cons. (%)', fontsize=9, fontweight='bold')
    ax_bottom.tick_params(axis='both', labelsize=8)
    ax_bottom.spines['bottom'].set_visible(True)
    ax_bottom.spines['left'].set_visible(True)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    
    # Mark conserved positions on bottom marginal
    for idx in both_conserved_indices:
        ax_bottom.axvline(x=idx, color=PUBLICATION_COLORS['shared'], alpha=0.3, linewidth=0.8)

    # Left marginal: PSY conservation profile (Y-axis)
    ax_left.fill_betweenx(positions, 0, cons_array1, color=PUBLICATION_COLORS['psy'], alpha=0.7)
    ax_left.plot(cons_array1, positions, color=PUBLICATION_COLORS['psy'], linewidth=1.0)
    ax_left.axvline(x=conservation_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_left.set_xlim(105, 0)  # Inverted so peaks point right toward main panel
    ax_left.set_ylabel(f'{name1} - Alignment Position', fontsize=16, fontweight='bold')
    ax_left.set_xlabel('Cons. (%)', fontsize=9, fontweight='bold')
    ax_left.tick_params(axis='both', labelsize=8)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['bottom'].set_visible(True)
    ax_left.spines['left'].set_visible(True)
    
    # Mark conserved positions on left marginal
    for idx in both_conserved_indices:
        ax_left.axhline(y=idx, color=PUBLICATION_COLORS['shared'], alpha=0.3, linewidth=0.8)

    # Highlight Binding Site on 2D Plot marginals
    highlight_color = PUBLICATION_COLORS['highlight']
    for pos in BINDING_SITE_ALIGNMENT_POSITIONS:
        idx = pos - 1
        # Optional: subtle markers on main panel edges
        ax_main.scatter([idx], [-0.5], marker='^', color=highlight_color, s=20, clip_on=False, zorder=10)
        ax_main.scatter([-0.5], [idx], marker='>', color=highlight_color, s=20, clip_on=False, zorder=10)

    # Set main panel limits
    ax_main.set_xlim(-0.5, max_len - 0.5)
    ax_main.set_ylim(-0.5, max_len - 0.5)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#333333')

    # Add Legend at the bottom of the figure in a horizontal format
    legend_elements = [
        mpatches.Patch(facecolor=PUBLICATION_COLORS['psy'], alpha=0.7, label=f'{name1} Profile'),
        mpatches.Patch(facecolor=PUBLICATION_COLORS['crtm'], alpha=0.7, label=f'{name2} Profile'),
        mlines.Line2D([0], [0], color=PUBLICATION_COLORS['shared'], marker='o', linestyle='None',
                      markersize=8, markeredgecolor='black', markeredgewidth=0.5, 
                      label='Both Conserved'),
        mlines.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.8, 
                      label=f'Threshold ({conservation_threshold}%)'),
        mlines.Line2D([0], [0], color=PUBLICATION_COLORS['highlight'], marker='>', linestyle='None',
                      markersize=6, label='Binding Site')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=16,
               bbox_to_anchor=(0.5, 0.03), frameon=True, edgecolor='#CCCCCC', facecolor='#fafafa')

    # Add extra padding at the bottom for the larger legend and top for the larger title
    plt.subplots_adjust(bottom=0.15, top=0.90)

    # Title with count
    fig.suptitle(f'2D Conservation Plot: {name1} vs {name2}\n'
                 f'Positions conserved in both (≥{conservation_threshold}%): {len(both_conserved_indices)}', 
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(output_dir, 'conservation_landscape_2d.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


    # 8. t-SNE visualization of sequences based on conserved positions
    create_tsne_plot(sequences1, sequences2, comparison, name1, name2, output_dir)

    print(f"Visualizations saved to: {output_dir}")

# ... (Other functions like generate_report, subsample_sequences, run_subsampled_analysis, etc., remain unchanged) ...

def generate_report(
    file1: str,
    file2: str,
    threshold: float = 0.8,
    min_coverage: float = 0.5,
    output_dir: str = None
) -> str:
    """
    Generate a comprehensive conservation analysis report.
    Args:
        file1: Path to first FASTA file
        file2: Path to second FASTA file
        threshold: Conservation threshold (0.0 to 1.0)
        min_coverage: Minimum coverage (fraction of sequences with data) for significance
        output_dir: Directory to save output files (optional)
    Returns:
        Report as string
    """
    # Standardize names for plotting/reporting
    name_map = {'finalpsy': 'PSY', 'finalcrtm': 'CrtM'}
    name1_raw = os.path.basename(file1).replace('.fasta', '')
    name1 = name_map.get(name1_raw.lower(), name1_raw)
    name2_raw = os.path.basename(file2).replace('.fasta', '')
    name2 = name_map.get(name2_raw.lower(), name2_raw)

    # Analyze both files
    print(f"Analyzing {file1}...")
    sequences1, conserved1 = analyze_fasta_file(file1, threshold, min_coverage)
    print(f"Analyzing {file2}...")
    sequences2, conserved2 = analyze_fasta_file(file2, threshold, min_coverage)

    # Compare conserved positions
    comparison = compare_conserved_positions(conserved1, conserved2, name1, name2)

    # Generate publication-quality report
    from datetime import datetime

    # Define formatting helpers
    def format_table_row(cols, widths, align='left'):
        """Format a table row with proper alignment."""
        row = "│"
        for i, (col, width) in enumerate(zip(cols, widths)):
            if align == 'right' or (isinstance(align, list) and align[i] == 'right'):
                row += f" {str(col):>{width}} │"
            elif align == 'center' or (isinstance(align, list) and align[i] == 'center'):
                row += f" {str(col):^{width}} │"
            else:
                row += f" {str(col):<{width}} │"
        return row

    def format_table_border(widths, style='mid'):
        """Format table borders with Unicode box characters."""
        if style == 'top':
            return "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        elif style == 'mid':
            return "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        elif style == 'bottom':
            return "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"
        elif style == 'double':
            return "╠" + "╬".join("═" * (w + 2) for w in widths) + "╣"

    report = []

    # Title block with professional styling
    report.append("╔" + "═" * 78 + "╗")
    report.append("║" + " " * 78 + "║")
    report.append("║" + "CONSERVED AMINO ACID COMPARATIVE ANALYSIS".center(78) + "║")
    report.append("║" + " " * 78 + "║")
    report.append("╚" + "═" * 78 + "╝")
    report.append("")

    # Analysis metadata
    report.append("─" * 80)
    report.append("ANALYSIS METADATA")
    report.append("─" * 80)
    report.append(f"  Analysis Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Conservation Threshold:  {threshold:.0%} (minimum residue frequency)")
    report.append(f"  Coverage Filter:         {min_coverage:.0%} (minimum non-gap sequences)")
    report.append("")

    # Input files summary table
    report.append("─" * 80)
    report.append("INPUT DATA SUMMARY")
    report.append("─" * 80)
    widths = [12, 25, 12, 16]
    headers = ["Dataset", "File", "Sequences", "Alignment Len."]
    report.append(format_table_border(widths, 'top'))
    report.append(format_table_row(headers, widths, 'center'))
    report.append(format_table_border(widths, 'double'))
    report.append(format_table_row([name1, os.path.basename(file1), len(sequences1), get_alignment_length(sequences1)], widths, ['left', 'left', 'right', 'right']))
    report.append(format_table_row([name2, os.path.basename(file2), len(sequences2), get_alignment_length(sequences2)], widths, ['left', 'left', 'right', 'right']))
    report.append(format_table_border(widths, 'bottom'))
    report.append("")

    # Statistical summary
    stats = comparison['stats']
    report.append("─" * 80)
    report.append("CONSERVATION STATISTICS SUMMARY")
    report.append("─" * 80)
    # Statistics table
    widths = [40, 10, 10]
    headers = ["Metric", name1, name2]
    report.append(format_table_border(widths, 'top'))
    report.append(format_table_row(headers, widths, ['left', 'center', 'center']))
    report.append(format_table_border(widths, 'double'))
    report.append(format_table_row(["Total conserved positions", stats['total_conserved_1'], stats['total_conserved_2']], widths, ['left', 'right', 'right']))
    report.append(format_table_row(["Unique conserved positions", stats['unique_to_1'], stats['unique_to_2']], widths, ['left', 'right', 'right']))
    report.append(format_table_border(widths, 'mid'))
    report.append(format_table_row(["Shared conserved positions", stats['shared_positions'], "—"], widths, ['left', 'right', 'center']))
    report.append(format_table_row(["  └─ Same amino acid", stats['same_amino_acid'], "—"], widths, ['left', 'right', 'center']))
    report.append(format_table_row(["  └─ Different amino acid", stats['different_amino_acid'], "—"], widths, ['left', 'right', 'center']))
    report.append(format_table_border(widths, 'mid'))
    report.append(format_table_row([f"Position overlap", f"{stats['overlap_percentage']:.1f}%", "—"], widths, ['left', 'right', 'center']))
    report.append(format_table_border(widths, 'bottom'))
    report.append("")

    # Shared conserved positions with same amino acid
    report.append("─" * 80)
    report.append("TABLE 1: SHARED CONSERVED POSITIONS (Identical Residues)")
    report.append("─" * 80)
    if comparison['shared_same_aa']:
        widths = [8, 6, 12, 12, 12, 12]
        headers = ["Position", "Residue", f"{name1} %", f"{name2} %", f"{name1} n", f"{name2} n"]
        report.append(format_table_border(widths, 'top'))
        report.append(format_table_row(headers, widths, 'center'))
        report.append(format_table_border(widths, 'double'))
        for item in comparison['shared_same_aa']:
            cons1 = f"{item[f'{name1}_conservation']*100:.1f}"
            cons2 = f"{item[f'{name2}_conservation']*100:.1f}"
            total1 = str(item[f'{name1}_total'])
            total2 = str(item[f'{name2}_total'])
            report.append(format_table_row(
                [item['position'], item['amino_acid'], cons1, cons2, total1, total2],
                widths, ['center', 'center', 'right', 'right', 'right', 'right']
            ))
        report.append(format_table_border(widths, 'bottom'))
        report.append(f"  Note: n = number of sequences with non-gap residue at position")
    else:
        report.append("  No shared positions with identical amino acids found.")
    report.append("")

    # Shared conserved positions with different amino acids
    report.append("─" * 80)
    report.append("TABLE 2: SHARED CONSERVED POSITIONS (Divergent Residues)")
    report.append("─" * 80)
    if comparison['shared_different_aa']:
        widths = [8, 10, 10, 12, 12]
        headers = ["Position", f"{name1} AA", f"{name2} AA", f"{name1} %", f"{name2} %"]
        report.append(format_table_border(widths, 'top'))
        report.append(format_table_row(headers, widths, 'center'))
        report.append(format_table_border(widths, 'double'))
        for item in comparison['shared_different_aa']:
            cons1 = f"{item[f'{name1}_conservation']*100:.1f}"
            cons2 = f"{item[f'{name2}_conservation']*100:.1f}"
            report.append(format_table_row(
                [item['position'], item[f'{name1}_aa'], item[f'{name2}_aa'], cons1, cons2],
                widths, ['center', 'center', 'center', 'right', 'right']
            ))
        report.append(format_table_border(widths, 'bottom'))
    else:
        report.append("  No shared positions with divergent amino acids found.")
    report.append("")

    # Unique to file 1
    report.append("─" * 80)
    report.append(f"TABLE 3: CONSERVED POSITIONS UNIQUE TO {name1.upper()}")
    report.append("─" * 80)
    if comparison['only_in_1']:
        widths = [8, 8, 16, 8, 8]
        headers = ["Position", "Residue", "Conservation (%)", "Count", "Total"]
        report.append(format_table_border(widths, 'top'))
        report.append(format_table_row(headers, widths, 'center'))
        report.append(format_table_border(widths, 'double'))
        for item in comparison['only_in_1']:
            cons = f"{item['conservation']*100:.1f}"
            report.append(format_table_row(
                [item['position'], item['amino_acid'], cons, item['count'], item['total']],
                widths, ['center', 'center', 'right', 'right', 'right']
            ))
        report.append(format_table_border(widths, 'bottom'))
    else:
        report.append(f"  No positions unique to {name1}.")
    report.append("")

    # Unique to file 2
    report.append("─" * 80)
    report.append(f"TABLE 4: CONSERVED POSITIONS UNIQUE TO {name2.upper()}")
    report.append("─" * 80)
    if comparison['only_in_2']:
        widths = [8, 8, 16, 8, 8]
        headers = ["Position", "Residue", "Conservation (%)", "Count", "Total"]
        report.append(format_table_border(widths, 'top'))
        report.append(format_table_row(headers, widths, 'center'))
        report.append(format_table_border(widths, 'double'))
        for item in comparison['only_in_2']:
            cons = f"{item['conservation']*100:.1f}"
            report.append(format_table_row(
                [item['position'], item['amino_acid'], cons, item['count'], item['total']],
                widths, ['center', 'center', 'right', 'right', 'right']
            ))
        report.append(format_table_border(widths, 'bottom'))
    else:
        report.append(f"  No positions unique to {name2}.")
    report.append("")

    # Footer
    report.append("─" * 80)
    report.append("END OF REPORT")
    report.append("─" * 80)
    report_str = '\n'.join(report)

    # Save outputs if directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save report
        report_file = os.path.join(output_dir, 'conservation_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"Report saved to: {report_file}")

        # Save detailed CSVs
        if comparison['shared_same_aa']:
            df_same = pd.DataFrame(comparison['shared_same_aa'])
            df_same.to_csv(os.path.join(output_dir, 'shared_same_aa.csv'), index=False)
            print(f"Shared same AA saved to: {os.path.join(output_dir, 'shared_same_aa.csv')}")
        if comparison['shared_different_aa']:
            df_diff = pd.DataFrame(comparison['shared_different_aa'])
            df_diff.to_csv(os.path.join(output_dir, 'shared_different_aa.csv'), index=False)
            print(f"Shared different AA saved to: {os.path.join(output_dir, 'shared_different_aa.csv')}")
        if comparison['only_in_1']:
            df_1 = pd.DataFrame(comparison['only_in_1'])
            df_1.to_csv(os.path.join(output_dir, f'{name1}_unique_conserved.csv'), index=False)
            print(f"{name1} unique positions saved to: {os.path.join(output_dir, f'{name1}_unique_conserved.csv')}")
        if comparison['only_in_2']:
            df_2 = pd.DataFrame(comparison['only_in_2'])
            df_2.to_csv(os.path.join(output_dir, f'{name2}_unique_conserved.csv'), index=False)
            print(f"{name2} unique positions saved to: {os.path.join(output_dir, f'{name2}_unique_conserved.csv')}")

        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(comparison, sequences1, sequences2, name1, name2, output_dir)

    return report_str, comparison

# ... (Other functions like subsample_sequences, run_subsampled_analysis, etc., remain unchanged) ...

def subsample_sequences(sequences: Dict[str, str], n_samples: int, seed: int = None) -> Dict[str, str]:
    """
    Randomly subsample sequences to a specified size.
    Args:
        sequences: Dictionary of sequence name -> sequence
        n_samples: Number of sequences to sample
        seed: Random seed for reproducibility
    Returns:
        Subsampled dictionary of sequences
    """
    if n_samples >= len(sequences):
        return sequences
    np.random.seed(seed)
    keys = list(sequences.keys())
    sampled_keys = np.random.choice(keys, size=n_samples, replace=False)
    return {k: sequences[k] for k in sampled_keys}

def run_subsampled_analysis(
    file1: str,
    file2: str,
    threshold: float = 0.8,
    min_coverage: float = 0.5,
    n_iterations: int = 5,
    output_dir: str = None
) -> Dict:
    """
    Run conservation analysis with subsampling for fair comparison.
    The larger dataset is subsampled to match the smaller one's size.
    Multiple iterations are run with different random seeds to get robust results.
    Args:
        file1: Path to first FASTA file
        file2: Path to second FASTA file
        threshold: Conservation threshold
        min_coverage: Minimum coverage threshold
        n_iterations: Number of bootstrap iterations
        output_dir: Output directory
    Returns:
        Dictionary with aggregated results
    """
    # Standardize names for plotting/reporting
    name_map = {'finalpsy': 'PSY', 'finalcrtm': 'CrtM'}
    name1_raw = os.path.basename(file1).replace('.fasta', '')
    name1 = name_map.get(name1_raw.lower(), name1_raw)
    name2_raw = os.path.basename(file2).replace('.fasta', '')
    name2 = name_map.get(name2_raw.lower(), name2_raw)

    # Parse both files
    print(f"Loading {file1}...")
    all_sequences1 = parse_fasta(file1)
    print(f"  Loaded {len(all_sequences1)} sequences")
    print(f"Loading {file2}...")
    all_sequences2 = parse_fasta(file2)
    print(f"  Loaded {len(all_sequences2)} sequences")

    # Determine which is larger and target size
    n1, n2 = len(all_sequences1), len(all_sequences2)
    target_size = min(n1, n2)
    larger_is_1 = n1 > n2

    print(f"\n{'='*60}")
    print(f"SUBSAMPLING FOR FAIR COMPARISON")
    print(f"{'='*60}")
    print(f"  {name1}: {n1} sequences")
    print(f"  {name2}: {n2} sequences")
    print(f"  Target size: {target_size} sequences")
    print(f"  Subsampling: {name1 if larger_is_1 else name2}")
    print(f"  Bootstrap iterations: {n_iterations}")
    print(f"{'='*60}\n")

    # Store results from each iteration
    all_results = []
    position_counts = {}  # Track how often each position appears as conserved

    for i in range(n_iterations):
        seed = 42 + i
        print(f"Iteration {i+1}/{n_iterations} (seed={seed})...")

        # Subsample the larger dataset
        if larger_is_1:
            sequences1 = subsample_sequences(all_sequences1, target_size, seed)
            sequences2 = all_sequences2
        else:
            sequences1 = all_sequences1
            sequences2 = subsample_sequences(all_sequences2, target_size, seed)

        # Calculate conservation for this iteration
        conserved1 = calculate_conservation(sequences1, threshold, min_coverage)
        conserved2 = calculate_conservation(sequences2, threshold, min_coverage)

        # Compare
        comparison = compare_conserved_positions(conserved1, conserved2, name1, name2)
        comparison['iteration'] = i + 1
        comparison['seed'] = seed
        all_results.append(comparison)

        # Track positions
        for c in conserved1:
            key = (name1, c['position'], c['amino_acid'])
            position_counts[key] = position_counts.get(key, 0) + 1
        for c in conserved2:
            key = (name2, c['position'], c['amino_acid'])
            position_counts[key] = position_counts.get(key, 0) + 1

        print(f"  {name1}: {len(conserved1)} conserved, {name2}: {len(conserved2)} conserved, "
              f"Shared: {comparison['stats']['shared_positions']}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS ACROSS ITERATIONS")
    print(f"{'='*60}")

    # Calculate statistics across iterations
    stats_across = {
        'total_conserved_1': [r['stats']['total_conserved_1'] for r in all_results],
        'total_conserved_2': [r['stats']['total_conserved_2'] for r in all_results],
        'shared_positions': [r['stats']['shared_positions'] for r in all_results],
        'same_amino_acid': [r['stats']['same_amino_acid'] for r in all_results],
        'different_amino_acid': [r['stats']['different_amino_acid'] for r in all_results],
        'unique_to_1': [r['stats']['unique_to_1'] for r in all_results],
        'unique_to_2': [r['stats']['unique_to_2'] for r in all_results],
    }

    # Positions that appear in majority of iterations (>=50%)
    min_appearances = n_iterations // 2 + 1
    robust_positions_1 = []
    robust_positions_2 = []
    for (dataset, pos, aa), count in position_counts.items():
        if count >= min_appearances:
            if dataset == name1:
                robust_positions_1.append({
                    'position': pos,
                    'amino_acid': aa,
                    'appearances': count,
                    'frequency': count / n_iterations
                })
            else:
                robust_positions_2.append({
                    'position': pos,
                    'amino_acid': aa,
                    'appearances': count,
                    'frequency': count / n_iterations
                })

    robust_positions_1 = sorted(robust_positions_1, key=lambda x: (-x['frequency'], x['position']))
    robust_positions_2 = sorted(robust_positions_2, key=lambda x: (-x['frequency'], x['position']))

    # Find shared robust positions
    pos_set_1 = {(p['position'], p['amino_acid']) for p in robust_positions_1}
    pos_set_2 = {(p['position'], p['amino_acid']) for p in robust_positions_2}
    shared_robust_same_aa = pos_set_1 & pos_set_2

    # Generate summary report
    report = []
    report.append("=" * 80)
    report.append("SUBSAMPLED CONSERVATION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("ANALYSIS PARAMETERS:")
    report.append(f"  {name1}: {n1} sequences (subsampled to {target_size})" if larger_is_1 else f"  {name1}: {n1} sequences")
    report.append(f"  {name2}: {n2} sequences (subsampled to {target_size})" if not larger_is_1 else f"  {name2}: {n2} sequences")
    report.append(f"  Conservation threshold: {threshold:.0%}")
    report.append(f"  Minimum coverage: {min_coverage:.0%}")
    report.append(f"  Bootstrap iterations: {n_iterations}")
    report.append("")

    report.append("=" * 80)
    report.append("SUMMARY STATISTICS (Mean ± Std across iterations)")
    report.append("=" * 80)
    for stat_name, values in stats_across.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        report.append(f"  {stat_name.replace('_', ' ').title()}: {mean_val:.1f} ± {std_val:.1f}")
    report.append("")

    report.append("=" * 80)
    report.append(f"ROBUST CONSERVED POSITIONS (appearing in >={min_appearances}/{n_iterations} iterations)")
    report.append("=" * 80)
    report.append("")
    report.append(f"--- {name1.upper()} ({len(robust_positions_1)} positions) ---")
    report.append(f"{'Position':<10} {'AA':<6} {'Appearances':<15} {'Frequency':<10}")
    report.append("-" * 41)
    for p in robust_positions_1:
        report.append(f"{p['position']:<10} {p['amino_acid']:<6} {p['appearances']}/{n_iterations:<12} {p['frequency']*100:.0f}%")
    report.append("")
    report.append(f"--- {name2.upper()} ({len(robust_positions_2)} positions) ---")
    report.append(f"{'Position':<10} {'AA':<6} {'Appearances':<15} {'Frequency':<10}")
    report.append("-" * 41)
    for p in robust_positions_2:
        report.append(f"{p['position']:<10} {p['amino_acid']:<6} {p['appearances']}/{n_iterations:<12} {p['frequency']*100:.0f}%")
    report.append("")

    report.append("=" * 80)
    report.append(f"SHARED ROBUST CONSERVED POSITIONS (Same AA in both)")
    report.append("=" * 80)
    shared_list = sorted(list(shared_robust_same_aa), key=lambda x: x[0])
    if shared_list:
        for pos, aa in shared_list:
            report.append(f"  Position {pos}: {aa}")
    else:
        report.append("  No shared robust positions with same amino acid found.")
    report.append("")
    report.append(f"Total shared robust positions: {len(shared_robust_same_aa)}")

    report_str = '\n'.join(report)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, 'subsampled_conservation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_str)
        print(f"\nReport saved to: {report_file}")

        # Save robust positions as CSV
        if robust_positions_1:
            df1 = pd.DataFrame(robust_positions_1)
            df1.to_csv(os.path.join(output_dir, f'{name1}_robust_conserved.csv'), index=False)
        if robust_positions_2:
            df2 = pd.DataFrame(robust_positions_2)
            df2.to_csv(os.path.join(output_dir, f'{name2}_robust_conserved.csv'), index=False)

        # Save iteration details
        iteration_stats = []
        for r in all_results:
            iteration_stats.append({
                'iteration': r['iteration'],
                'seed': r['seed'],
                f'{name1}_conserved': r['stats']['total_conserved_1'],
                f'{name2}_conserved': r['stats']['total_conserved_2'],
                'shared': r['stats']['shared_positions'],
                'same_aa': r['stats']['same_amino_acid'],
                'different_aa': r['stats']['different_amino_acid']
            })
        df_iterations = pd.DataFrame(iteration_stats)
        df_iterations.to_csv(os.path.join(output_dir, 'iteration_statistics.csv'), index=False)

        # Create visualization of results across iterations
        create_subsample_visualizations(stats_across, name1, name2, n_iterations,
                                        robust_positions_1, robust_positions_2,
                                        shared_robust_same_aa, output_dir)

    return {
        'all_results': all_results,
        'stats_across': stats_across,
        'robust_positions_1': robust_positions_1,
        'robust_positions_2': robust_positions_2,
        'shared_robust_same_aa': shared_robust_same_aa,
        'report': report_str
    }

def create_subsample_visualizations(
    stats_across: Dict,
    name1: str,
    name2: str,
    n_iterations: int,
    robust_1: List[Dict],
    robust_2: List[Dict],
    shared: set,
    output_dir: str
):
    """Create publication-quality visualizations for subsampled analysis."""
    # Apply publication-quality styling
    setup_publication_style()

    # 1. Box plot of statistics across iterations
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['landscape'])

    # Total conserved
    ax = axes[0, 0]
    data = [stats_across['total_conserved_1'], stats_across['total_conserved_2']]
    bp = ax.boxplot(data, tick_labels=[name1, name2], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel('Number of Positions')
    ax.set_title('Total Conserved Positions\n(across iterations)')

    # Shared positions
    ax = axes[0, 1]
    ax.boxplot(stats_across['shared_positions'], patch_artist=True)
    ax.set_ylabel('Number of Positions')
    ax.set_title('Shared Conserved Positions\n(across iterations)')

    # Unique positions
    ax = axes[1, 0]
    data = [stats_across['unique_to_1'], stats_across['unique_to_2']]
    bp = ax.boxplot(data, tick_labels=[name1, name2], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel('Number of Positions')
    ax.set_title('Unique Conserved Positions\n(across iterations)')

    # Line plot showing values across iterations
    ax = axes[1, 1]
    x = range(1, n_iterations + 1)
    ax.plot(x, stats_across['total_conserved_1'], 'o-', color='#3498db', label=f'{name1} total')
    ax.plot(x, stats_across['total_conserved_2'], 'o-', color='#e74c3c', label=f'{name2} total')
    ax.plot(x, stats_across['shared_positions'], 's--', color='#27ae60', label='Shared')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Positions')
    ax.set_title('Conservation Across Iterations')
    ax.legend()
    ax.set_xticks(list(x))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsample_statistics.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    # 2. Robust position frequency bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if robust_1:
        ax = axes[0]
        positions = [p['position'] for p in robust_1[:20]]  # Top 20
        frequencies = [p['frequency'] * 100 for p in robust_1[:20]]
        colors = ['#27ae60' if (p['position'], p['amino_acid']) in shared else '#3498db' for p in robust_1[:20]]
        ax.barh(range(len(positions)), frequencies, color=colors)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels([f"{p} ({robust_1[i]['amino_acid']})" for i, p in enumerate(positions)])
        ax.set_xlabel('Appearance Frequency (%)')
        ax.set_title(f'{name1} Robust Conserved Positions')
        ax.invert_yaxis()
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')

    if robust_2:
        ax = axes[1]
        positions = [p['position'] for p in robust_2[:20]]
        frequencies = [p['frequency'] * 100 for p in robust_2[:20]]
        colors = ['#27ae60' if (p['position'], p['amino_acid']) in shared else '#e74c3c' for p in robust_2[:20]]
        ax.barh(range(len(positions)), frequencies, color=colors)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels([f"{p} ({robust_2[i]['amino_acid']})" for i, p in enumerate(positions)])
        ax.set_xlabel('Appearance Frequency (%)')
        ax.set_title(f'{name2} Robust Conserved Positions')
        ax.invert_yaxis()
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', label='Shared'),
                       Patch(facecolor='#3498db', label=f'{name1} unique'),
                       Patch(facecolor='#e74c3c', label=f'{name2} unique')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(output_dir, 'robust_positions.png'), dpi=PUBLICATION_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Subsample visualizations saved to: {output_dir}")


# =============================================================================
# STRUCTURE MAPPING FUNCTIONS
# =============================================================================

def map_conserved_to_structure(
    comparison: Dict,
    msa_file: str,
    pdb_file: str,
    output_dir: str,
    name1: str = "PSY",
    name2: str = "CrtM",
    chain: str = 'A'
) -> Dict:
    """
    Map conserved positions from analysis to PDB structure residue numbers.
    
    Args:
        comparison: Comparison results from compare_conserved_positions
        msa_file: Path to the MSA FASTA file that corresponds to the structure
        pdb_file: Path to PDB structure file
        output_dir: Directory for output files
        name1: Name of first protein family
        name2: Name of second protein family
        chain: PDB chain identifier
    
    Returns:
        Dictionary with mapping results and output file paths
    """
    # Import structure mapper functions
    from structure_mapper import (
        parse_fasta as sm_parse_fasta,
        get_pdb_info,
        get_representative_sequence,
        simple_align,
        get_alignment_stats,
        create_msa_position_map,
        generate_pymol_script,
        generate_chimerax_script,
        save_mapping_csv
    )
    
    print(f"\n{'='*60}")
    print("MAPPING CONSERVED POSITIONS TO STRUCTURE")
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
    
    # Collect all conserved positions and their categories
    residue_groups = {
        'conserved_same': [],      # Shared conserved, same AA
        'conserved_different': [], # Shared conserved, different AA
        f'unique_{name1.lower()}': [],  # Unique to family 1
        f'unique_{name2.lower()}': []   # Unique to family 2
    }
    
    all_mappings = []
    
    # Process shared same AA positions
    print(f"\nMapping shared conserved positions (same AA)...")
    for item in comparison['shared_same_aa']:
        msa_pos = item['position']
        mapping = _map_single_position(msa_pos, msa_position_map, ungapped_to_pdb)
        mapping['category'] = 'shared_same_aa'
        mapping['amino_acid'] = item['amino_acid']
        all_mappings.append(mapping)
        
        if mapping['mapped']:
            residue_groups['conserved_same'].append(mapping['pdb_residue_number'])
    
    # Process shared different AA positions
    print(f"Mapping shared conserved positions (different AA)...")
    for item in comparison['shared_different_aa']:
        msa_pos = item['position']
        mapping = _map_single_position(msa_pos, msa_position_map, ungapped_to_pdb)
        mapping['category'] = 'shared_different_aa'
        mapping[f'{name1}_aa'] = item[f'{name1}_aa']
        mapping[f'{name2}_aa'] = item[f'{name2}_aa']
        all_mappings.append(mapping)
        
        if mapping['mapped']:
            residue_groups['conserved_different'].append(mapping['pdb_residue_number'])
    
    # Process unique to family 1
    print(f"Mapping positions unique to {name1}...")
    for item in comparison['only_in_1']:
        msa_pos = item['position']
        mapping = _map_single_position(msa_pos, msa_position_map, ungapped_to_pdb)
        mapping['category'] = f'unique_{name1}'
        mapping['amino_acid'] = item['amino_acid']
        all_mappings.append(mapping)
        
        if mapping['mapped']:
            residue_groups[f'unique_{name1.lower()}'].append(mapping['pdb_residue_number'])
    
    # Process unique to family 2
    print(f"Mapping positions unique to {name2}...")
    for item in comparison['only_in_2']:
        msa_pos = item['position']
        mapping = _map_single_position(msa_pos, msa_position_map, ungapped_to_pdb)
        mapping['category'] = f'unique_{name2}'
        mapping['amino_acid'] = item['amino_acid']
        all_mappings.append(mapping)
        
        if mapping['mapped']:
            residue_groups[f'unique_{name2.lower()}'].append(mapping['pdb_residue_number'])
    
    # Print summary
    print(f"\n{'='*60}")
    print("MAPPING SUMMARY")
    print(f"{'='*60}")
    total_mapped = sum(len(v) for v in residue_groups.values())
    print(f"  Total positions mapped: {total_mapped}")
    for group, residues in residue_groups.items():
        print(f"    {group}: {len(residues)} residues")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'conserved_positions_pdb_mapping.csv')
    _save_conserved_mapping_csv(all_mappings, csv_path, name1, name2)
    
    # Generate PyMOL script
    pml_path = os.path.join(output_dir, 'visualize_conserved.pml')
    color_scheme = {
        'conserved_same': 'green',
        'conserved_different': 'yellow',
        f'unique_{name1.lower()}': 'marine',
        f'unique_{name2.lower()}': 'salmon'
    }
    generate_pymol_script(pdb_file, residue_groups, pml_path, color_scheme)
    
    # Generate ChimeraX script
    cxc_path = os.path.join(output_dir, 'visualize_conserved.cxc')
    chimerax_color_scheme = {
        'conserved_same': 'green',
        'conserved_different': 'gold',
        f'unique_{name1.lower()}': 'dodger blue',
        f'unique_{name2.lower()}': 'salmon'
    }
    generate_chimerax_script(pdb_file, residue_groups, cxc_path, chimerax_color_scheme)
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  CSV mapping: {csv_path}")
    print(f"  PyMOL script: {pml_path}")
    print(f"  ChimeraX script: {cxc_path}")
    print(f"\nTo visualize, run:")
    print(f"  PyMOL: pymol {pml_path}")
    print(f"  ChimeraX: chimerax {cxc_path}")
    
    return {
        'mappings': all_mappings,
        'residue_groups': residue_groups,
        'pdb_info': pdb_info,
        'alignment_stats': stats,
        'csv_path': csv_path,
        'pml_path': pml_path,
        'cxc_path': cxc_path
    }


def _map_single_position(
    msa_pos: int,
    msa_position_map: Dict[int, int],
    ungapped_to_pdb: Dict[int, int]
) -> Dict:
    """Helper function to map a single MSA position to PDB residue number."""
    result = {
        'msa_position': msa_pos,
        'ungapped_position': None,
        'pdb_residue_number': None,
        'mapped': False
    }
    
    if msa_pos in msa_position_map:
        ungapped_pos = msa_position_map[msa_pos]
        result['ungapped_position'] = ungapped_pos
        
        if ungapped_pos in ungapped_to_pdb:
            result['pdb_residue_number'] = ungapped_to_pdb[ungapped_pos]
            result['mapped'] = True
    
    return result


def _save_conserved_mapping_csv(
    mappings: List[Dict],
    output_path: str,
    name1: str,
    name2: str
) -> str:
    """Save conserved position mappings to CSV."""
    import csv
    
    # Collect all possible columns
    all_columns = set()
    for m in mappings:
        all_columns.update(m.keys())
    
    # Order columns logically
    base_columns = ['msa_position', 'ungapped_position', 'pdb_residue_number', 'mapped', 'category']
    other_columns = sorted(all_columns - set(base_columns))
    columns = base_columns + other_columns
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mappings)
    
    print(f"  Mapping CSV saved: {output_path}")
    return output_path


def main():
    """Main function to run the analysis."""
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file1 = os.path.join(base_dir, "finalpsy.fasta")
    file2 = os.path.join(base_dir, "finalcrtm.fasta")
    output_dir = os.path.join(base_dir, "conservation_analysis_output")
    
    # PDB structure file for visualization mapping
    pdb_file = os.path.join(base_dir, "3W7F_A.pdb")
    
    print("=" * 60)
    print("CONSERVED AMINO ACID ANALYSIS WITH SUBSAMPLING")
    print("=" * 60)
    print("This analysis subsamples the larger dataset to match")
    print("the smaller one for fair comparison.")
    print("Multiple bootstrap iterations are run for robust results.")
    print("=" * 60)
    
    # Run subsampled analysis with 5 iterations
    results = run_subsampled_analysis(
        file1=file1,
        file2=file2,
        threshold=0.8,
        min_coverage=0.5,
        n_iterations=1000,
        output_dir=output_dir
    )
    
    print("\n" + results['report'])
    
    # Run full analysis with all visualizations
    print("\n" + "=" * 60)
    print("RUNNING FULL CONSERVATION ANALYSIS WITH ALL VISUALIZATIONS")
    print("=" * 60)
    
    report_str, comparison = generate_report(
        file1=file1,
        file2=file2,
        threshold=0.8,
        min_coverage=0.5,
        output_dir=output_dir
    )
    
    # Map conserved positions to PDB structure
    if os.path.exists(pdb_file):
        print("\n" + "=" * 60)
        print("MAPPING TO PDB STRUCTURE")
        print("=" * 60)
        
        structure_mapping = map_conserved_to_structure(
            comparison=comparison,
            msa_file=file2,  # Use CrtM MSA for 3W7F_A (CrtM structure)
            pdb_file=pdb_file,
            output_dir=output_dir,
            name1="PSY",
            name2="CrtM",
            chain='A'
        )
    else:
        print(f"\nWarning: PDB file not found: {pdb_file}")
        print("Skipping structure mapping.")
        structure_mapping = None
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output files saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  - summary_comparison.png")
    print("  - overlap_diagram.png")
    print("  - conservation_distribution.png")
    print("  - shared_positions_comparison.png")
    print("  - amino_acid_distribution.png")
    print("  - conservation_landscape.png")
    print("  - tsne_clustering.png")
    print("  - subsample_statistics.png")
    print("  - robust_positions.png")
    
    if structure_mapping:
        print("\nStructure mapping files:")
        print("  - conserved_positions_pdb_mapping.csv")
        print("  - visualize_conserved.pml (PyMOL script)")
        print("  - visualize_conserved.cxc (ChimeraX script)")
    
    return results


if __name__ == "__main__":
    main()