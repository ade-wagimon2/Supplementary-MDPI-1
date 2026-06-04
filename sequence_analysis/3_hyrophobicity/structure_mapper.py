"""
Structure Mapper Module

Maps MSA (Multiple Sequence Alignment) positions to PDB structure residue numbers
through sequence alignment, enabling visualization of conserved and hydrophobic
residues on protein structures.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


# =============================================================================
# PDB SEQUENCE EXTRACTION
# =============================================================================

# Standard amino acid 3-letter to 1-letter conversion
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Non-standard amino acids
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K', 'SEP': 'S', 'TPO': 'T',
    'PTR': 'Y', 'HYP': 'P', 'CSO': 'C', 'CSS': 'C'
}


def extract_pdb_sequence(pdb_file: str, chain: str = 'A') -> Tuple[str, List[int]]:
    """
    Extract amino acid sequence and residue numbers from a PDB file.
    
    Args:
        pdb_file: Path to PDB file
        chain: Chain identifier (default: 'A')
    
    Returns:
        Tuple of (sequence string, list of residue numbers)
    """
    sequence = []
    residue_numbers = []
    seen_residues = set()
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB ATOM record
                atom_chain = line[21].strip()
                if atom_chain != chain:
                    continue
                
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                
                # Skip if we've already seen this residue
                res_key = (res_num, res_name)
                if res_key in seen_residues:
                    continue
                seen_residues.add(res_key)
                
                # Convert to single letter
                if res_name in AA_3TO1:
                    sequence.append(AA_3TO1[res_name])
                    residue_numbers.append(res_num)
    
    return ''.join(sequence), residue_numbers


def get_pdb_info(pdb_file: str, chain: str = 'A') -> Dict:
    """
    Get comprehensive information from a PDB file.
    
    Args:
        pdb_file: Path to PDB file
        chain: Chain identifier
    
    Returns:
        Dictionary with sequence, residue numbers, and metadata
    """
    sequence, residue_numbers = extract_pdb_sequence(pdb_file, chain)
    
    return {
        'sequence': sequence,
        'residue_numbers': residue_numbers,
        'length': len(sequence),
        'chain': chain,
        'file': pdb_file,
        'first_residue': residue_numbers[0] if residue_numbers else None,
        'last_residue': residue_numbers[-1] if residue_numbers else None
    }


# =============================================================================
# MSA SEQUENCE HANDLING
# =============================================================================

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


def get_representative_sequence(sequences: Dict[str, str]) -> Tuple[str, str]:
    """
    Get the first (representative) sequence from an MSA.
    
    Args:
        sequences: Dictionary of sequence name -> aligned sequence
    
    Returns:
        Tuple of (header, ungapped sequence)
    """
    if not sequences:
        return None, None
    
    first_header = next(iter(sequences.keys()))
    first_seq = sequences[first_header]
    
    # Remove gaps
    ungapped = first_seq.replace('-', '').replace('.', '')
    
    return first_header, ungapped


def get_consensus_sequence(sequences: Dict[str, str], threshold: float = 0.5) -> str:
    """
    Build a consensus sequence from an MSA.
    
    Args:
        sequences: Dictionary of sequence name -> aligned sequence
        threshold: Minimum fraction for consensus (default: 0.5)
    
    Returns:
        Consensus sequence (gaps removed)
    """
    if not sequences:
        return ""
    
    alignment_length = len(next(iter(sequences.values())))
    consensus = []
    
    for pos in range(alignment_length):
        column = [seq[pos].upper() for seq in sequences.values() if pos < len(seq)]
        
        # Count non-gap amino acids
        aa_counts = Counter([aa for aa in column if aa not in '-.' and aa != 'X'])
        
        if not aa_counts:
            continue  # Skip gap-only columns
        
        most_common_aa, count = aa_counts.most_common(1)[0]
        total_non_gap = sum(aa_counts.values())
        
        if count / total_non_gap >= threshold:
            consensus.append(most_common_aa)
        else:
            consensus.append('X')  # Ambiguous
    
    # Remove X's and return
    return ''.join([aa for aa in consensus if aa != 'X'])


def create_msa_position_map(aligned_seq: str) -> Dict[int, int]:
    """
    Create a mapping from MSA (1-indexed) positions to ungapped sequence positions.
    
    Args:
        aligned_seq: Aligned sequence with gaps
    
    Returns:
        Dictionary mapping MSA position (1-indexed) to ungapped position (1-indexed)
    """
    position_map = {}
    ungapped_pos = 0
    
    for msa_pos, aa in enumerate(aligned_seq, start=1):
        if aa not in '-.':
            ungapped_pos += 1
            position_map[msa_pos] = ungapped_pos
    
    return position_map


# =============================================================================
# SEQUENCE ALIGNMENT
# =============================================================================

def simple_align(seq1: str, seq2: str) -> Tuple[str, str, List[Tuple[int, int]]]:
    """
    Simple pairwise sequence alignment using dynamic programming (Needleman-Wunsch).
    
    Args:
        seq1: First sequence
        seq2: Second sequence
    
    Returns:
        Tuple of (aligned seq1, aligned seq2, list of position mappings)
    """
    # Scoring parameters
    MATCH = 2
    MISMATCH = -1
    GAP = -2
    
    m, n = len(seq1), len(seq2)
    
    # Initialize scoring matrix
    score = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize gap penalties
    for i in range(m + 1):
        score[i][0] = i * GAP
    for j in range(n + 1):
        score[0][j] = j * GAP
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i-1][j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            delete = score[i-1][j] + GAP
            insert = score[i][j-1] + GAP
            score[i][j] = max(match, delete, insert)
    
    # Traceback
    aligned1, aligned2 = [], []
    position_mapping = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            current = score[i][j]
            diagonal = score[i-1][j-1]
            match_score = MATCH if seq1[i-1] == seq2[j-1] else MISMATCH
            
            if current == diagonal + match_score:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                position_mapping.append((i, j))  # 1-indexed
                i -= 1
                j -= 1
            elif i > 0 and current == score[i-1][j] + GAP:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
        elif i > 0:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1
    
    # Reverse (traceback goes backwards)
    aligned1.reverse()
    aligned2.reverse()
    position_mapping.reverse()
    
    return ''.join(aligned1), ''.join(aligned2), position_mapping


def map_msa_to_pdb(
    msa_positions: List[int],
    msa_sequences: Dict[str, str],
    pdb_sequence: str,
    pdb_residue_numbers: List[int]
) -> List[Dict]:
    """
    Map MSA positions to PDB residue numbers through sequence alignment.
    
    Args:
        msa_positions: List of MSA positions (1-indexed) to map
        msa_sequences: Dictionary of MSA sequences
        pdb_sequence: Sequence extracted from PDB
        pdb_residue_numbers: List of residue numbers from PDB
    
    Returns:
        List of dictionaries with mapping information
    """
    # Get representative sequence from MSA
    header, rep_seq = get_representative_sequence(msa_sequences)
    
    if not rep_seq:
        print("Error: Could not get representative sequence from MSA")
        return []
    
    # Get the original aligned sequence for position mapping
    first_aligned_seq = next(iter(msa_sequences.values()))
    msa_position_map = create_msa_position_map(first_aligned_seq)
    
    # Align representative to PDB sequence
    aligned_msa, aligned_pdb, position_pairs = simple_align(rep_seq, pdb_sequence)
    
    # Create mapping from ungapped MSA positions to PDB residue numbers
    ungapped_to_pdb = {}
    for msa_pos, pdb_pos in position_pairs:
        if pdb_pos <= len(pdb_residue_numbers):
            ungapped_to_pdb[msa_pos] = pdb_residue_numbers[pdb_pos - 1]
    
    # Map MSA positions to PDB residue numbers
    results = []
    for msa_pos in msa_positions:
        result = {
            'msa_position': msa_pos,
            'ungapped_position': None,
            'pdb_residue_number': None,
            'mapped': False
        }
        
        # Convert MSA position to ungapped position
        if msa_pos in msa_position_map:
            ungapped_pos = msa_position_map[msa_pos]
            result['ungapped_position'] = ungapped_pos
            
            # Map to PDB residue number
            if ungapped_pos in ungapped_to_pdb:
                result['pdb_residue_number'] = ungapped_to_pdb[ungapped_pos]
                result['mapped'] = True
        
        results.append(result)
    
    return results


def get_alignment_stats(aligned1: str, aligned2: str) -> Dict:
    """
    Calculate alignment statistics.
    
    Args:
        aligned1: First aligned sequence
        aligned2: Second aligned sequence
    
    Returns:
        Dictionary with alignment statistics
    """
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
    mismatches = sum(1 for a, b in zip(aligned1, aligned2) if a != b and a != '-' and b != '-')
    gaps = sum(1 for a, b in zip(aligned1, aligned2) if a == '-' or b == '-')
    
    aligned_length = len(aligned1)
    
    return {
        'matches': matches,
        'mismatches': mismatches,
        'gaps': gaps,
        'identity': matches / (matches + mismatches) * 100 if (matches + mismatches) > 0 else 0,
        'aligned_length': aligned_length,
        'coverage': (aligned_length - gaps) / aligned_length * 100 if aligned_length > 0 else 0
    }


# =============================================================================
# VISUALIZATION SCRIPT GENERATION
# =============================================================================

def generate_pymol_script(
    pdb_file: str,
    residue_groups: Dict[str, List[int]],
    output_path: str,
    color_scheme: Dict[str, str] = None
) -> str:
    """
    Generate a PyMOL script to visualize residue groups on a structure.
    
    Args:
        pdb_file: Path to PDB file
        residue_groups: Dictionary of group name -> list of residue numbers
        output_path: Path to save the PyMOL script
        color_scheme: Dictionary of group name -> color (PyMOL color name or hex)
    
    Returns:
        Path to generated script
    """
    if color_scheme is None:
        color_scheme = {
            'conserved_same': 'green',
            'conserved_different': 'yellow',
            'unique_psy': 'marine',
            'unique_crtm': 'salmon',
            'hydrophobic_high': 'red',
            'hydrophobic_medium': 'orange',
            'hydrophobic_low': 'yellow'
        }
    
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    script_lines = [
        "# PyMOL Visualization Script",
        f"# Generated for: {pdb_basename}",
        "# Auto-generated - do not edit manually",
        "",
        "# Reinitialize PyMOL",
        "reinitialize",
        "",
        f"# Load structure",
        f"load {rel_pdb_path}, {pdb_name}",
        "",
        "# Set visualization style",
        f"hide all",
        f"show cartoon, {pdb_name}",
        f"color gray80, {pdb_name}",
        "",
        "# Create selections and color residue groups",
    ]
    
    for group_name, residues in residue_groups.items():
        if not residues:
            continue
        
        # Create residue selection string
        res_str = '+'.join(str(r) for r in sorted(residues))
        selection_name = group_name.replace(' ', '_').lower()
        color = color_scheme.get(group_name, 'white')
        
        script_lines.extend([
            f"",
            f"# {group_name} ({len(residues)} residues)",
            f"select {selection_name}, {pdb_name} and resi {res_str}",
            f"color {color}, {selection_name}",
            f"show sticks, {selection_name}",
        ])
    
    # Add final visualization settings
    script_lines.extend([
        "",
        "# Final view settings",
        "deselect",
        "orient",
        "zoom vis",
        "bg_color white",
        "set ray_trace_mode, 1",
        "",
        "# Save session (optional)",
        f"# save {os.path.splitext(os.path.basename(output_path))[0]}.pse",
    ])
    
    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"  PyMOL script saved: {output_path}")
    return output_path


def generate_hydrophobicity_pymol_script(
    pdb_file: str,
    residue_hydrophobicity: Dict[int, float],
    output_path: str,
    scale_name: str = "Kyte-Doolittle"
) -> str:
    """
    Generate a PyMOL script to color residues by hydrophobicity.
    
    Args:
        pdb_file: Path to PDB file
        residue_hydrophobicity: Dictionary of residue number -> hydrophobicity value
        output_path: Path to save the PyMOL script
        scale_name: Name of the hydrophobicity scale used
    
    Returns:
        Path to generated script
    """
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    # Determine min/max for normalization
    if residue_hydrophobicity:
        values = list(residue_hydrophobicity.values())
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
    else:
        min_val, max_val, range_val = 0, 1, 1
    
    script_lines = [
        "# PyMOL Hydrophobicity Visualization Script",
        f"# Generated for: {pdb_basename}",
        f"# Hydrophobicity scale: {scale_name}",
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
        "# Color by hydrophobicity (blue = hydrophilic, red = hydrophobic)",
    ]
    
    # Group residues by hydrophobicity level
    hydrophilic = []  # < -1
    neutral = []      # -1 to 1
    hydrophobic = []  # > 1
    
    for res_num, hydro_val in residue_hydrophobicity.items():
        if hydro_val < -1:
            hydrophilic.append(res_num)
        elif hydro_val > 1:
            hydrophobic.append(res_num)
        else:
            neutral.append(res_num)
    
    if hydrophilic:
        res_str = '+'.join(str(r) for r in sorted(hydrophilic))
        script_lines.extend([
            f"",
            f"# Hydrophilic residues ({len(hydrophilic)} residues)",
            f"select hydrophilic, {pdb_name} and resi {res_str}",
            f"color marine, hydrophilic",
        ])
    
    if neutral:
        res_str = '+'.join(str(r) for r in sorted(neutral))
        script_lines.extend([
            f"",
            f"# Neutral residues ({len(neutral)} residues)",
            f"select neutral, {pdb_name} and resi {res_str}",
            f"color white, neutral",
        ])
    
    if hydrophobic:
        res_str = '+'.join(str(r) for r in sorted(hydrophobic))
        script_lines.extend([
            f"",
            f"# Hydrophobic residues ({len(hydrophobic)} residues)",
            f"select hydrophobic, {pdb_name} and resi {res_str}",
            f"color firebrick, hydrophobic",
            f"show sticks, hydrophobic",
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
    
    print(f"  PyMOL hydrophobicity script saved: {output_path}")
    return output_path


def generate_chimerax_script(
    pdb_file: str,
    residue_groups: Dict[str, List[int]],
    output_path: str,
    color_scheme: Dict[str, str] = None
) -> str:
    """
    Generate a ChimeraX script to visualize residue groups on a structure.
    
    Args:
        pdb_file: Path to PDB file
        residue_groups: Dictionary of group name -> list of residue numbers
        output_path: Path to save the ChimeraX script
        color_scheme: Dictionary of group name -> color (ChimeraX color name or hex)
    
    Returns:
        Path to generated script
    """
    # ChimeraX color mapping (convert PyMOL colors to ChimeraX equivalents)
    if color_scheme is None:
        color_scheme = {
            'conserved_same': 'green',
            'conserved_different': 'gold',
            'unique_psy': 'dodger blue',
            'unique_crtm': 'salmon',
            'hydrophobic_high': 'red',
            'hydrophobic_medium': 'orange',
            'hydrophobic_low': 'yellow'
        }
    
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    script_lines = [
        "# ChimeraX Visualization Script",
        f"# Generated for: {pdb_basename}",
        "# Auto-generated - do not edit manually",
        "",
        "# Close all models",
        "close all",
        "",
        f"# Load structure",
        f"open {rel_pdb_path}",
        "",
        "# Set visualization style",
        "hide atoms",
        "show cartoons",
        "color gray target c",
        "",
        "# Create selections and color residue groups",
    ]
    
    for group_name, residues in residue_groups.items():
        if not residues:
            continue
        
        # Create residue selection string for ChimeraX
        res_str = ','.join(str(r) for r in sorted(residues))
        selection_name = group_name.replace(' ', '_').lower()
        color = color_scheme.get(group_name, 'white')
        
        script_lines.extend([
            f"",
            f"# {group_name} ({len(residues)} residues)",
            f"name {selection_name} :{res_str}",
            f"color {selection_name} {color} target c",
            f"show {selection_name} atoms",
            f"style {selection_name} stick",
        ])
    
    # Add final visualization settings
    script_lines.extend([
        "",
        "# Final view settings",
        "view",
        "set bgColor white",
        "lighting soft",
        "",
        "# Save session (optional)",
        f"# save {os.path.splitext(os.path.basename(output_path))[0]}.cxs",
    ])
    
    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"  ChimeraX script saved: {output_path}")
    return output_path


def generate_hydrophobicity_chimerax_script(
    pdb_file: str,
    residue_hydrophobicity: Dict[int, float],
    output_path: str,
    scale_name: str = "Kyte-Doolittle"
) -> str:
    """
    Generate a ChimeraX script to color residues by hydrophobicity.
    
    Args:
        pdb_file: Path to PDB file
        residue_hydrophobicity: Dictionary of residue number -> hydrophobicity value
        output_path: Path to save the ChimeraX script
        scale_name: Name of the hydrophobicity scale used
    
    Returns:
        Path to generated script
    """
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    script_lines = [
        "# ChimeraX Hydrophobicity Visualization Script",
        f"# Generated for: {pdb_basename}",
        f"# Hydrophobicity scale: {scale_name}",
        "",
        "close all",
        "",
        f"open {rel_pdb_path}",
        "",
        "# Base visualization",
        "hide atoms",
        "show cartoons",
        "color gray target c",
        "",
        "# Color by hydrophobicity (blue = hydrophilic, red = hydrophobic)",
    ]
    
    # Group residues by hydrophobicity level
    hydrophilic = []  # < -1
    neutral = []      # -1 to 1
    hydrophobic = []  # > 1
    
    for res_num, hydro_val in residue_hydrophobicity.items():
        if hydro_val < -1:
            hydrophilic.append(res_num)
        elif hydro_val > 1:
            hydrophobic.append(res_num)
        else:
            neutral.append(res_num)
    
    if hydrophilic:
        res_str = ','.join(str(r) for r in sorted(hydrophilic))
        script_lines.extend([
            f"",
            f"# Hydrophilic residues ({len(hydrophilic)} residues)",
            f"name hydrophilic :{res_str}",
            f"color hydrophilic dodger blue target c",
        ])
    
    if neutral:
        res_str = ','.join(str(r) for r in sorted(neutral))
        script_lines.extend([
            f"",
            f"# Neutral residues ({len(neutral)} residues)",
            f"name neutral :{res_str}",
            f"color neutral white target c",
        ])
    
    if hydrophobic:
        res_str = ','.join(str(r) for r in sorted(hydrophobic))
        script_lines.extend([
            f"",
            f"# Hydrophobic residues ({len(hydrophobic)} residues)",
            f"name hydrophobic :{res_str}",
            f"color hydrophobic firebrick target c",
            f"show hydrophobic atoms",
            f"style hydrophobic stick",
        ])
    
    script_lines.extend([
        "",
        "# Final settings",
        "view",
        "set bgColor white",
        "lighting soft",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"  ChimeraX hydrophobicity script saved: {output_path}")
    return output_path


def generate_chimerax_regions_script(
    pdb_file: str,
    regions: List[Dict],
    output_path: str
) -> str:
    """
    Generate a ChimeraX script to highlight hydrophobic regions.
    
    Args:
        pdb_file: Path to PDB file
        regions: List of region dictionaries with 'residues' key
        output_path: Path to save the ChimeraX script
    
    Returns:
        Path to generated script
    """
    pdb_basename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_basename)[0]
    rel_pdb_path = os.path.relpath(pdb_file, os.path.dirname(output_path)).replace('\\', '/')
    
    script_lines = [
        "# ChimeraX Hydrophobic Regions Visualization Script",
        f"# Generated for: {pdb_basename}",
        "",
        "close all",
        "",
        f"open {rel_pdb_path}",
        "",
        "# Base visualization",
        "hide atoms",
        "show cartoons",
        "color gray target c",
        "",
        f"# Highlight {len(regions)} hydrophobic regions",
    ]
    
    # Color palette for regions
    colors = ['red', 'orange', 'gold', 'forest green', 'dodger blue', 'purple', 'hot pink']
    
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        res_str = ','.join(str(r) for r in region['residues'])
        selection_name = f"hydro_region_{i+1}"
        
        script_lines.extend([
            f"",
            f"# Region {i+1}: residues {region['start']}-{region['end']} (mean hydro: {region['mean_hydrophobicity']:.2f})",
            f"name {selection_name} :{res_str}",
            f"color {selection_name} {color} target c",
            f"surface {selection_name}",
            f"transparency {selection_name} 50 target s",
        ])
    
    script_lines.extend([
        "",
        "# Final settings",
        "view",
        "set bgColor white",
        "lighting soft",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"  ChimeraX regions script saved: {output_path}")
    return output_path


def save_mapping_csv(
    mappings: List[Dict],
    output_path: str,
    additional_columns: Dict[int, Dict] = None
) -> str:
    """
    Save position mappings to a CSV file.
    
    Args:
        mappings: List of mapping dictionaries from map_msa_to_pdb
        output_path: Path to save CSV
        additional_columns: Optional dict of msa_position -> additional data
    
    Returns:
        Path to saved file
    """
    import csv
    
    # Determine all columns
    columns = ['msa_position', 'ungapped_position', 'pdb_residue_number', 'mapped']
    if additional_columns:
        sample = next(iter(additional_columns.values()), {})
        columns.extend(sample.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for mapping in mappings:
            row = mapping.copy()
            if additional_columns and mapping['msa_position'] in additional_columns:
                row.update(additional_columns[mapping['msa_position']])
            writer.writerow(row)
    
    print(f"  Mapping CSV saved: {output_path}")
    return output_path


# =============================================================================
# MAIN MAPPING FUNCTION
# =============================================================================

def map_analysis_to_structure(
    msa_file: str,
    pdb_file: str,
    positions: List[int],
    output_dir: str,
    analysis_name: str = "analysis",
    chain: str = 'A'
) -> Dict:
    """
    Main function to map MSA analysis positions to PDB structure.
    
    Args:
        msa_file: Path to MSA FASTA file
        pdb_file: Path to PDB file
        positions: List of MSA positions to map
        output_dir: Directory for output files
        analysis_name: Name for output files
        chain: PDB chain identifier
    
    Returns:
        Dictionary with mapping results and file paths
    """
    print(f"\n{'='*60}")
    print(f"Mapping {analysis_name} to Structure")
    print(f"{'='*60}")
    
    # Parse inputs
    print(f"\nParsing MSA: {msa_file}")
    msa_sequences = parse_fasta(msa_file)
    print(f"  Loaded {len(msa_sequences)} sequences")
    
    print(f"\nParsing PDB: {pdb_file}")
    pdb_info = get_pdb_info(pdb_file, chain)
    print(f"  Chain {chain}: {pdb_info['length']} residues")
    print(f"  Residue range: {pdb_info['first_residue']}-{pdb_info['last_residue']}")
    
    # Get representative sequence
    header, rep_seq = get_representative_sequence(msa_sequences)
    print(f"\nRepresentative sequence: {header[:50]}...")
    print(f"  Length (ungapped): {len(rep_seq)}")
    
    # Align sequences
    print(f"\nAligning sequences...")
    aligned_msa, aligned_pdb, position_pairs = simple_align(rep_seq, pdb_info['sequence'])
    stats = get_alignment_stats(aligned_msa, aligned_pdb)
    print(f"  Identity: {stats['identity']:.1f}%")
    print(f"  Coverage: {stats['coverage']:.1f}%")
    print(f"  Matches: {stats['matches']}, Mismatches: {stats['mismatches']}, Gaps: {stats['gaps']}")
    
    # Map positions
    print(f"\nMapping {len(positions)} positions...")
    mappings = map_msa_to_pdb(
        positions,
        msa_sequences,
        pdb_info['sequence'],
        pdb_info['residue_numbers']
    )
    
    mapped_count = sum(1 for m in mappings if m['mapped'])
    print(f"  Successfully mapped: {mapped_count}/{len(positions)}")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'{analysis_name}_pdb_mapping.csv')
    save_mapping_csv(mappings, csv_path)
    
    return {
        'mappings': mappings,
        'pdb_info': pdb_info,
        'alignment_stats': stats,
        'representative_sequence': rep_seq,
        'csv_path': csv_path,
        'mapped_residues': [m['pdb_residue_number'] for m in mappings if m['mapped']]
    }


if __name__ == "__main__":
    # Example usage
    print("Structure Mapper Module")
    print("=" * 60)
    print("This module provides functions to map MSA positions to PDB residue numbers.")
    print("\nExample usage:")
    print("  from structure_mapper import map_analysis_to_structure")
    print("  result = map_analysis_to_structure('msa.fasta', 'structure.pdb', [10, 20, 30], 'output/')")
