# Conserved Amino Acid Comparative Analysis Protocol

## 1. Final Alignment

### 1.1. Final Multiple Sequence Alignment

Take the highly curated, trimmed, and outlier-free sequences and subject them to a final alignment using Clustal Omega (ClustalO).

Save the output as the final processed FASTA alignment, ready for downstream phylogenetic or structural analyses.

## 2. EC Number Separation

Separate the final aligned sequences into two distinct datasets based on their respective Enzyme Commission (EC) numbers (CrtM and PSY). Save these as `finalcrtm.fasta` and `finalpsy.fasta`.

## 3. Conserved Amino Acid Comparative Analysis

Once the sequences are separated, execute the custom Conserved Amino Acid Analysis Tool. This tool performs a robust comparative overlay analysis between the two protein families (PSY and CrtM), handling class imbalances through bootstrap subsampling and mapping the conserved positions to a 3D structural model.

### 3.1. Analysis Parameters

*   **Conservation Threshold:** 80% (0.8) minimum fraction of sequences that must have the same amino acid.
*   **Minimum Coverage:** 50% (0.5) minimum fraction of total sequences that must have non-gap data.
*   **Subsampling Iterations:** 1000 iterations for robust statistical comparison.
*   **Reference Structure:** `3W7F_A.pdb` (S. aureus CrtM, Chain A).

### 3.2. Analysis Script Execution

Execute the custom `conserved_aa_analysis.py` script in your working directory via the command line or your preferred Python environment.

Ensure that your working directory contains the following required input files:
*   The separated MSA FASTA files (`finalcrtm.fasta` and `finalpsy.fasta`).
*   The reference PDB file (`3W7F_A.pdb`).

*(Note: The full Python source code for `conserved_aa_analysis.py` is maintained as a separate file to keep this protocol concise.)*

### 3.3. Expected Outputs and Visualizations

Upon successful execution, the script will generate a comprehensive suite of analytical data and publication-quality plots in the `conservation_analysis_output` directory:

**Statistical Reports & Data:**
*   `conservation_report.txt` & `subsampled_conservation_report.txt`: Detailed text summaries of the shared and unique conserved positions.
*   `iteration_statistics.csv`: Raw statistical data across all 1000 bootstrap iterations.
*   `*_robust_conserved.csv` and `shared_*_aa.csv`: Tabular data of the specific conserved residues.

**Generated Plot Results (.png):**
*   `summary_comparison.png`: Bar chart summarizing total, shared, and unique conserved positions.
*   `overlap_diagram.png`: Venn-style diagram showing the overlap of conserved positions between PSY and CrtM.
*   `conservation_distribution.png`: Histograms of conservation levels across both datasets.
*   `shared_positions_comparison.png`: Position-wise comparison of conservation scores at shared sites.
*   `amino_acid_distribution.png`: Bar charts showing the frequency of specific amino acids at conserved sites.
*   `conservation_landscape.png`: 1D heatmap showing the conservation landscape across the alignment sequence.
*   `conservation_landscape_2d.png`: A 2D NMR-style cross-peak plot highlighting positions conserved in both protein families, including marginal profiles.
*   `tsne_clustering.png`: t-SNE dimensionality reduction plot grouping sequences based on their amino acid patterns at conserved positions.
*   `subsample_statistics.png`: Box and line plots tracking conservation metrics across the bootstrap iterations.
*   `robust_positions.png`: Bar charts showing the frequency of appearance for specific conserved positions across all iterations.

**Structural Mapping Files:**
*   `conserved_positions_pdb_mapping.csv`: Direct mapping of MSA positions to PDB residue numbers.
*   `visualize_conserved.pml`: PyMOL script to automatically color the 3W7F_A structure based on conservation categories (Shared = Green, Divergent = Yellow/Gold, Unique = Blue/Red).
*   `visualize_conserved.cxc`: ChimeraX equivalent script for 3D structural visualization.
