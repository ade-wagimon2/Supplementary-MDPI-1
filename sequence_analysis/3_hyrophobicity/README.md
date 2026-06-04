# Hydrophobicity Analysis and Visualization Protocol

## 1. Final Alignment

### 1.1. Final Multiple Sequence Alignment

Take the highly curated, trimmed, and outlier-free sequences and subject them to a final alignment using Clustal Omega (ClustalO).

Save the output as the final processed FASTA alignment, ready for downstream phylogenetic or structural analyses.

## 2. EC Number Separation

Separate the final aligned sequences into two distinct datasets based on their respective Enzyme Commission (EC) numbers (CrtM and PSY). Save these as `finalcrtm.fasta` and `finalpsy.fasta`.

## 3. Hydrophobicity Analysis and Visualization

Once the sequences are separated, execute the custom Hydrophobicity Analysis Tool. This tool evaluates sequence hydrophobicity using multiple standard scales, detects potential transmembrane segments and amphipathic helices, performs random subsampling to handle class imbalances, and maps these physical properties to a 3D structural model.

### 3.1. Analysis Parameters

*   **Hydrophobicity Scales Evaluated:** Kyte-Doolittle (default/primary), Hopp-Woods, Eisenberg, Rose, Janin, Engelman (GES), and Cowan-Whittaker (HPLC pH 7.5).
*   **Sliding Window Sizes:**
    *   General profiles: 9 amino acids.
    *   Hydrophobic moments (Amphipathic helix detection): 11 amino acids (100° angle for α-helix).
    *   Transmembrane (TM) region detection: 19 amino acids.
*   **Subsampling Iterations:** 5 random sampling iterations of the larger PSY dataset to match the CrtM sequence count.
*   **Reference Structure:** `3W7F_A.pdb` (S. aureus CrtM, Chain A).

### 3.2. Analysis Script Execution

Execute the custom `hydrophobicity_analysis.py` script in your working directory via the command line or your preferred Python environment.

Ensure that your working directory contains the following required input files:
*   The separated MSA FASTA files (`finalcrtm.fasta` and `finalpsy.fasta`).
*   The reference PDB file (`3W7F_A.pdb`).

*(Note: The full Python source code for `hydrophobicity_analysis.py` is maintained as a separate file to keep this protocol concise.)*

### 3.3. Expected Outputs and Visualizations

Upon successful execution, the script will generate a comprehensive suite of analytical data and publication-quality plots in the `hydrophobicity_analysis_output` directory. Data is organized into subdirectories for `CrtM/`, `PSY_Full/`, and `PSY_Sampled/` (containing `Sample_1` through `Sample_5`).

**Generated Plot Results (.png for each target/sample):**
*   `*_kyte_doolittle_profile.png`: Standard sliding-window hydropathy profile (Kyte-Doolittle scale) with ±1 SD shading.
*   `*_multi_scale_comparison.png`: Side-by-side comparison of hydropathy profiles across all 7 evaluated scales.
*   `*_hydrophobicity_heatmap.png`: A QR code-style 2D map and linear barcode view of hydrophobicity (Cowan-Whittaker scale).
*   `*_distribution_analysis.png`: Histograms and box plots showing the frequency and distribution of hydrophobic vs. hydrophilic residues.
*   `*_hydrophobic_regions.png`: Identification and mapping of highly hydrophobic regions (potential TM segments).
*   `*_summary_statistics.png`: Bar charts showing cumulative hydrophobicity, sequence length distributions, and mean hydrophobicity by scale.
*   `*_hydrophobic_moments.png`: Hydrophobic moment profile to detect significant amphipathic alpha-helices.
*   `*_vs_CrtM_comparison.png`: Direct overlay and difference plots comparing a specific PSY sample against the CrtM baseline.
*   `PSY_samples_vs_CrtM_aggregate.png`: A final aggregate plot (in the main output folder) combining all PSY subsamples with calculated sampling uncertainty compared against CrtM.

**Structural Mapping Files (located in the CrtM/ directory):**
*   `hydrophobicity_pdb_mapping.csv`: Direct mapping of calculated MSA hydrophobicity values to PDB residue numbers.
*   `hydrophobic_regions_pdb.csv`: Summary of the isolated highly hydrophobic regions mapped to the structure.
*   `visualize_hydrophobicity.pml` / `.cxc`: PyMOL and ChimeraX scripts to color the `3W7F_A` structure continuously based on the hydrophobicity scale.
*   `visualize_hydrophobic_regions.pml` / `.cxc`: PyMOL and ChimeraX scripts to highlight and color-code the specific isolated hydrophobic/TM regions on the 3D structure.
