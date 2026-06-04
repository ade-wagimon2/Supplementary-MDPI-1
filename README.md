# Comparative Structural Modeling of Phytoene Synthase & Dehydrosqualene Synthase

[![Bioinformatics](https://img.shields.io/badge/Domain-Bioinformatics-blue.svg)](https://en.wikipedia.org/wiki/Bioinformatics)
[![Molecular Dynamics](https://img.shields.io/badge/Simulations-Amber%20MD-orange.svg)](https://ambermd.org/)
[![Data Analysis](https://img.shields.io/badge/Analysis-ProLIF%20%7C%20MDAnalysis-green.svg)](https://github.com/chemosim-lab/ProLIF)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

This repository serves as the official supplementary data and analysis archive for the study: **"Comparative Structural Modeling Suggests Distinct Signatures of Conformational Plasticity and Surface Physicochemistry in Phytoene Synthase and Dehydrosqualene Synthase."**

Phytoene Synthase (**PSY**) and Dehydrosqualene Synthase (**CrtM**) are crucial enzymes in the isoprenoid and carotenoid biosynthesis pathways. This study investigates their structural dynamics, highlighting the distinct evolutionary signatures of conformational plasticity and surface physicochemistry that differentiate their catalytic mechanisms and substrate specificities.

---

## 📂 Repository Structure

The codebase is organized into four main functional directories corresponding to key stages of our comparative structural biology pipeline:

```
repository/
├── 🧬 sequence_analysis/           # Bioinformatics, alignments, and conservation analysis
├── 🧱 input_structure/             # Prepared 3D structural models and simulation topologies
├── ⚛️ MD_analysis/                  # MD trajectory metrics, convergence, and interaction fingerprints
└── 📊 figures/                      # High-resolution publication figures and supplemental data
```

### 🧬 `sequence_analysis/`

Contains the scripts, curated datasets, and standard operating procedures (SOP) used to retrieve, filter, align, and analyze protein sequences for CrtM and PSY.

* **`1_sequence_preparation/`**: Implements the sequence processing pipeline (SOP) including retrieval from the KEGG database, EC number purification, duplicate removal, N-terminal signal peptide trimming, length filtration (250–500 aa), sequence clustering (at 95% identity using USEARCH), outlier detection (using R `odseq`), and Maximum Likelihood tree construction using IQ-TREE (`Q.PFAM+R8` substitution model).
* **`2_Conservation_Analysis/`**: Contains python scripts (`conserved_analysis.py`, `structure_mapper.py`) and outputs measuring sequence-level conservation, shared amino acids, and mapping them directly to ChimeraX (`.cxc`) and PyMOL (`.pml`) visualization scripts.
* **`3_hyrophobicity/`**: Conducts multi-scale hydropathy profiling using 7 standard scales (Kyte-Doolittle, Hopp-Woods, Eisenberg, Rose, Janin, Engelman (GES), and Cowan-Whittaker) and random subsampling, generating 3D mappings.
* **`4_helix-13_hydrophobicity/`**: Focuses specifically on the hydrophobicity calculations of helical regions (Helix 13) including alignment files (`.fasta`), trees (`.tre`), annotation files, and calculations in Jupyter (`hidrophob_calculation.ipynb`).

### 🧱 `input_structure/`

Contains the 3D structural coordinates and topological configurations prior to MD simulations.

* **Reference PDBs**: Crystal structures and OpenFold3 models, including `2ZCO.pdb`, `3AE0_A.pdb`, `3W7F_A.pdb`, and `5IYS.pdb`.
* **System Coordinates**: Starting structures for simulations, such as `CRTB_COMPLEX_FSPP_seed_42` and `CRTB_COMPLEX_GGPP_seed_42`.

### ⚛️ `MD_analysis/`

Contains simulation outputs, convergence verification, and custom scripts to parse trajectories.

* **`1_MD_analysis/`**: Contains time-series datasets of Root Mean Square Deviation (RMSD), Root Mean Square Fluctuation (RMSF), and Radius of Gyration (Rg) across the independent MD replicas (MD1, MD2, MD3) for each system. Includes scripts for metric aggregation (`calculate_basic_metrics.py`, `visualize_metrics.py`, `visualize_rmsf_zscores.py`).
* **`2_Convergence/`**: Evaluates trajectory sampling convergence using **Clustering Ensemble Similarity (CES)** analysis and Jensen-Shannon Divergence (JSD) implemented via `MDAnalysis` and `mdaencore` (`Convergence_analysis.ipynb`).
* **`3_Interactions/`**: Quantifies and compares protein-ligand interaction fingerprints (PLIF) across systems (e.g., Hydrophobic, Hydrogen Bonds, Anionic, VdW contacts) using **ProLIF** and generates Nature-style comparative figures (`Full_interactions.ipynb` and `interaction_analysis_restyle.py`).

### 📊 `figures/`

Includes all publication-ready graphical outputs, supplementary diagrams, tables, and visualization sessions.

* **Main Figures**: High-resolution representations of conservation landscapes, RMSF/RMSD profiles, PCA landscape distributions, ProLIF fingerprints, and surface hydrophobicity mappings.
* **Supplemental Figures**: Alignment metrics, B-factor representations, model confidence plots, and convergence summary graphs.

---

## 💻 Software & Dependencies

The computational workflows in this repository depend on standard bioinformatics and molecular modeling tools.

### Python Dependencies

We recommend setting up a dedicated virtual environment (Conda or Pip) using Python 3.9+:

```bash
# Install core MD and trajectory manipulation libraries
conda install -c conda-forge mdanalysis prolif rdkit

# Install scientific stack and visualization libraries
pip install numpy pandas matplotlib seaborn scipy tqdm jupyter mdaencore
```

### External Tools

Depending on which stage of the pipeline you run, you may need:

* **MD Trajectory Processing**: [AmberTools (CPPTRAJ)](https://ambermd.org/)
* **Sequence Processing & Alignment**: [Clustal Omega](https://www.ebi.ac.uk/Tools/msa/clustalo/), [trimAl](http://trimal.cgenomics.org/), [uclust](https://github.com/gcybis/Uclust)
* **Phylogenetics**: [IQ-TREE](http://www.iqtree.org/)
* **Molecular Visualization**: [PyMOL](https://pymol.org/), [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/)

---
