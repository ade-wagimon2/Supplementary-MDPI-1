# Comparative Structural Modeling Suggests Distinct Signatures of Conformational Plasticity and Surface Physicochemistry in Phytoene Synthase and Dehydrosqualene Synthase

## 📖 Overview

This repository serves as the supplementary data archive for the study: **"Comparative Structural Modeling Suggests Distinct Signatures of Conformational Plasticity and Surface Physicochemistry in Phytoene Synthase and Dehydrosqualene Synthase."**

Phytoene Synthase (PSY) and Dehydrosqualene Synthase (CrtM) are crucial enzymes in the isoprenoid/carotenoid biosynthesis pathways. This study investigates their structural dynamics, highlighting the distinct evolutionary signatures of conformational plasticity and surface physicochemistry that differentiate their catalytic mechanisms and substrate specificities.

The repository contains all raw and processed data, 3D structural models, molecular dynamics (MD) analysis scripts, and high-resolution figures necessary to reproduce the findings of the manuscript.

---

## 📁 Repository Structure

The repository is organized into the following directories:

### 🧬 `sequence_analysis/`

Contains bioinformatics data and sequence-level comparisons of PSY and CrtM.

* Multiple Sequence Alignments (MSA).
* Phylogenetic tree files (`.tree`).
* Domain architecture and motif conservation mappings.

### 🧱 `input_structure/`

Contains the 3D structural models and topological setups prepared prior to molecular dynamics simulations.

* Initial structural models (Homology models / AlphaFold2 / PDB files).
* System preparation files.
* Simulation setup parameters.

### ⚛️ `MD_analysis/`

Contains processed results, metrics, and custom scripts extracted from the Molecular Dynamics (MD) simulations.

* **Conformational Plasticity:** Time-series data for RMSD (Root Mean Square Deviation), RMSF (Root Mean Square Fluctuation), and Radius of Gyration (Rg).
* **Essential Dynamics:** Principal Component Analysis (PCA).
* **Scripts:** Python scripts used to parse trajectories and generate plots.

### 📊 `figures/`

Contains all graphical outputs generated for the manuscript and supplementary information.

* **`main/`**: High-resolution, publication-ready figures for the main text.
* **`supplementary/`**: Extended data and supplementary figures.
* **`sessions/`**: Molecular visualization session files (e.g., PyMOL `.pse`, VMD `.vmd`, UCSF Chimera) used to render structural figures.

---

## 💻 Software & Dependencies

The analyses and modeling pipelines in this repository rely on several standard computational biology tools. Depending on the scripts you intend to run, you may need:

* **MD Simulation & Analysis:** [AMBER, CPPTRAJ, MDAnalysis, and proliff]
* **Structural Modeling & Visualization:** [e.g., PyMOL, UCSF Chimera, VMD]
* **Sequence Analysis:** [e.g., Clustal Omega, MUSCLE, MEGA, Biopython]
* **Data Processing & Plotting:** Python 3.9+ (with `pandas`, `matplotlib`, `seaborn`, `scipy`).

---

## 📖 Citation

> **[Author 1], [Author 2], et al.** (2026). *Comparative Structural Modeling Suggests Distinct Signatures of Conformational Plasticity and Surface Physicochemistry in Phytoene Synthase and Dehydrosqualene Synthase*. **[Journal Name]**, **[Volume]([Issue])**, [Page range].
> DOI: [Insert DOI here]

---
