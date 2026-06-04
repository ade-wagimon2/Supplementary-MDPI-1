# ⚛️ AMBER Multi-Replica Trajectory Convergence Analysis

This directory contains the scripts and Jupyter notebooks used to evaluate the convergence of Molecular Dynamics (MD) simulations for the study: **"Comparative Structural Modeling Suggests Distinct Signatures of Conformational Plasticity and Surface Physicochemistry in Phytoene Synthase and Dehydrosqualene Synthase."**

## 📖 Overview

To ensure that our MD simulations have adequately sampled the conformational space of the enzymes, we perform **Clustering Ensemble Similarity (CES)** analysis across multiple independent replicas. This method calculates the Jensen-Shannon Divergence (JSD) between conformational clusters of overlapping trajectory windows to quantitatively assess convergence.

*Originally adapted from MDAnalysis ENCORE examples.*

---

## 🧬 Systems Analyzed

The pipeline is configured to analyze four distinct protein-ligand systems, each consisting of **3 independent replicas** (`MD`, `MD2`, `MD3`):

| System Name | Protein | Ligand | Residues Selected (CA) |
| :--- | :--- | :--- | :--- |
| **PSY-FSPP** | Phytoene Synthase (PSY) | FSPP | `:1-300` |
| **PSY-GGPP** | Phytoene Synthase (PSY) | GGPP | `:1-300` |
| **CrtM-FSPP** | Dehydrosqualene Synthase (CrtM) | FSPP | `:1-284` |
| **CrtM-GGPP** | Dehydrosqualene Synthase (CrtM) | GGPP | `:1-284` |

---

## 📦 Dependencies & Environment

To run the `Convergence_analysis.ipynb` notebook, you will need a Python environment with the following packages installed:

* **Core:** `python >= 3.9`, `numpy`, `matplotlib`, `tqdm`, `scipy`
* **Trajectory Analysis:** `MDAnalysis` 
* **Convergence Toolkit:** `mdaencore` *(Note: The legacy `MDAnalysis.analysis.encore` module is deprecated. We highly recommend installing the standalone MDAKit `mdaencore` for future-proofing).*

**Quick install via conda/pip:**
```bash
conda install -c conda-forge mdanalysis numpy matplotlib scipy tqdm
pip install mdaencore