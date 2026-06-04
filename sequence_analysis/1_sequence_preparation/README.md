# 🧬 Sequence Analysis: CrtM and PSY Processing Pipeline

This directory contains the scripts, curated datasets, and standard operating procedures (SOP) used to retrieve, filter, align, and analyze the protein sequences for **CrtM (Dehydrosqualene Synthase)** and **PSY (Phytoene Synthase)**. 

The high-quality multiple sequence alignments (MSA) and phylogenetic trees generated here were used to identify distinct evolutionary signatures, domain conservations, and guide the downstream comparative structural modeling.

---

## 🔄 The Sequence Processing Pipeline (SOP)

The sequence curation follows a rigorous, multi-step pipeline combining automated scripting with manual quality control. 

### 1. Sequence Retrieval and Initial Cleaning
* **1.1. Data Download:** Protein sequences for CrtM and PSY were retrieved from the **KEGG database** based on their specific Enzyme Commission (EC) numbers. 
  * *Automation:* Execute the custom script `download_sequence.py` to automate the retrieval and generate the initial raw FASTA files.
* **1.2. Enzyme and Duplicate Cleaning:** Raw FASTA files are processed using the custom script `ec_cleaner.py`.
  * Filters out multi-functional enzymes (proteins associated with multiple EC numbers) to retain only single-EC enzymes.
  * Removes exact duplicate sequences to create a clean, non-redundant FASTA dataset.

### 2. Manual Filtering and Pre-Processing
* **2.1. Initial Alignment and Truncation:** An initial MSA is performed on the filtered dataset using **Clustal Omega (ClustalO)**. Sequences are manually inspected to trim and remove signal peptide regions from the N-terminus.
* **2.2. Annotation Quality Control:** Sequence metadata/headers are reviewed. Sequences with ambiguous, hypothetical, or uninformative annotations (e.g., "uncharacterized protein", "strange" names) are excluded.
* **2.3. Sequence Length Filtration:** Sequences are filtered based on amino acid (aa) length. 
  * *Criteria:* Retain only sequences between **250 aa (minimum)** and **500 aa (maximum)**.

### 3. Clustering and Dereplication
* **3.1. Sequence Clustering:** The aligned, signal-peptide-truncated sequences are clustered using **USEARCH / uclust**.
  * *Parameters:* Maximum sequence identity threshold set to **95%** to reduce redundancy while maintaining representative diversity.
* **3.2. Post-Clustering Curation:** A secondary round of manual cleaning is performed on the clustered output to ensure alignment integrity and remove obvious sequence artifacts.

### 4. Automated Trimming and Outlier Removal
* **Trimming:** Poorly aligned regions and large gaps are removed from the clustered alignment using **trimAl**.
* **Outlier Detection:** Statistically significant sequence outliers are identified and removed using the **`odseq`** package within an **R / Biostrings** environment.
* *Workflow Note:* This protocol is applied to clean all individual FASTA files first, followed by a final recleaning step on the combined (mixed) version.

### 5. Final Alignment
* **5.1. Final Multiple Sequence Alignment:** The highly curated, trimmed, and outlier-free sequences are subjected to a final alignment using **Clustal Omega (ClustalO)**.
* The output is saved as the final processed FASTA alignment, ready for downstream phylogenetic or structural analyses.

### 6. Phylogenetic Tree Construction
* **6.1. Tree Inference:** A maximum likelihood phylogenetic tree is constructed from the final aligned sequences using **IQ-TREE 3**.
* **Substitution Model:** `Q.PFAM+R8`
* **Command Line Execution:**
  ```bash
  iqtree3 -s final_alignment.fasta -m Q.PFAM+R8