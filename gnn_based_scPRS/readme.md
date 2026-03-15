# 📌 Project Overview

Type 2 Diabetes (T2D) is a complex metabolic disorder influenced by both **genetic susceptibility and cellular dysfunction** within pancreatic tissue.

Although **Genome-Wide Association Studies (GWAS)** have identified numerous risk variants, linking these variants to **specific pancreatic cell types** remains a major challenge.

This project presents an **integrative computational framework** combining:

* GWAS summary statistics
* Single-cell RNA sequencing (scRNA-seq)
* Polygenic Risk Score (PRS) analysis
* Protein–Protein Interaction networks
* Graph Neural Networks (GNN)

The goal is to **identify cell-type–specific genetic risk signals and biological pathways involved in Type 2 Diabetes.**

---

# 🧪 Data Sources

## 1️⃣ Single-Cell RNA Sequencing Dataset

* Source: **DISCO Database**
* GEO accession: **GSE221156**
* Format: `.h5ad`
* Cells analyzed: **131,696**
* Highly variable genes: **3,000**

### Pancreatic Cell Types

* Beta cells
* Alpha cells
* Delta cells
* Acinar cells
* Ductal cells
* Fibroblasts
* Endothelial cells
* Macrophages
* Perivascular cells
* Glial cells
* EndoMT cells

---

## 2️⃣ GWAS Summary Statistics

Source: **GWAS Catalog**

* Variants: **29,713,544 SNPs**
* Format: `.tsv`

### Quality Control Pipeline

| Step                    | Variants   |
| ----------------------- | ---------- |
| Raw GWAS dataset        | 29,713,544 |
| After QC filtering      | 12,169,541 |
| Chromosome 22 variants  | 156,003    |
| Gene annotated variants | 116,757    |

Filtering criteria:

* INFO ≥ 0.8
* Allele frequency between 0.01–0.99
* Valid p-values
* Valid nucleotide alleles (A, T, C, G)

---

# ⚙️ Methodology

## Step 1 — scRNA-seq Processing

Using **Scanpy**:

* Quality control filtering
* Normalization
* Log transformation
* PCA dimensionality reduction
* Clustering analysis
* Cell type annotation
* UMAP visualization

---

## Step 2 — Polygenic Risk Score Calculation

### Gene-Level PRS

PRS scores were calculated by aggregating **GWAS SNP effect sizes mapped to genes.**

### Cell-Level PRS

PRS scores were integrated with **single-cell gene expression** to compute **cell-type-specific genetic risk scores.**

---

## Step 3 — Feature Matrix Construction

The feature matrix contained:

* Gene-level PRS features
* Cell-type PRS scores
* Integrated genetic risk signals

Total features: **174**

---

## Step 4 — Simulated Patient Dataset

Due to the absence of matched genotype–phenotype data:

* **200 simulated patients** were generated
* Case/control outcomes assigned using binomial distribution

---

## Step 5 — Predictive Modeling

Models evaluated:

* Logistic Regression (L1 / LASSO)
* Logistic Regression (L2 / Ridge)
* Elastic Net

Evaluation:

* **5-fold cross-validation**
* **AUC metric**

---

## Step 6 — Protein–Protein Interaction Network

Network built using **STRING database**

Graph structure:

Nodes → Genes
Edges → Protein interactions

---

## Step 7 — Graph Neural Network

Model: **Graph Attention Network (GAT)**

Node features included:

* Gene expression
* Gene-level PRS

The GNN captured **gene interaction patterns and disease-associated network signals.**

---

# 🔬 Key Findings

## Cell Types with Highest Genetic Risk

| Cell Type         | PRS Score |
| ----------------- | --------- |
| Macrophages       | 128.62    |
| Fibroblasts       | 102.26    |
| Alpha Cells       | 69.62     |
| Endothelial Cells | 58.44     |

Lower risk:

* Beta cells
* Acinar cells

This suggests **immune and stromal cells play a major role in T2D genetic susceptibility.**

---

## Important Genes Identified

Top genes identified by PRS and GNN analysis:

* SOX10
* SHANK3
* NCF4
* OSM
* PARVB
* RAC2
* MIOX
* ADM2

These genes are involved in:

* immune signaling
* cytokine pathways
* vascular regulation
* extracellular matrix remodeling

---

## Functional Enrichment

Gene Ontology analysis identified pathways related to:

* Cytokine signaling
* Immune response
* STAT phosphorylation
* T-cell proliferation
* Adaptive immune regulation

---

# 🧰 Software and Libraries

Programming Language:

* Python

Major libraries:

* Scanpy
* Pandas
* NumPy
* PyTorch
* PyTorch Geometric
* AnnData
* NetworkX
* Scikit-learn

External tools:

* Cytoscape
* STRING database

---

# 📁 Repository Structure

```
gnn_based_scPRS
│
├── data
│   ├── scRNA_seq_dataset.h5ad
│   ├── gwas_summary_statistics.tsv
│
├── preprocessing
│   ├── gwas_qc_processing.py
│   ├── scrna_preprocessing.py
│
├── prs_analysis
│   ├── gene_level_prs.py
│   ├── cell_level_prs.py
│
├── gnn_model
│   ├── graph_builder.py
│   ├── gat_model.py
│   ├── train_gnn.py
│
├── visualization
│   ├── umap_plots.py
│   ├── network_visualization.py
│
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Rupanjalisingh/Single_Cell__Polygenic_risk_score.git
cd Single_Cell__Polygenic_risk_score
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```
scanpy
anndata
pandas
numpy
torch
torch_geometric
networkx
scikit-learn
matplotlib
seaborn
```

---

# ▶️ Running the Pipeline

### 1️⃣ Preprocess GWAS data

```bash
python preprocessing/gwas_qc_processing.py
```

### 2️⃣ Preprocess scRNA-seq dataset

```bash
python preprocessing/scrna_preprocessing.py
```

### 3️⃣ Compute Polygenic Risk Scores

```bash
python prs_analysis/gene_level_prs.py
python prs_analysis/cell_level_prs.py
```

### 4️⃣ Train Graph Neural Network

```bash
python gnn_model/train_gnn.py
```

---

# 📊 Visualization Outputs

The pipeline generates:

* UMAP cell clustering
* Differential gene expression plots
* PRS distribution heatmaps
* Cell-type risk ranking
* Gene interaction networks
* GNN embeddings
* Gene ontology enrichment plots

---

# 📂 Code Availability

GitHub repository:

[https://github.com/Rupanjalisingh/Single_Cell__Polygenic_risk_score/tree/main/gnn_based_scPRS](https://github.com/Rupanjalisingh/Single_Cell__Polygenic_risk_score/tree/main/gnn_based_scPRS)

```
Singh, R. (2026).
Integration of GWAS-Derived Polygenic Risk Scores with Single-Cell RNA Sequencing
to Identify Cell-Type–Specific Genetic Risk in Type 2 Diabetes.
```

---



# Limitation of the study

Chromosome-specific analysis – The study focused only on chromosome 22 variants, which may not capture the full genetic architecture of Type 2 Diabetes present across the entire genome.
Simulated patient data – Due to the absence of matched genotype–phenotype datasets, predictive modeling was performed on synthetic patient data, which limits the real-world predictive validity.
Limited variant–gene mapping accuracy – Mapping SNPs to genes may not fully represent the true regulatory relationships, especially for variants located in non-coding regions.
Incomplete protein interaction networks – The STRING PPI database contains known interactions, but may miss undiscovered or context-specific gene interactions.
Single dataset dependency – The analysis relied on one scRNA-seq dataset, which may not fully represent population diversity or disease heterogeneity.
