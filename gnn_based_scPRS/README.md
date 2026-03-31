# Single Cell Polygenic Risk Score (scPRS)

A comprehensive pipeline for computing polygenic risk scores (PRS) using single-cell transcriptomic data integrated with large-scale GWAS summary statistics. This project leverages Graph Neural Networks (GNNs) to identify disease-associated genes and prioritize therapeutic targets in disease-relevant cell types.

## Overview

This repository implements a multi-modal approach to disease genetics:

- **Data Integration**: Combines single-cell RNA-seq data with GWAS summary statistics
- **Gene Network Analysis**: Builds interaction networks using STRING and custom annotation
- **PRS Computation**: Calculates genome-wide and gene-level polygenic risk scores
- **Cell-type Enrichment**: Identifies disease-associated cell types using CELLECT
- **GNN-based Target Prioritization**: Uses Graph Attention Networks for deep learning-based gene prioritization
- **Clinical Validation**: Simulates and validates findings on clinical datasets

## Project Structure

```
├── data/
│   ├── gwas_chr*.tsv              # GWAS summary statistics by chromosome
│   ├── gene_coordinates.tsv       # Gene position information
│   ├── gene_embeddings.csv        # Pre-trained gene embeddings
│   ├── gene_interaction_graph.gml # Gene network structure
│   ├── type_2_diabetes_*.h5ad     # Single-cell expression data (example: T2D)
│   ├── simulated_clinical_dataset.csv
│   └── other processed data files
├── scripts/
│   ├── gnn_model.py              # GNN architecture implementation
│   ├── gnn_run.py                # Main GNN training pipeline
│   ├── graph_builder.py          # Network construction from interaction data
│   ├── gwas_loader.py            # GWAS data preprocessing
│   ├── prs_calculation.py        # Polygenic risk score computation
│   ├── cell_type_expression_specificity.py  # Cell-type specificity analysis
│   └── other utility scripts
├── models/
│   ├── prs_gnn_model.pt         # Trained GNN model weights
│   └── gene_embeddings.pt       # Learned gene embeddings
├── results/
│   ├── cellect_output/          # Cell-type enrichment results
│   ├── cellex_output/           # Cell expression specificity
│   ├── targets_output/          # Gene prioritization results
│   ├── plots/                   # Visualization outputs
│   └── visualizations/
├── requirements.txt             # Python dependencies
├── prs_calculation.py           # Main calculation script (standalone)
└── README.md
```

## Dependencies

Python 3.9+ with the following packages:

```
torch
torch-geometric
pandas
numpy
scipy
scikit-learn
networkx
anndata
scanpy
matplotlib
seaborn
```

Full list in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rupanjalisingh/Single_cell_polygenic_risk_score.git
cd Single_cell_polygenic_risk_score
```

2. Create a virtual environment:
```bash
python3 -m venv scprs_env
source scprs_env/bin/activate  # On Windows: scprs_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Compute PRS from GWAS data
```bash
python prs_calculation.py --gwas data/gwas_summary_statistics.tsv --output results/
```

### 2. Build gene interaction network
```bash
python scripts/graph_builder.py --string-edges data/string_edges.csv --output data/gene_interaction_graph.gml
```

### 3. Train GNN model
```bash
python scripts/gnn_run.py --data data/type_2_diabetes_pancreas.h5ad --graph data/gene_interaction_graph.gml --output models/
```

### 4. Perform cell-type enrichment analysis
```bash
python scripts/cell_type_expression_specificity.py --h5ad data/type_2_diabetes_pancreas.h5ad --output results/cellex_output/
```

## Key Features

- **Multi-omics Integration**: Combines genomics (GWAS) with transcriptomics (scRNA-seq)
- **Network-based Analysis**: Leverages protein-protein interactions to contextualize genetic associations
- **Deep Learning**: Graph Neural Networks for improved prediction accuracy
- **Cell-type Resolution**: Identifies disease mechanisms in specific cell populations
- **Reproducible Pipeline**: Modular, well-documented code for transparent analysis

## Data Requirements

### Input Files
- **GWAS**: Tab-separated summary statistics (SNP ID, chromosome, position, effect size, p-value)
- **scRNA-seq**: H5AD format (AnnData object) with gene expression matrix
- **Gene Network**: Protein-protein interaction network (edge list or GML format)
- **Annotations**: Gene coordinates and metadata

### Key Data Columns
- GWAS: `SNP`, `CHR`, `BP`, `BETA`/`OR`, `P`
- Gene info: `gene_id`, `gene_name`, `chromosome`, `start`, `end`

## Methodology

1. **GWAS QC**: Filters and harmonizes GWAS summary statistics
2. **PRS Calculation**: Computes weighted sum of effect alleles
3. **Gene-level Aggregation**: Maps variants to genes and aggregates scores
4. **Network Integration**: Embeds genes in interaction network
5. **GNN Learning**: Trains attention-based network on gene-cell associations
6. **Target Ranking**: Prioritizes genes based on predicted disease association
7. **Enrichment**: Tests for cell-type specific associations using CELLECT

## Results Interpretation

### Output Files
- `gene_level_prs.csv`: PRS scores per gene
- `celltype_prs_scores.csv`: Cell-type specific risk patterns
- `gnn_gene_prs.csv`: GNN-predicted gene importance
- `T2D_prioritized_targets.tsv`: Top candidate therapeutic targets

## Citation

If you use this repository, please cite:
```
@software{singh2024scprs,
  title={Single Cell Polygenic Risk Score: Integrating GWAS and scRNA-seq for Disease Gene Discovery},
  author={Singh, Rupanjali},
  year={2024},
  url={https://github.com/rupanjalisingh/Single_cell_polygenic_risk_score}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the author at [your email].

## Related Resources

- [CELLECT Documentation](https://github.com/immunogenomics/CELLECT)
- [Graph Neural Networks](https://pytorch-geometric.readthedocs.io/)
- [Single-cell Analysis with Scanpy](https://scanpy.readthedocs.io/)
- [GWAS File Format](https://github.com/snpEff/snpSift/wiki/gwasFile)

---

**Last Updated**: April 2024  
**Status**: Active Development
