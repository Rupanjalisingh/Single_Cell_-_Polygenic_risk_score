"""
Gene-weighted proxy PRS pipeline

This script integrates:
1. GWAS summary statistics (gwas_chr22_gene_annotated.tsv)
2. Single-cell DEG dataset (deg_filtered.csv)

to compute:
- Gene-level proxy PRS
- Cell-type specific risk scores

Formula:
GenePRS = beta * avg_log2FC
CellTypePRS = sum(GenePRS) per cluster

Author: Rupanjali Singh
"""

import pandas as pd

# -----------------------------
# Load Data
# -----------------------------
gwas = pd.read_csv("gwas_chr22_gene_annotated.tsv", sep="\t")
deg = pd.read_csv("deg_filtered.csv")

# Clean column names
gwas.columns = gwas.columns.str.strip()
deg.columns = deg.columns.str.strip()

# -----------------------------
# Aggregate GWAS beta per gene
# -----------------------------
gwas_gene = (
    gwas
    .groupby("gene")["beta"]
    .mean()
    .reset_index()
)

# -----------------------------
# Merge GWAS + DEG
# -----------------------------
merged = pd.merge(
    gwas_gene,
    deg,
    on="gene",
    how="inner"
)

# -----------------------------
# Calculate Gene-level PRS
# -----------------------------
merged["gene_prs"] = merged["beta"] * merged["avg_log2FC"]

gene_prs = merged[["gene", "cluster", "gene_prs"]]

# Save gene PRS file
gene_prs.to_csv("gene_level_prs.csv", index=False)

print("\nGene-level PRS calculated successfully")
print("gene_level_prs.csv generated\n")
print(gene_prs.head(10))


# -----------------------------
# Calculate Cell-type PRS
# -----------------------------
cell_prs = (
    merged
    .groupby("cluster")["gene_prs"]
    .sum()
    .reset_index()
)

# Save cell PRS file
cell_prs.to_csv("celltype_prs_scores.csv", index=False)

print("\nCell-type PRS calculated successfully")
print("celltype_prs_scores.csv generated\n")
print(cell_prs)


'''
The integration of GWAS summary statistics with single-cell differential expression profiles revealed heterogeneous genetic 
risk signals across pancreatic cell types. Delta cells exhibited the highest positive proxy PRS, suggesting a potential 
role of somatostatin-secreting endocrine cells in mediating genetic susceptibility to Type 2 Diabetes. Moderate positive 
signals were also observed in ductal epithelial subtypes, indicating possible involvement of pancreatic tissue remodeling or 
stress response pathways. In contrast, immune and stromal populations such as macrophages and fibroblasts displayed negative 
aggregate scores, potentially reflecting opposing transcriptional regulation or protective expression patterns. 
However, as the analysis utilized chromosome 22 GWAS variants only, the observed risk landscape represents a partial view of 
the full genetic architecture of Type 2 Diabetes.
'''

'''
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load datasets
# -----------------------------
gwas = pd.read_csv("gwas_chr22_gene_annotated.tsv", sep="\t")
deg = pd.read_csv("deg_filtered.csv")

# -----------------------------
# Clean column names
# -----------------------------
gwas.columns = gwas.columns.str.strip()
deg.columns = deg.columns.str.strip()

# fix avg_log2FC column formatting
deg.columns = deg.columns.str.replace(" ", "")

# -----------------------------
# Select relevant columns
# -----------------------------
gwas = gwas[["snp", "gene", "beta"]]

deg = deg[[
    "cluster",
    "Group1",
    "Group2",
    "gene",
    "avg_log2FC"
]]

# -----------------------------
# Get all clusters
# -----------------------------
all_clusters = deg[["cluster", "Group1", "Group2"]].drop_duplicates()

# -----------------------------
# Merge GWAS and DEG datasets
# -----------------------------
merged = pd.merge(gwas, deg, on="gene", how="inner")

# -----------------------------
# Compute Gene-level proxy PRS
# -----------------------------
merged["gene_score"] = merged["beta"] * merged["avg_log2FC"]

# Save gene-level results
merged.to_csv("gene_level_proxy_prs.csv", index=False)

# -----------------------------
# Compute Cell-type PRS
# -----------------------------
cell_prs = (
    merged
    .groupby(["cluster", "Group1", "Group2"], as_index=False)["gene_score"]
    .sum()
)

# -----------------------------
# Ensure all clusters appear
# -----------------------------
cell_prs = pd.merge(all_clusters, cell_prs,
                    on=["cluster","Group1","Group2"],
                    how="left")

cell_prs["gene_score"] = cell_prs["gene_score"].fillna(0)

# sort clusters by PRS score
cell_prs = cell_prs.sort_values(by="gene_score", ascending=False)

# -----------------------------
# Save results
# -----------------------------
cell_prs.to_csv("celltype_prs_scores.csv", index=False)

# -----------------------------
# Print all clusters
# -----------------------------
print("\nCell-type specific PRS scores:\n")
print(cell_prs.to_string(index=False))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,6))

plt.barh(cell_prs["cluster"], cell_prs["gene_score"])

plt.xlabel("Proxy PRS Score")
plt.ylabel("Cell Type")
plt.title("Cell-Type Specific Genetic Risk Scores for Type 2 Diabetes")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# -----------------------------
# SNP contribution table
# -----------------------------
snp_contribution = merged[["snp", "gene", "cluster", "gene_score"]]

snp_contribution.to_csv("snp_gene_contribution.csv", index=False)
'''