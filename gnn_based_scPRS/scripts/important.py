import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Data
# -----------------------------
cell_prs = pd.read_csv("celltype_prs_scores.csv")
gene_prs = pd.read_csv("gene_level_prs.csv")

print("Cell PRS Columns:", cell_prs.columns)
print("Gene PRS Columns:", gene_prs.columns)

# -----------------------------
# Identify Important Cell Types
# -----------------------------
cell_importance = (
    cell_prs.groupby("cluster")["gene_prs"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print("\nTop Cell Types Associated with Type 2 Diabetes:")
print(cell_importance.head(10))

# -----------------------------
# Plot Important Cell Types
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(
    data=cell_importance.head(10),
    x="gene_prs",
    y="cluster"
)

plt.title("Top Cell Types Associated with Type 2 Diabetes (PRS)")
plt.xlabel("PRS Score")
plt.ylabel("Cell Type")
plt.tight_layout()
plt.show()


# -----------------------------
# Identify PRS column automatically
# -----------------------------
score_col = "gene_prs"
gene_col = "gene"

# -----------------------------
# Identify Important Genes
# -----------------------------
top_genes = (
    gene_prs.sort_values(by=score_col, ascending=False)
    .drop_duplicates("gene")
    .head(20)
)

print("\nTop Genes Associated with Type 2 Diabetes:")
print(top_genes)

# -----------------------------
# Plot Important Genes
# -----------------------------
plt.figure(figsize=(9,6))
sns.barplot(
    data=top_genes,
    x=score_col,
    y=gene_col
)

plt.title("Top Genes Associated with Type 2 Diabetes")
plt.xlabel("Gene PRS Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()


# -----------------------------
# Heatmap of PRS by Cell Type
# -----------------------------
if "cluster" in gene_prs.columns:
    heatmap_data = gene_prs.pivot_table(
        index="cluster",
        values="gene_prs",
        aggfunc="mean"
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True)

    plt.title("Cell-Type PRS Distribution")
    plt.tight_layout()
    plt.show()