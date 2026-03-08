import pandas as pd

# Read DEG file
deg = pd.read_csv("deg_filtered.csv")

# Read GWAS annotated file
gwas = pd.read_csv("gwas_chr22_gene_annotated.tsv", sep="\t")

# Clean column names (remove spaces)
deg.columns = deg.columns.str.strip()
gwas.columns = gwas.columns.str.strip()

# Extract unique genes
deg_genes = set(deg["gene"].dropna().unique())
gwas_genes = set(gwas["gene"].dropna().unique())

# Find overlap
overlap_genes = deg_genes.intersection(gwas_genes)

# Convert to dataframe
overlap_df = pd.DataFrame({"gene": list(overlap_genes)})

# Print results
print("Number of overlapping genes:", len(overlap_df))
print(overlap_df.head())

# Save overlapping genes
overlap_df.to_csv("deg_gwas_overlap_genes.csv", index=False)

print("Saved overlapping genes to deg_gwas_overlap_genes.csv")