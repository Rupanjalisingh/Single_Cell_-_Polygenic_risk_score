import scanpy as sc

# Load in backed mode (does not load entire matrix)
adata = sc.read_h5ad("type_2_diabetes_pancreas.h5ad", backed="r")

print(adata)
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)

# Display basic information about the AnnData object
print(adata)
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print("Observation (cell) metadata:\n", adata.obs.head())
print("Variable (gene) metadata:\n", adata.var.head())
print("Available obsm (embeddings) keys:\n", adata.obsm.keys())
print("Available uns (unstructured annotation) keys:\n", adata.uns.keys())

print("Unique cell types:\n", adata.obs['cell_type'].unique())
print("\nNumber of cells per cell type:\n", adata.obs['cell_type'].value_counts())

print("Unique genders:\n", adata.obs['gender'].unique())
print("\nNumber of cells per gender:\n", adata.obs['gender'].value_counts())

print("Unique races:\n", adata.obs['race'].unique())
print("\nNumber of cells per race:\n", adata.obs['race'].value_counts())

print("Descriptive statistics for 'nCount_RNA':\n", adata.obs['nCount_RNA'].describe())
print("\nDescriptive statistics for 'nFeature_RNA':\n", adata.obs['nFeature_RNA'].describe())
print("\nDescriptive statistics for 'age':\n", adata.obs['age'].describe())
'''
import numpy as np

# Randomly sample 5000 cells
subset_cells = np.random.choice(adata.n_obs, 5000, replace=False)

adata_subset = adata[subset_cells, :].to_memory()

# Save smaller file
adata_subset.write("pancreas_subset_5k.h5ad")

import scanpy as sc

adata = sc.read_h5ad("type_2_diabetes_pancreas.h5ad")

chunk_size = 5000
num_chunks = adata.n_obs // chunk_size + 1

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, adata.n_obs)

    chunk = adata[start:end].copy()
    chunk.write(f"pancreas_chunk_{i}.h5ad")

    print(f"Saved chunk {i}")

    import scanpy as sc

adata = sc.read_h5ad("type_2_diabetes_pancreas.h5ad")

# Randomly sample 10k cells
adata = sc.pp.subsample(adata, n_obs=10000, copy=True)

adata.write("pancreas_10k_cells.h5ad")

import scanpy as sc

adata = sc.read_h5ad("pancreas_subset_5k.h5ad")

# QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Highly variable genes
sc.pp.highly_variable_genes(adata)

# PCA
sc.tl.pca(adata)

# Neighborhood graph
sc.pp.neighbors(adata)

# UMAP
sc.tl.umap(adata)

# Clustering
sc.tl.leiden(adata)

# Plot
sc.pl.umap(adata, color=['leiden'])
sc.pl.pca(adata, color=['leiden'])

'''