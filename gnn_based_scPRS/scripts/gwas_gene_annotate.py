import pandas as pd
import pyranges as pr

GWAS_FILE = "gwas_chr22.tsv"
GENE_FILE = "gene_coordinates.tsv"
OUTPUT_FILE = "gwas_chr22_gene_annotated.tsv"

# Load gene coordinates
genes = pd.read_csv(GENE_FILE, sep="\t")

# Keep only chromosome 22 genes
genes = genes[genes["chr"].astype(str) == "22"]

# Prepare gene ranges
gene_ranges = pr.PyRanges(
    pd.DataFrame({
        "Chromosome": genes["chr"].astype(str),
        "Start": genes["start"],
        "End": genes["end"],
        "gene": genes["gene"]
    })
)

# Load GWAS chr22 data
gwas = pd.read_csv(GWAS_FILE, sep="\t")

gwas["chr"] = gwas["chr"].astype(str)
gwas["pos"] = pd.to_numeric(gwas["pos"], errors="coerce")

# Create SNP ranges
snp_ranges = pr.PyRanges(
    pd.DataFrame({
        "Chromosome": gwas["chr"],
        "Start": gwas["pos"],
        "End": gwas["pos"]
    })
)

# Find overlaps
overlaps = snp_ranges.join(gene_ranges)

if overlaps.df.empty:
    print("No gene overlaps found.")
else:
    overlap_df = overlaps.df

    mapped = gwas.loc[overlap_df.index].copy()
    mapped["gene"] = overlap_df["gene"].values

    mapped.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print(f"Annotated SNPs saved to {OUTPUT_FILE}")