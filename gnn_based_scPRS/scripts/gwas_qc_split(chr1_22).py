import pandas as pd

INPUT_FILE = "gwas_qc_filtered.tsv"
CHUNK_SIZE = 500000

written = {}

for chunk in pd.read_csv(
        INPUT_FILE,
        sep=r"\s+",   # FIXED
        chunksize=CHUNK_SIZE,
        dtype={"chr": str},
        low_memory=False):

    chunk["chr"] = chunk["chr"].astype(str)

    for chrom in range(1, 23):

        chrom = str(chrom)

        chr_rows = chunk[chunk["chr"] == chrom]

        if chr_rows.empty:
            continue

        output_file = f"gwas_chr{chrom}.tsv"

        if chrom not in written:
            chr_rows.to_csv(output_file, sep="\t", index=False, mode="w")
            written[chrom] = True
        else:
            chr_rows.to_csv(output_file, sep="\t", index=False, mode="a", header=False)

        print(f"Saved {len(chr_rows)} rows → {output_file}")

print("Finished splitting GWAS file.")