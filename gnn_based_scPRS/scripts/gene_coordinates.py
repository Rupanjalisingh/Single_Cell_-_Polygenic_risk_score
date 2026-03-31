import pandas as pd

gtf_file = "gencode.v38.annotation.gtf"

genes = []

with open(gtf_file) as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split("\t")

        if parts[2] != "gene":
            continue

        chr = parts[0].replace("chr", "")
        start = int(parts[3])
        end = int(parts[4])

        info = parts[8]

        gene_name = info.split('gene_name "')[1].split('"')[0]

        genes.append([chr, start, end, gene_name])

gene_df = pd.DataFrame(genes, columns=["chr", "start", "end", "gene"])

gene_df.to_csv("gene_coordinates.tsv", sep="\t", index=False)

print(gene_df.head())