import pandas as pd

INPUT_FILE = "gwas_summary_statistics.tsv"
OUTPUT_FILE = "gwas_qc_filtered.tsv"

CHUNK_SIZE = 300000


def perform_gwas_qc(df):

    print("Initial rows:", len(df))

    # Rename columns
    df = df.rename(columns={
        "chromosome": "chr",
        "base_pair_location": "pos",
        "variant_id": "snp",
        "INFO-score": "info",
        "effect_allele_frequency": "eaf"
    })

    # Convert allele columns to uppercase
    df["effect_allele"] = df["effect_allele"].str.upper()
    df["other_allele"] = df["other_allele"].str.upper()

    # Convert numeric columns
    numeric_cols = [
        "pos",
        "beta",
        "standard_error",
        "p_value",
        "info",
        "eaf"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("After numeric conversion:", len(df))

    # Remove rows with missing essential fields
    df = df.dropna(subset=["snp", "beta", "p_value"])

    print("After NA removal:", len(df))

    # Valid p-values
    df = df[(df["p_value"] > 0) & (df["p_value"] <= 1)]

    print("After p-value filter:", len(df))

    # INFO filter (relaxed)
    df = df[df["info"] >= 0.8]

    print("After INFO filter:", len(df))

    # MAF filter (relaxed)
    df = df[(df["eaf"] > 0.01) & (df["eaf"] < 0.99)]

    print("After MAF filter:", len(df))

    # Valid alleles
    valid = {"A", "T", "C", "G"}

    df = df[
        df["effect_allele"].isin(valid) &
        df["other_allele"].isin(valid)
    ]

    print("After allele filter:", len(df))

    return df


def process_gwas():

    first = True

    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        cleaned = perform_gwas_qc(chunk)

        cleaned.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first
        )

        first = False


if __name__ == "__main__":
    process_gwas()

'''
| Filter                   | Threshold   |
| ------------------------ | ----------- |
| INFO score               | ≥ 0.8      |
| Allele frequency         | 0.01 – 0.99|
| p-value                  | 0 < p ≤ 1   |
| Genome-wide significance | 5 × 10⁻⁸    |

'''
