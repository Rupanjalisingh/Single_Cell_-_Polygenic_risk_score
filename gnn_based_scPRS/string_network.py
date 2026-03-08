"""
Download STRING protein interaction network for gene list
"""

import requests
import pandas as pd


def download_string_network(gene_list, species=9606):

    genes = "%0d".join(gene_list)

    url = "https://string-db.org/api/tsv/network"

    params = {
        "identifiers": genes,
        "species": species
    }

    response = requests.post(url, data=params)

    lines = response.text.strip().split("\n")

    data = [line.split("\t") for line in lines]

    df = pd.DataFrame(data[1:], columns=data[0])

    return df


def build_string_edges():

    gene_df = pd.read_csv("gene_level_prs.csv")

    genes = gene_df["gene"].unique().tolist()

    string_df = download_string_network(genes)

    print("STRING columns:", string_df.columns)

    # Detect STRING column format
    if {"preferredName_A", "preferredName_B"}.issubset(string_df.columns):
        edges = string_df[["preferredName_A", "preferredName_B"]]
        edges.columns = ["gene1", "gene2"]

    elif {"protein1", "protein2"}.issubset(string_df.columns):
        edges = string_df[["protein1", "protein2"]]
        edges.columns = ["gene1", "gene2"]

    else:
        raise ValueError("Unexpected STRING column format")

    edges.to_csv("string_edges.csv", index=False)

    print("Saved STRING edges to string_edges.csv")


if __name__ == "__main__":
    build_string_edges()