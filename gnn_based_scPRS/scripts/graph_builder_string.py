import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph():

    prs_df = pd.read_csv("gene_level_prs.csv")
    edges_df = pd.read_csv("string_edges.csv")

    genes = prs_df["gene"].unique()

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # node features
    features = torch.tensor(
        prs_df.groupby("gene")["gene_prs"].mean().values,
        dtype=torch.float
    ).unsqueeze(1)

    edge_list = []

    for _, row in edges_df.iterrows():

        g1 = row["preferredName_A"]
        g2 = row["preferredName_B"]

        if g1 in gene_to_idx and g2 in gene_to_idx:

            edge_list.append([gene_to_idx[g1], gene_to_idx[g2]])
            edge_list.append([gene_to_idx[g2], gene_to_idx[g1]])

    edge_index = torch.tensor(edge_list).t().contiguous()

    graph = Data(x=features, edge_index=edge_index)

    return graph