"""
Graph construction module for PRS propagation
"""

import torch
import pandas as pd
from torch_geometric.data import Data


def build_gene_graph(prs_file="gene_level_prs.csv"):
    """
    Build gene interaction graph.

    Nodes = genes
    Node features = gene PRS scores
    Edges = gene adjacency (placeholder)
    """

    df = pd.read_csv(prs_file)

    genes = df["gene"].unique()

    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    # node features
    features = torch.tensor(
        df.groupby("gene")["gene_prs"].mean().values,
        dtype=torch.float
    ).unsqueeze(1)

    edges = []

    # simple neighbor graph
    for i in range(len(genes) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=features, edge_index=edge_index)

    return graph