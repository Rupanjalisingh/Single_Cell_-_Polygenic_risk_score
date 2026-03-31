"""
GNN pipeline for PRS gene interaction network

Steps:
1. Load STRING edges
2. Load gene PRS features
3. Build NetworkX graph
4. Convert graph to PyTorch Geometric format
5. Train Graph Neural Network
"""

import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ------------------------------------------------
# Load data
# ------------------------------------------------

def load_data():

    edges = pd.read_csv("string_edges.csv")
    gene_prs = pd.read_csv("gene_level_prs.csv")

    print("Edges:", edges.shape)
    print("Gene PRS:", gene_prs.shape)

    return edges, gene_prs


# ------------------------------------------------
# Build NetworkX graph
# ------------------------------------------------

def build_graph(edges):

    G = nx.Graph()

    for _, row in edges.iterrows():
        G.add_edge(row["gene1"], row["gene2"])

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G


# ------------------------------------------------
# Add PRS node features
# ------------------------------------------------

def add_node_features(G, gene_prs):

    prs_dict = gene_prs.groupby("gene")["gene_prs"].mean().to_dict()

    for node in G.nodes():

        if node in prs_dict:
            G.nodes[node]["prs"] = prs_dict[node]
        else:
            G.nodes[node]["prs"] = 0

    return G


# ------------------------------------------------
# Convert graph to PyTorch Geometric
# ------------------------------------------------

def convert_to_pyg(G):

    node_list = list(G.nodes())

    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Create edge_index
    edge_index = []

    for u, v in G.edges():

        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # Create node features
    features = []

    for node in node_list:
        prs = G.nodes[node]["prs"]
        features.append([prs])

    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    print(data)

    return data


# ------------------------------------------------
# Define GNN model
# ------------------------------------------------

class GCN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x


# ------------------------------------------------
# Train GNN
# ------------------------------------------------

def train_gnn(data):

    model = GCN()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining GNN...")

    for epoch in range(50):

        optimizer.zero_grad()

        out = model(data)

        loss = out.mean()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

    return model, out


# ------------------------------------------------
# Save embeddings
# ------------------------------------------------

def save_embeddings(G, embeddings):

    node_list = list(G.nodes())

    emb_df = pd.DataFrame(embeddings.detach().numpy())

    emb_df["gene"] = node_list

    emb_df.to_csv("gene_embeddings.csv", index=False)

    print("\nSaved embeddings: gene_embeddings.csv")


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------

def main():

    edges, gene_prs = load_data()

    G = build_graph(edges)

    G = add_node_features(G, gene_prs)

    data = convert_to_pyg(G)

    model, embeddings = train_gnn(data)

    save_embeddings(G, embeddings)


if __name__ == "__main__":
    main()