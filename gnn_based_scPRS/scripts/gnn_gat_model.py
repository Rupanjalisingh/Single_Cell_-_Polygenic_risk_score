"""
scPRS-style Graph Neural Network pipeline

Stages:
1. Input embedding
2. Graph construction
3. Message passing (GAT)
4. Readout layer
5. Output generation
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# ------------------------------------------------
# 1. INPUT EMBEDDING
# ------------------------------------------------

def build_input_embeddings():

    gene_prs = pd.read_csv("gene_level_prs.csv")

    print("Loaded gene PRS:", gene_prs.shape)

    gene_features = gene_prs.groupby("gene")["gene_prs"].mean()

    features = gene_features.fillna(0)

    node_names = features.index.tolist()

    x = torch.tensor(features.values).unsqueeze(1).float()

    print("Node features shape:", x.shape)

    return x, node_names


# ------------------------------------------------
# 2. GRAPH CONSTRUCTION
# ------------------------------------------------

def build_string_graph(node_names):

    edges = pd.read_csv("string_edges.csv")

    G = nx.Graph()

    for _, row in edges.iterrows():
        G.add_edge(row["gene1"], row["gene2"])

    node_index = {node: i for i, node in enumerate(node_names)}

    edge_index = []

    for u, v in G.edges():

        if u in node_index and v in node_index:

            edge_index.append([node_index[u], node_index[v]])
            edge_index.append([node_index[v], node_index[u]])

    edge_index = torch.tensor(edge_index).t().contiguous()

    print("STRING edges:", edge_index.shape)

    return edge_index


# ------------------------------------------------
# OPTIONAL: similarity edges
# ------------------------------------------------

def build_similarity_edges(x, threshold=0.8):

    sim = cosine_similarity(x.numpy())

    edges = []

    for i in range(len(sim)):
        for j in range(len(sim)):

            if i != j and sim[i][j] > threshold:
                edges.append([i, j])

    edge_index = torch.tensor(edges).t().contiguous()

    print("Similarity edges:", edge_index.shape)

    return edge_index


# ------------------------------------------------
# 3. GAT MODEL (MESSAGE PASSING)
# ------------------------------------------------

class GATModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(1, 32, heads=4)

        self.gat2 = GATConv(128, 16)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x


# ------------------------------------------------
# 4. READOUT LAYER
# ------------------------------------------------

class Readout(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(16, 1)

    def forward(self, embeddings):

        out = self.fc(embeddings)

        return torch.sigmoid(out)


# ------------------------------------------------
# 5. TRAIN GNN
# ------------------------------------------------

def train_model(data):

    model = GATModel()
    readout = Readout()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(readout.parameters()),
        lr=0.01
    )

    print("\nTraining GNN")

    for epoch in range(100):

        optimizer.zero_grad()

        embeddings = model(data)

        predictions = readout(embeddings)

        loss = predictions.mean()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

    return embeddings, predictions


# ------------------------------------------------
# 6. SAVE OUTPUT
# ------------------------------------------------

def save_results(node_names, embeddings, predictions):

    emb_df = pd.DataFrame(embeddings.detach().numpy())
    emb_df["gene"] = node_names

    emb_df.to_csv("gene_embeddings.csv", index=False)

    pred_df = pd.DataFrame({
        "gene": node_names,
        "risk_score": predictions.detach().numpy().flatten()
    })

    pred_df.to_csv("gene_risk_scores.csv", index=False)

    print("\nSaved outputs:")
    print("gene_embeddings.csv")
    print("gene_risk_scores.csv")


# ------------------------------------------------
# 7. VISUALIZATION
# ------------------------------------------------

def visualize_embeddings():

    emb = pd.read_csv("gene_embeddings.csv")

    genes = emb["gene"]

    features = emb.drop(columns=["gene"])

    pca = PCA(n_components=2)

    X = pca.fit_transform(features)

    plt.figure(figsize=(8,6))

    plt.scatter(X[:,0], X[:,1], alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title("GNN Gene Embeddings")

    plt.show()


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def main():

    x, node_names = build_input_embeddings()

    string_edges = build_string_graph(node_names)

    sim_edges = build_similarity_edges(x)

    edge_index = torch.cat([string_edges, sim_edges], dim=1)

    data = Data(x=x, edge_index=edge_index)

    embeddings, predictions = train_model(data)

    save_results(node_names, embeddings, predictions)

    visualize_embeddings()


if __name__ == "__main__":

    main()