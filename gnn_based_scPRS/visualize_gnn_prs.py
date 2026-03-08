"""
Visualization of PRS and GNN results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from graph_builder import build_gene_graph


def plot_gene_prs_distribution():

    df = pd.read_csv("gnn_gene_prs.csv")

    plt.figure(figsize=(7,5))

    sns.histplot(df["gnn_prs"], bins=30, kde=True)

    plt.title("Distribution of GNN-smoothed Gene PRS")
    plt.xlabel("PRS score")
    plt.ylabel("Gene count")

    plt.tight_layout()
    plt.show()


def plot_celltype_prs():

    gene_scores = pd.read_csv("gnn_gene_prs.csv")
    deg = pd.read_csv("deg_filtered.csv")

    merged = pd.merge(gene_scores, deg, on="gene")

    merged["cell_score"] = merged["gnn_prs"] * merged["avg_log2FC"]

    cell_prs = merged.groupby("cluster")["cell_score"].sum().reset_index()

    plt.figure(figsize=(9,5))

    sns.barplot(
        data=cell_prs,
        x="cluster",
        y="cell_score"
    )

    plt.xticks(rotation=45)

    plt.title("Cell-type Specific PRS (After GNN)")
    plt.ylabel("PRS Score")

    plt.tight_layout()
    plt.show()


def plot_gene_graph():

    graph = build_gene_graph()

    G = nx.Graph()

    edges = graph.edge_index.t().numpy()

    for e in edges:
        G.add_edge(int(e[0]), int(e[1]))

    plt.figure(figsize=(7,7))

    nx.draw(
        G,
        node_size=50,
        with_labels=False
    )

    plt.title("Gene Interaction Graph")

    plt.show()


def main():

    plot_gene_prs_distribution()

    plot_celltype_prs()

    plot_gene_graph()


if __name__ == "__main__":
    main()