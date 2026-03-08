import torch
import pandas as pd

from graph_builder import build_gene_graph
from gnn_model import PRSGNN


def extract_scores():

    # rebuild graph
    graph = build_gene_graph()

    # create model
    model = PRSGNN()

    # load trained weights
    model.load_state_dict(torch.load("prs_gnn_model.pt"))

    model.eval()

    # run model
    with torch.no_grad():
        output = model(graph)

    gnn_scores = output.squeeze().numpy()

    # load gene list
    df = pd.read_csv("gene_level_prs.csv")

    gene_scores = df.groupby("gene")["gene_prs"].mean().reset_index()

    # attach GNN scores
    gene_scores["gnn_prs"] = gnn_scores

    # save results
    gene_scores.to_csv("gnn_gene_prs.csv", index=False)

    print("Saved: gnn_gene_prs.csv")


if __name__ == "__main__":
    extract_scores()

# cell level prs after gnn
import pandas as pd

deg = pd.read_csv("deg_filtered.csv")
gene_scores = pd.read_csv("gnn_gene_prs.csv")

merged = pd.merge(gene_scores, deg, on="gene")

merged["cell_score"] = merged["gnn_prs"] * merged["avg_log2FC"]

cell_prs = merged.groupby("cluster")["cell_score"].sum().reset_index()

print(cell_prs)