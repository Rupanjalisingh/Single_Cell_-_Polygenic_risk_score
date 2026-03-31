import pandas as pd
import networkx as nx


def build_gene_graph():

    # -----------------------------
    # Load data
    # -----------------------------
    edges = pd.read_csv("string_edges.csv")
    gene_prs = pd.read_csv("gene_level_prs.csv")

    print("Edges file:")
    print(edges.head())

    print("\nGene PRS file:")
    print(gene_prs.head())


    # -----------------------------
    # Create graph
    # -----------------------------
    G = nx.Graph()

    # Add edges
    for _, row in edges.iterrows():
        G.add_edge(row["gene1"], row["gene2"])


    # -----------------------------
    # Add node features (PRS)
    # -----------------------------
    prs_dict = gene_prs.groupby("gene")["gene_prs"].mean().to_dict()

    for gene, prs in prs_dict.items():
        if gene in G.nodes:
            G.nodes[gene]["prs"] = prs


    # -----------------------------
    # Graph summary
    # -----------------------------
    print("\nGraph created successfully")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())


    # -----------------------------
    # Save graph
    # -----------------------------
    nx.write_gml(G, "gene_interaction_graph.gml")

    print("\nGraph saved as gene_interaction_graph.gml")


if __name__ == "__main__":
    build_gene_graph()