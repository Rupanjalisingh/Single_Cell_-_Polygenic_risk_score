import torch

import torch
from graph_builder import build_gene_graph
from gnn_model import PRSGNN


def train():

    graph = build_gene_graph()

    model = PRSGNN()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):

        model.train()
        optimizer.zero_grad()

        output = model(graph)

        loss = torch.mean((output - graph.x) ** 2)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss {loss.item()}")

    # save trained model
    torch.save(model.state_dict(), "prs_gnn_model.pt")

    print("Model saved: prs_gnn_model.pt")


if __name__ == "__main__":
    train()