"""
Graph Neural Network for PRS propagation
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PRSGNN(torch.nn.Module):

    def __init__(self):

        super(PRSGNN, self).__init__()

        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 16)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return x