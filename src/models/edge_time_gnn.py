import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class EdgeTimeGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=1):
        """
        Lightweight edge-level model that predicts travel time using only edge features.
        - in_channels: number of features per edge
        - hidden_channels: hidden layer size
        - out_channels: predicted output (travel time)
        """
        super(EdgeTimeGNN, self).__init__()

        # Simple neural network that works directly on edge features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x_node, edge_index, x_edge):
        """
        Forward pass: ignores node features and graph structure, 
        operates directly on edge attributes.
        """
        return self.mlp(x_edge)
