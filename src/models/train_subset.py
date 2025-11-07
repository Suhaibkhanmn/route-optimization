import torch
import random
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
from src.models.edge_time_gnn import EdgeTimeGNN
import torch_geometric

# Allowlist torch_geometric Data class for safe loading
torch.serialization.add_safe_globals([torch_geometric.data.Data])

def train_subset(data_path, out_dir, subset_size=100_000, epochs=10, lr=1e-3):
    print(f"Loading dataset from {data_path}")

    # Disable weights_only restriction (PyTorch 2.6+)
    data = torch.load(data_path, weights_only=False)

    n_edges = data.x_edge.shape[0]
    print(f"Dataset has {n_edges:,} edges — using subset of {subset_size:,}")

    # Randomly select subset of edges
    idx = np.random.choice(n_edges, min(subset_size, n_edges), replace=False)
    x_edge = data.x_edge[idx]
    y_edge = data.y_edge_time[idx]

    # Use all edges for connectivity (not subset-based, avoids shape issues)
    edge_index = data.edge_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = EdgeTimeGNN(
        in_channels=x_edge.shape[1],
        hidden_channels=64,
        out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()

    x_edge, y_edge = x_edge.to(device), y_edge.to(device)
    edge_index = edge_index.to(device)

    print("Starting subset training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(None, edge_index, x_edge)
        loss = loss_fn(pred, y_edge)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(out_dir) / "edge_time_sage_subset.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Subset model saved at {model_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train GNN on a small random subset of edges")
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="models_subset")
    p.add_argument("--subset_size", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    train_subset(args.data, args.out_dir, args.subset_size, args.epochs, args.lr)
