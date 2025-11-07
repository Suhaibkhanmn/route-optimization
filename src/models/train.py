import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from edge_time_gnn import EdgeTimeSAGE


def train_model(data_path, out_dir, epochs=100, patience=10, lr=1e-3, hidden_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data = torch.load(data_path)
    x_node = data["x_node"].to(device)
    edge_index = data["edge_index"].to(device)
    x_edge = data["x_edge"].to(device)
    y = data["y"].to(device)
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]

    model = EdgeTimeSAGE(in_node_feats=x_node.shape[1],
                         in_edge_feats=x_edge.shape[1],
                         hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    best_val_mae = float("inf")
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_node, edge_index, x_edge)
        loss = criterion(y_pred[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_node, edge_index, x_edge)
            val_mae = (y_val_pred[val_mask] - y[val_mask]).abs().mean().item()

        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val MAE: {val_mae:.4f}")

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            torch.save(model.state_dict(), out_dir / "edge_time_sage_v1.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation
    model.load_state_dict(torch.load(out_dir / "edge_time_sage_v1.pt"))
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_node, edge_index, x_edge)
        test_mae = (y_test_pred[test_mask] - y[test_mask]).abs().mean().item()
    print(f"Test MAE: {test_mae:.4f}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN to predict edge travel times.")
    parser.add_argument("--data", required=True, help="Path to dataset (.pt)")
    parser.add_argument("--out_dir", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model(args.data, args.out_dir, args.epochs, args.patience, args.lr)
