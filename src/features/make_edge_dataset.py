import os
import torch
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data

def make_edge_dataset(graph_path, speeds_path, out_path):
    print(f"Loading graph: {graph_path}")
    G = ox.load_graphml(graph_path)

    # Handle OSMnx version differences
    try:
        edges = ox.graph_to_gdfs(G, nodes=False)
    except AttributeError:
        from osmnx import utils_graph
        edges = utils_graph.graph_to_gdfs(G, nodes=False)

    # Reset index if u, v missing
    if "u" not in edges.columns or "v" not in edges.columns:
        edges = edges.reset_index()

    print(f"Edges loaded: {len(edges):,}")

    # Load simulated hourly speeds
    print(f"Loading speeds: {speeds_path}")
    df = pd.read_parquet(speeds_path)
    print(f"Speed records: {len(df):,}")

    # Merge graph edges with speed table
    df["edge_id"] = df["u"].astype(str) + "_" + df["v"].astype(str)
    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str)
    merged = pd.merge(df, edges, on="edge_id", how="inner")

    print(f"Merged dataset: {len(merged):,} records")

    # Prepare edge-level features
    X = pd.DataFrame({
        "length": merged["length_x"].astype(float),
        "hour": merged["hour"].astype(int),
    })

    # One-hot encode hour (0–23)
    hour_oh = pd.get_dummies(X["hour"], prefix="h")
    X = pd.concat([X.drop(columns=["hour"]), hour_oh], axis=1)

    # Target variable
    y = merged["true_time"].astype(float)

    # Build edge index (PyTorch Geometric format)
    # Note: cast to numpy array before torch.tensor to avoid "list of ndarrays" warning
    u = merged["u_x"].astype(np.int64)
    v = merged["v_x"].astype(np.int64)
    edge_index = torch.tensor(np.array([u.values, v.values]), dtype=torch.long)

    # Ensure all features numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Convert to tensors
    x_edge = torch.tensor(X.to_numpy(dtype=np.float32))
    y_edge = torch.tensor(y.to_numpy(dtype=np.float32)).view(-1, 1)

    data = Data(edge_index=edge_index, x_edge=x_edge, y_edge_time=y_edge)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"Saved dataset to {out_path} (edges: {edge_index.shape[1]:,}, features: {x_edge.shape[1]})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build edge-level dataset with features and hourly true times")
    p.add_argument("--graph", required=True, help="Path to enriched graph file (.graphml)")
    p.add_argument("--speeds", required=True, help="Path to hourly speed parquet file")
    p.add_argument("--out", required=True, help="Output .pt dataset path")
    a = p.parse_args()
    make_edge_dataset(a.graph, a.speeds, a.out)
