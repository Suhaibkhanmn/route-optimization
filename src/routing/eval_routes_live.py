import os
import torch
import pandas as pd
import networkx as nx
import osmnx as ox
from torch_geometric.data import Data
from src.models.edge_time_gnn import EdgeTimeGNN
from src.routing.eval_routes_offline import estimate_route_eta


def load_and_merge(base_edges_path, live_path):
    """Merge static edge features with live traffic speeds."""
    base = pd.read_parquet(base_edges_path)
    live = pd.read_parquet(live_path)

    live = live.drop_duplicates(subset=["lat", "lon"])
    merged = pd.merge(
        base,
        live[["lat", "lon", "currentSpeed", "freeFlowSpeed", "confidence"]],
        on=["lat", "lon"],
        how="left",
    )

    merged["currentSpeed"] = merged["currentSpeed"].fillna(merged["freeFlowSpeed"])
    merged["currentSpeed"] = merged["currentSpeed"].fillna(merged["length_m"] / 10)
    merged["confidence"] = merged["confidence"].fillna(1.0)

    print(f"Merged {len(live)} live records into {len(base)} edges")
    return merged


def to_torch_data(merged_df):
    """Convert merged DataFrame to a PyTorch Geometric Data object with numeric edge features."""
    # Only use numeric columns for features
    numeric_df = merged_df.select_dtypes(include=["number"]).copy()

    # Make sure we have u and v columns for building edge connections
    if "u" not in numeric_df.columns or "v" not in numeric_df.columns:
        numeric_df["u"] = merged_df["u"].astype(int)
        numeric_df["v"] = merged_df["v"].astype(int)

    # Pick all columns except u and v as features
    feature_cols = [c for c in numeric_df.columns if c not in ["u", "v"]]
    x_edge = numeric_df[feature_cols].fillna(0)

    # Make sure we have exactly 25 features (pad or trim if needed)
    if x_edge.shape[1] < 25:
        pad_cols = 25 - x_edge.shape[1]
        for i in range(pad_cols):
            x_edge[f"pad_{i}"] = 0
    elif x_edge.shape[1] > 25:
        x_edge = x_edge.iloc[:, :25]

    # Turn everything into PyTorch tensors
    x_edge = torch.tensor(x_edge.values, dtype=torch.float)
    u = torch.tensor(numeric_df["u"].values, dtype=torch.long)
    v = torch.tensor(numeric_df["v"].values, dtype=torch.long)
    edge_index = torch.stack([u, v], dim=0)

    return Data(edge_index=edge_index, x_edge=x_edge)


def run_live_eval(graph_path, merged_edges, model_path, n_routes=20):
    """Evaluate ETAs using live-traffic-enhanced edge features."""
    print("Loading model...")
    model = EdgeTimeGNN(in_channels=25, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Loading graph...")
    G = ox.load_graphml(graph_path)
    print(f"Graph nodes: {len(G.nodes)}, edges: {len(G.edges)}")

    edges = ox.graph_to_gdfs(G, nodes=False)

    # Deal with newer OSMnx versions that store u, v in the index
    if isinstance(edges.index, pd.MultiIndex) and "u" not in edges.columns:
        print("Detected newer OSMnx format - extracting u, v from index")
        idx_df = pd.DataFrame(edges.index.to_list(), columns=["u", "v", "_key"])
        edges = edges.reset_index(drop=True)
        edges = pd.concat([edges, idx_df[["u", "v"]]], axis=1)

    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    # Turn the merged data into the format PyTorch Geometric expects
    data = to_torch_data(merged_edges)

    print("Evaluating with live traffic speeds...")
    for i in range(n_routes):
        try:
            start = int(edges.sample(1)["u"].iloc[0])
            end = int(edges.sample(1)["v"].iloc[0])

            baseline = nx.shortest_path_length(G, start, end, weight="length")
            pred_eta = estimate_route_eta(G, model, data, start, end)
            diff = (pred_eta - baseline) / baseline * 100

            print(f"Route {i+1}: baseline={baseline:.2f}, pred={pred_eta:.2f}, delta={diff:+.2f}%")
        except Exception as e:
            print(f"Route {i+1} failed: {e}")
            continue

    print("Live evaluation complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to .graphml file")
    parser.add_argument("--edges", required=True, help="Path to static edge parquet file")
    parser.add_argument("--live", required=True, help="Path to live speeds parquet file")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--n", type=int, default=20, help="Number of random routes to test")
    args = parser.parse_args()

    merged = load_and_merge(args.edges, args.live)
    run_live_eval(args.graph, merged, args.model, n_routes=args.n)
