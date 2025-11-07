import torch
import osmnx as ox
import networkx as nx
import numpy as np
import random
import argparse
import torch_geometric
from src.models.edge_time_gnn import EdgeTimeGNN


def estimate_route_eta(G, model, data, start, end):
    """Estimate ETA between start and end using model-predicted edge times."""
    x_edge = data.x_edge
    with torch.no_grad():
        preds = model(None, data.edge_index, x_edge).squeeze().cpu().numpy()

    # Load edge info from the graph
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.reset_index()  # Make sure u, v, key columns are available
    edges["pred_time"] = preds[: len(edges)]

    # Map predictions to edges including the key (important for multi-edge nodes)
    nx.set_edge_attributes(
        G,
        {(row["u"], row["v"], row["key"]): row["pred_time"] for _, row in edges.iterrows()},
        "pred_time",
    )

    try:
        pred_path = nx.shortest_path(G, source=start, target=end, weight="pred_time")
        pred_eta = sum(
            G[u][v][0].get("pred_time", G[u][v][0].get("length", 0))
            for u, v in zip(pred_path[:-1], pred_path[1:])
        )
        return pred_eta
    except Exception:
        return None


def run_eval(graph_path, dataset_path, model_path, n_pairs=20):
    print(f"Loading graph: {graph_path}")
    G = ox.load_graphml(graph_path)

    print(f"Loading dataset: {dataset_path}")
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.Data,
        torch_geometric.data.storage.GlobalStorage,
    ])
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    print(f"Loading model: {model_path}")
    model = EdgeTimeGNN(
        in_channels=data.x_edge.shape[1],
        hidden_channels=64,
        out_channels=1,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"Dataset features: {data.x_edge.shape[1]} columns")
    print("Evaluating routes...\n")

    node_ids = list(G.nodes())
    total_diff = []
    total_pairs = 0

    for _ in range(n_pairs):
        start, end = random.sample(node_ids, 2)
        try:
            baseline_path = nx.shortest_path(G, source=start, target=end, weight="length")
            baseline_eta = sum(
                G[u][v][0].get("length", 0)
                / max(G[u][v][0].get("speed_kph", 1), 1)
                for u, v in zip(baseline_path[:-1], baseline_path[1:])
            )
        except Exception:
            continue

        pred_eta = estimate_route_eta(G, model, data, start, end)
        if pred_eta is None:
            continue

        diff = abs(pred_eta - baseline_eta) / baseline_eta
        total_diff.append(diff)
        total_pairs += 1
        print(f"Route {total_pairs}: baseline={baseline_eta:.2f}, pred={pred_eta:.2f}, diff={diff:.2%}")

    if total_pairs == 0:
        print("No valid routes found.")
    else:
        print(f"\nAvg ETA deviation: {np.mean(total_diff) * 100:.2f}% over {total_pairs} routes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    run_eval(args.graph, args.dataset, args.model, n_pairs=args.n)
