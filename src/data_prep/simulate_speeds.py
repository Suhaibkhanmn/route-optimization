import os
import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np
from pathlib import Path

def simulate_hourly_speeds(graph_path, out_path):
    print(f"Loading graph from {graph_path}")
    G = ox.load_graphml(graph_path)

    # Handle OSMnx version differences
    try:
        edges = ox.graph_to_gdfs(G, nodes=False)
    except AttributeError:
        edges = ox.utils_graph_to_gdfs(G, nodes=False)

    # Extract u, v if stored in index
    if 'u' not in edges.columns or 'v' not in edges.columns:
        edges = edges.reset_index()
        print("Extracted 'u' and 'v' from index")

    records = []

    for _, row in edges.iterrows():
        # Clean highway field
        highway = row.get("highway", "residential")
        if isinstance(highway, list):
            highway = ",".join(map(str, highway))  # convert list -> comma-separated string

        base_speed = row.get("speed_kph", 30) or 30
        base_time = row.get("travel_time", row["length"] / max(base_speed, 5))

        for hour in range(24):
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                slowdown = 1.6
            elif 11 <= hour <= 16:
                slowdown = 1.3
            else:
                slowdown = 1.0

            surface = row.get("surface", "")
            if isinstance(surface, str) and ("gravel" in surface or "unpaved" in surface):
                slowdown *= 1.2

            true_time = base_time * slowdown

            records.append({
                "u": row["u"],
                "v": row["v"],
                "key": row.get("key", 0),
                "edge_id": f"{row['u']}_{row['v']}",
                "highway": highway,
                "length": row["length"],
                "base_time": base_time,
                "hour": hour,
                "true_time": true_time,
            })

    df = pd.DataFrame(records)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved hourly simulation to {out_path} ({len(df):,} rows)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Simulate hourly travel times for each edge in a road network")
    p.add_argument("--graph", required=True, help="Path to input GraphML file")
    p.add_argument("--out", required=True, help="Path to save simulated hourly Parquet")
    args = p.parse_args()
    simulate_hourly_speeds(args.graph, args.out)
