import os
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString

def extract_edge_points(graph_path, out_path):
    print(f"Loading graph: {graph_path}")
    G = ox.load_graphml(graph_path)

    # Use new unified API
    try:
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    except Exception:
        # For older versions
        edges = ox.utils_graph.graph_to_gdfs(G, nodes=False)
    print(f"Edges loaded: {len(edges)}")

    data = []
    for idx, row in edges.iterrows():
        # Handle OSMnx 1.9+ where 'u','v','key' are in index
        if isinstance(idx, tuple) and len(idx) >= 2:
            u, v = idx[0], idx[1]
            key = idx[2] if len(idx) > 2 else 0
        else:
            u = row.get("u")
            v = row.get("v")
            key = row.get("key", 0)

        geom = row.geometry
        if geom is None:
            continue

        # Handle both LineString and MultiLineString
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda g: g.length)

        midpoint = geom.interpolate(0.5, normalized=True)
        data.append({
            "u": u,
            "v": v,
            "key": key,
            "lat": midpoint.y,
            "lon": midpoint.x,
            "length_m": row.get("length", 0),
            "highway": str(row.get("highway")),
            "maxspeed": str(row.get("maxspeed")),
        })

        if len(data) % 2000 == 0:
            print(f"Processed {len(data)} edges...")

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved midpoints to {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to enriched graph (.graphml)")
    parser.add_argument("--out", required=True, help="Output path for parquet file")
    args = parser.parse_args()
    extract_edge_points(args.graph, args.out)
