import argparse
from pathlib import Path
import osmnx as ox
import networkx as nx


def enrich_graph(in_path: str, out_path: str) -> None:
    
    print(f"Loading graph from {in_path}")
    G = ox.load_graphml(in_path)

    # Merge consecutive edges to simplify the graph
    print("Simplifying graph ...")
    try:
        G = ox.simplify_graph(G)
    except Exception as e:
        if "already been simplified" in str(e):
            print("Graph is already simplified, skipping simplification step.")
        else:
            raise


    # Estimate speeds and travel times for each edge
    print("Adding edge speeds and travel times ...")
    G = ox.add_edge_speeds(G)           # Adds speed_kph column
    G = ox.add_edge_travel_times(G)     # Adds travel_time column

    # Remove extra attributes we don't need to keep the file smaller
    keep_attrs = [
        "length", "speed_kph", "travel_time",
        "highway", "lanes", "maxspeed", "surface", "oneway"
    ]
    for u, v, k, data in G.edges(keys=True, data=True):
        keys_to_remove = [key for key in data if key not in keep_attrs]
        for key in keys_to_remove:
            del data[key]

    # Save the final enriched graph
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out_path)

    print(f"Saved enriched graph to {out_path}")
    print(f"Nodes: {len(G.nodes):,}, Edges: {len(G.edges):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and enrich OSM road graph.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input raw .graphml file")
    parser.add_argument("--out", dest="out_path", required=True, help="Output enriched .graphml file")
    args = parser.parse_args()

    enrich_graph(args.in_path, args.out_path)
