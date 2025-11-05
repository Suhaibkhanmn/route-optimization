import argparse
from pathlib import Path
import osmnx as ox


def build_osm_graph(place: str, out_path: str) -> None:
    
    print(f"Downloading road network for: {place}")

    # Download the road network (only drivable roads, no pedestrian paths)
    G = ox.graph_from_place(place, network_type="drive")

    # Create the output folder if it doesn't exist
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Save in GraphML format (XML-based, works with NetworkX and other tools)
    ox.save_graphml(G, out_path)

    print(f"Graph saved to: {out_path}")
    print(f"Nodes: {len(G.nodes):,}, Edges: {len(G.edges):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSM road graph for a city.")
    parser.add_argument("--place", required=True, help="City or region name, e.g. 'Bengaluru, India'")
    parser.add_argument("--out", required=True, help="Output path for the GraphML file")
    args = parser.parse_args()

    build_osm_graph(args.place, args.out)
