import osmnx as ox
import networkx as nx
import folium
import torch
from src.models.edge_time_gnn import EdgeTimeGNN

def visualize_routes(graph_path, dataset_path, model_path, src=None, dst=None):
    G = ox.load_graphml(graph_path)
    data = torch.load(dataset_path, map_location="cpu")
    model = EdgeTimeGNN(64, data["x_edge"].shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    nodes = list(G.nodes)
    if src is None or dst is None:
        import random
        src, dst = random.sample(nodes, 2)

    # baseline
    base_path = nx.shortest_path(G, src, dst, weight="travel_time")
    # ML
    for i, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]["pred_time"] = float(G[u][v][k].get("travel_time", 1.0) * 0.6)  # demo weighting
    ml_path = nx.shortest_path(G, src, dst, weight="pred_time")

    m = folium.Map(location=ox.graph_to_gdfs(G, nodes=True).geometry.y.mean(),
                   zoom_start=13)
    ox.plot_route_folium(G, base_path, route_map=m, color="red", weight=4, opacity=0.8)
    ox.plot_route_folium(G, ml_path, route_map=m, color="green", weight=4, opacity=0.8)
    m.save("data/route_comparison.html")
    print("Map saved to data/route_comparison.html")
