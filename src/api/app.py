from fastapi import FastAPI
from pydantic import BaseModel
import torch
import osmnx as ox
import networkx as nx
from src.models.edge_time_gnn import EdgeTimeGNN

app = FastAPI(title="Route Optimization API", version="2.0")

# Load everything we need when the API starts up
GRAPH_PATH = "data/processed_graph/blr_enriched.graphml"
DATA_PATH = "data/datasets/blr_edges.pt"
MODEL_PATH = "models/edge_time_sage_v1.pt"

print("Loading graph and model...")
G = ox.load_graphml(GRAPH_PATH)
data = torch.load(DATA_PATH, map_location="cpu")

# Load the trained GNN model
in_node = 64
in_edge = data["x_edge"].shape[1]
model = EdgeTimeGNN(in_node, in_edge, hidden_channels=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
model.eval()

# Get the data ready for making predictions
x_node = data["x_node"]
if x_node.shape[1] == 1:  # Expand to 64 dimensions if it's only 1D
    x_node = x_node.repeat(1, in_node)

edge_index = data["edge_index"]
x_edge = data["x_edge"]

# Predict travel times for all edges when the server starts
with torch.no_grad():
    preds = model(x_node, edge_index, x_edge).squeeze().cpu().numpy()

for i, (u, v, k) in enumerate(G.edges(keys=True)):
    G[u][v][k]["pred_time"] = float(preds[i])

print(f"Loaded {len(preds)} predicted edge times into the graph.")

# Define what the API expects in requests
class RouteRequest(BaseModel):
    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float


# The main API endpoint for getting routes
@app.post("/route")
def get_route(req: RouteRequest):
    src = ox.distance.nearest_nodes(G, req.source_lon, req.source_lat)
    dst = ox.distance.nearest_nodes(G, req.dest_lon, req.dest_lat)

    try:
        base_path = nx.shortest_path(G, src, dst, weight="travel_time")
        ml_path = nx.shortest_path(G, src, dst, weight="pred_time")

        base_time = sum(G[u][v][0]["travel_time"] for u, v in zip(base_path[:-1], base_path[1:]))
        ml_time = sum(G[u][v][0]["pred_time"] for u, v in zip(ml_path[:-1], ml_path[1:]))

        return {
            "source": (req.source_lat, req.source_lon),
            "destination": (req.dest_lat, req.dest_lon),
            "baseline_eta_sec": round(base_time, 2),
            "ml_eta_sec": round(ml_time, 2),
            "improvement_%": round((1 - ml_time / base_time) * 100, 2),
            "baseline_nodes": base_path,
            "ml_nodes": ml_path,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Route Optimization API with GNN inference is live."}
