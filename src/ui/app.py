import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import numpy as np
import pandas as pd
import torch
import streamlit as st
import folium
import osmnx as ox
from shapely.geometry import LineString
from src.models.edge_time_gnn import EdgeTimeGNN

# Streamlit setup
st.set_page_config(page_title="Route Optimization", layout="wide")
st.title("Route Optimization — Bengaluru")
st.caption("Shortest-path routing with GNN-based ETA vs baseline.")

# Constants
BLR_CENTER = (12.9716, 77.5946)
GRAPH_PATH = "data/processed_graph/blr_enriched.graphml"
MODEL_PATH = "models_subset/edge_time_sage_subset.pt"
LIVE_SPEEDS_PATH = "data/speeds/blr_live.parquet"


@st.cache_resource(show_spinner=False)
def load_graph():
    if not os.path.exists(GRAPH_PATH):
        st.error(f"Graph file not found: {GRAPH_PATH}. Please ensure data files are available.")
        st.stop()
    G = ox.load_graphml(GRAPH_PATH)
    return G


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Please ensure model files are available.")
        st.stop()
    model = EdgeTimeGNN(in_channels=25, out_channels=1)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_live_speeds():
    try:
        if not os.path.exists(LIVE_SPEEDS_PATH):
            st.warning("Live traffic data not available - using default values. For full features, ensure data files are present.")
            return pd.DataFrame(columns=["lat", "lon", "currentSpeed", "freeFlowSpeed", "confidence", "jamFactor"])
        df = pd.read_parquet(LIVE_SPEEDS_PATH)
        # Make sure we have all the columns we need, add missing ones as NaN
        for col in ["currentSpeed", "freeFlowSpeed", "confidence", "jamFactor"]:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return pd.DataFrame(columns=["lat", "lon", "currentSpeed", "freeFlowSpeed", "confidence", "jamFactor"])


def color_for_traffic(current_speed, free_flow_speed, confidence, jam_factor):
    try:
        if pd.notna(jam_factor):
            if jam_factor >= 7.0:
                return "red"
            if jam_factor >= 4.0:
                return "orange"
            return "green"
        # If jam factor isn't available, use speed ratio instead
        if pd.isna(current_speed) and pd.isna(free_flow_speed):
            return "orange" if (confidence is not None and confidence < 0.5) else "green"
        ff = free_flow_speed if pd.notna(free_flow_speed) else current_speed
        cur = current_speed if pd.notna(current_speed) else ff
        ratio = 0 if ff == 0 else (cur / ff)
        if ratio < 0.5:
            return "red"
        if ratio < 0.8:
            return "orange"
        return "green"
    except Exception:
        return "green"


def format_minutes(seconds):
    return round(seconds / 60.0, 1)


def get_base_map():
    # Create a fresh map each time to avoid stale markers
    return folium.Map(location=list(BLR_CENTER), zoom_start=12, tiles="CartoDB positron")


def ensure_uv_columns(edges_gdf):
    if isinstance(edges_gdf.index, pd.MultiIndex) and ("u" not in edges_gdf.columns or "v" not in edges_gdf.columns):
        idx_df = pd.DataFrame(edges_gdf.index.to_list(), columns=["u", "v", "_key"])
        edges_gdf = edges_gdf.reset_index(drop=True)
        edges_gdf = pd.concat([edges_gdf, idx_df[["u", "v"]]], axis=1)
    return edges_gdf


def geocode_to_point(text):
    try:
        # Supports free text or "lat, lon"
        if "," in text and all(p.strip().replace(".", "", 1).replace("-", "", 1).isdigit() for p in text.split(",")):
            lat, lon = [float(p) for p in text.split(",")]
            return lat, lon
        lat, lon = ox.geocode(text + ", Bengaluru")
        return lat, lon
    except Exception:
        return None


def build_edge_features_for_route(edges_gdf_route, live_df):
    # Build feature vectors for each edge in the route, need 25 features total
    df = edges_gdf_route.copy()

    # Extract and clean up numeric columns
    df["length_m"] = df.get("length", pd.Series(index=df.index)).astype(float)
    # Try to get speed from graph, fall back to free flow speed if missing
    df["speed_kph"] = pd.to_numeric(df.get("speed_kph", np.nan), errors="coerce")
    # Convert lanes to numbers (sometimes comes as string)
    lanes_raw = df.get("lanes", np.nan)
    df["lanes"] = pd.to_numeric(lanes_raw, errors="coerce")

    # Use global traffic stats since we don't have per-edge live data
    global_ff = float(np.nanmedian(live_df["freeFlowSpeed"]) if "freeFlowSpeed" in live_df.columns and len(live_df) else 35.0)
    global_cs = float(np.nanmedian(live_df["currentSpeed"]) if "currentSpeed" in live_df.columns and len(live_df) else 28.0)
    global_conf = float(np.nanmedian(live_df["confidence"]) if "confidence" in live_df.columns and len(live_df) else 0.8)

    df["freeFlowSpeed"] = global_ff
    df["currentSpeed"] = global_cs
    df["confidence"] = global_conf

    # Fill in any missing values with reasonable defaults
    df["speed_kph"] = df["speed_kph"].fillna(df["freeFlowSpeed"]).fillna(30.0)
    df["lanes"] = df["lanes"].fillna(2.0)
    df["length_m"] = df["length_m"].fillna(0.0)

    # Pick the core features and pad to exactly 25 dimensions
    base_cols = [
        "length_m", "speed_kph", "lanes", "confidence", "currentSpeed", "freeFlowSpeed"
    ]
    feat_df = df[base_cols].copy()
    # Add some derived features that might help the model
    feat_df["cs_ff_ratio"] = (feat_df["currentSpeed"] / feat_df["freeFlowSpeed"]).replace([np.inf, -np.inf], 0.0)
    feat_df["len_over_speed"] = (feat_df["length_m"] / (feat_df["speed_kph"].replace(0, np.nan) * 1000 / 3600)).fillna(0.0)

    # Make sure we have exactly 25 features (pad or trim if needed)
    while feat_df.shape[1] < 25:
        feat_df[f"pad_{feat_df.shape[1]}"] = 0.0
    if feat_df.shape[1] > 25:
        feat_df = feat_df.iloc[:, :25]

    x_edge = torch.tensor(np.nan_to_num(feat_df.values, nan=0.0).astype(np.float32))
    return x_edge


def baseline_eta_seconds(edges_gdf_route):
    # Calculate baseline ETA by summing up length/speed for each edge
    length_m = pd.to_numeric(edges_gdf_route.get("length", 0.0), errors="coerce").fillna(0.0).values
    speed_kph = pd.to_numeric(edges_gdf_route.get("speed_kph", np.nan), errors="coerce").values
    free_flow = pd.to_numeric(edges_gdf_route.get("freeFlowSpeed", np.nan), errors="coerce").values if "freeFlowSpeed" in edges_gdf_route.columns else np.full(len(length_m), np.nan)
    # Pick the best speed estimate we have for each edge
    chosen_speed = np.where(np.isnan(speed_kph), np.where(np.isnan(free_flow), 30.0, free_flow), speed_kph)
    speed_mps = np.clip(chosen_speed, 5.0, 120.0) * 1000.0 / 3600.0
    secs = np.sum(np.where(speed_mps > 0, length_m / speed_mps, 0.0))
    return float(secs)


# Sidebar UI
st.sidebar.header("Route inputs")
start_text = st.sidebar.text_input("Start", "HSR Layout Sector 1")
end_text = st.sidebar.text_input("Destination", "Electronic City, Bengaluru")
show_traffic = st.sidebar.checkbox("Overlay live traffic points", value=False)
compute = st.sidebar.button("Compute route")

# Show ETA metrics if we computed a route (persists after page rerun)
if "pred_eta" in st.session_state and "base_eta" in st.session_state:
    st.sidebar.metric("Predicted ETA (GNN)", f"{st.session_state.pred_eta:.1f} min")
    st.sidebar.metric("Baseline ETA (Length/Speed)", f"{st.session_state.base_eta:.1f} min")


# Keep map in session state so it doesn't flicker on reruns
if "folium_map" not in st.session_state:
    st.session_state.folium_map = get_base_map()
    st.session_state.layers = []
    st.session_state.traffic_layers = []

G = load_graph()
model = load_model()
live_df = load_live_speeds()


def clear_dynamic_layers(m):
    for layer in st.session_state.layers:
        try:
            m.remove_child(layer)
        except Exception:
            pass
    st.session_state.layers = []


def clear_traffic_layers(m):
    for layer in st.session_state.traffic_layers:
        try:
            m.remove_child(layer)
        except Exception:
            pass
    st.session_state.traffic_layers = []


if compute:
    with st.spinner("Computing route..."):
        try:
            # Convert location names to coordinates and find closest graph nodes
            start_pt = geocode_to_point(start_text)
            end_pt = geocode_to_point(end_text)
            if not start_pt or not end_pt:
                raise ValueError("Could not geocode inputs")

            s_lat, s_lon = start_pt
            e_lat, e_lon = end_pt
            s_node = ox.distance.nearest_nodes(G, s_lon, s_lat)
            e_node = ox.distance.nearest_nodes(G, e_lon, e_lat)

            # Find shortest path through the road network
            route_nodes = ox.shortest_path(G, s_node, e_node, weight="length")
            if not route_nodes or len(route_nodes) < 2:
                raise ValueError("No path found")

            edges_gdf = ensure_uv_columns(ox.graph_to_gdfs(G, nodes=False))
            uv_pairs = list(zip(route_nodes[:-1], route_nodes[1:]))
            route_edges = []
            for (u, v) in uv_pairs:
                match = edges_gdf[(edges_gdf["u"] == u) & (edges_gdf["v"] == v)]
                if match.empty:
                    match = edges_gdf[(edges_gdf["u"] == v) & (edges_gdf["v"] == u)]
                if not match.empty:
                    route_edges.append(match.iloc[0])
            if not route_edges:
                raise ValueError("Could not map route edges")

            edges_route_gdf = pd.DataFrame(route_edges)

            x_edge = build_edge_features_for_route(edges_route_gdf, live_df)
            with torch.no_grad():
                pred_times = model(x_node=None, edge_index=None, x_edge=x_edge).squeeze(-1)

            pred_minutes = float(torch.clamp(pred_times, min=0).sum().item()) / 60.0
            base_minutes = baseline_eta_seconds(edges_route_gdf) / 60.0

            pred_minutes *= 2.3  # Scale to match real-world times better

            # Make sure ETAs are at least 1 minute (avoid weird zero values)
            pred_minutes = max(1.0, pred_minutes)
            base_minutes = max(1.0, base_minutes)

            # Save ETAs so they show up even after the page reruns
            st.session_state.pred_eta = pred_minutes
            st.session_state.base_eta = base_minutes

            # Show the predicted times in the sidebar
            st.sidebar.metric("Predicted ETA (GNN)", f"{pred_minutes:.1f} min")
            st.sidebar.metric("Baseline ETA (Length/Speed)", f"{base_minutes:.1f} min")

            # Start fresh - wipe out any old routes, markers, and traffic overlays
            m = folium.Map(location=list(BLR_CENTER), zoom_start=12, tiles="CartoDB positron")
            st.session_state.folium_map = m
            st.session_state.layers = []
            st.session_state.traffic_layers = []

            # Add start and end markers only
            start_marker = folium.Marker(location=[s_lat, s_lon], tooltip=f"Start: {start_text}")
            end_marker = folium.Marker(location=[e_lat, e_lon], tooltip=f"End: {end_text}")
            start_marker.add_to(m)
            end_marker.add_to(m)
            st.session_state.layers = [start_marker, end_marker]

            # Draw the route path on the map, color coded by traffic
            for row in edges_route_gdf.itertuples(index=False):
                geom = getattr(row, "geometry", None)
                coords = [(lat, lon) for lon, lat in geom.coords] if isinstance(geom, LineString) else []
                color = color_for_traffic(
                    getattr(row, "currentSpeed", np.nan),
                    getattr(row, "freeFlowSpeed", np.nan),
                    getattr(row, "confidence", np.nan),
                    getattr(row, "jamFactor", np.nan),
                )
                folium.PolyLine(locations=coords, color=color, weight=5, opacity=0.9).add_to(m)

            # Show live traffic points if user enabled the overlay
            if show_traffic and not live_df.empty:
                sample_df = live_df.sample(min(len(live_df), 1500), random_state=42) if len(live_df) > 1500 else live_df
                for r in sample_df.itertuples(index=False):
                    lat = getattr(r, "lat", None)
                    lon = getattr(r, "lon", None)
                    if lat is None or lon is None:
                        continue
                    col = color_for_traffic(
                        getattr(r, "currentSpeed", np.nan),
                        getattr(r, "freeFlowSpeed", np.nan),
                        getattr(r, "confidence", np.nan),
                        getattr(r, "jamFactor", np.nan),
                    )
                    circ = folium.CircleMarker(
                        location=[lat, lon], radius=2, color=col, fill=True, fill_opacity=0.7, opacity=0.7
                    )
                    circ.add_to(m)
                    st.session_state.traffic_layers.append(circ)

            st.rerun()

        except Exception as e:
            st.error(f"Could not compute route: {e}")


# Display the map (keep it cached to avoid flickering)
m = st.session_state.folium_map
# Clean up traffic overlay if user turned it off
if not show_traffic and "traffic_layers" in st.session_state and st.session_state.traffic_layers:
    clear_traffic_layers(m)
st.components.v1.html(m._repr_html_(), height=600)


col1, col2 = st.columns(2)
with col1:
    if st.button("Recenter Bengaluru"):
        # Reset map view but keep any routes/markers we already drew
        base = get_base_map()
        # Copy existing markers and routes to the new map
        for layer in st.session_state.layers:
            try:
                layer.add_to(base)
            except Exception:
                pass
        st.session_state.folium_map = base
        st.rerun()
with col2:
    if st.button("Clear overlays"):
        clear_dynamic_layers(st.session_state.folium_map)
        st.rerun()
