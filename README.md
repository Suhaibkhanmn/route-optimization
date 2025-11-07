# Route Optimization with Live Traffic ETA Prediction

Predicts real-time travel times across Bengaluru using a Graph Neural Network (GNN) model. Combines historical road network data with live traffic from TomTom API to provide dynamic route optimization and ETA predictions.

## Features

- Graph-based route modeling using OpenStreetMap
- GNN model for edge-level travel time prediction
- Live traffic integration via TomTom API
- Interactive Streamlit UI with route visualization

## Quick Start

### Prerequisites

- Python 3.8+
- TomTom API key ([get one here](https://developer.tomtom.com/))

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/route-optimization.git
cd route-optimization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your TomTom API key
echo "TOMTOM_API_KEY=your_key_here" > .env
```

### Usage

1. **Extract road network midpoints:**
   ```bash
   python -m src.data_prep.extract_edge_points --graph data/processed_graph/blr_enriched.graphml --out data/speeds/blr_edge_points.parquet
   ```

2. **Fetch live traffic data:**
   ```bash
   python -m src.data_prep.fetch_live_speeds --points data/speeds/blr_edge_points.parquet --out data/speeds/blr_live.parquet
   ```

3. **Train the model:**
   ```bash
   python -m src.models.train_subset --data data/datasets/blr_edges.pt --out_dir models_subset --subset_size 100000 --epochs 10
   ```

4. **Run the UI:**
   ```bash
   streamlit run src/ui/app.py
   ```

## Project Structure

```
route-optimization/
├── data/
│   ├── datasets/           # Preprocessed datasets
│   ├── processed_graph/    # OSM road network
│   └── speeds/             # Live traffic data
├── src/
│   ├── data_prep/         # Data processing scripts
│   ├── models/            # GNN model & training
│   ├── routing/            # Route evaluation
│   └── ui/                 # Streamlit frontend
└── models_subset/          # Trained model weights
```

## How It Works

1. **Graph Modeling**: Road network is represented as a graph with edges (roads) and nodes (intersections)
2. **Feature Extraction**: Each edge gets attributes (length, speed limit, lanes, etc.)
3. **GNN Training**: Model learns to predict travel times from edge features
4. **Live Integration**: TomTom API provides real-time traffic speeds
5. **Route Prediction**: Combines model predictions with live data for accurate ETAs

## Tech Stack

- **PyTorch Geometric** - Graph neural networks
- **OSMnx** - OpenStreetMap data processing
- **Streamlit** - Web UI
- **TomTom API** - Live traffic data

## Limitations

- Trained on subset of data (100k edges) for faster iteration
- Live traffic covers ~4.5k edges due to API rate limits
- ETA scaling factor is approximate (not production-ready)

## Future Work

- Train on full dataset
- Integrate additional APIs (Google Maps, HERE)
- Add calibration regression for better accuracy
- Deploy as hosted web app

## License

MIT

## Author

Suhaib Khan
