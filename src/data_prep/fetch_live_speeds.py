import os
import time
import math
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

if not TOMTOM_API_KEY:
    raise RuntimeError("TOMTOM_API_KEY not found. Please add it to your .env file or environment.")

# ======= CONFIG =======
FREE_TIER_LIMIT = 4500  # TomTom free tier monthly limit
BATCH_SIZE = 100
MAX_WORKERS = 8
RETRY_SLEEP = 2
MAX_RETRIES = 3
# ======================

def fetch_batch(batch_points, retry=MAX_RETRIES, sleep=RETRY_SLEEP):
    """Fetch one batch of live speeds from TomTom API."""
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    results = []
    for p in batch_points:
        lat, lon = p["lat"], p["lon"]
        for attempt in range(retry):
            try:
                r = requests.get(url, params={
                    "key": TOMTOM_API_KEY,
                    "point": f"{lat},{lon}"
                }, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    flow = data.get("flowSegmentData", {})
                    results.append({
                        "lat": lat,
                        "lon": lon,
                        "currentSpeed": flow.get("currentSpeed"),
                        "freeFlowSpeed": flow.get("freeFlowSpeed"),
                        "confidence": flow.get("confidence"),
                        "roadClosure": flow.get("roadClosure"),
                        "frc": flow.get("frc"),
                        "ts": time.time()
                    })
                    break
                else:
                    time.sleep(sleep)
            except Exception:
                time.sleep(sleep)
    return results


def fetch_live_speeds(points_path, out_path, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """Fetch traffic data for road midpoints using TomTom API with auto-limit."""
    df = pd.read_parquet(points_path)
    print(f"Loaded {len(df)} total road midpoints")

    # Auto-limit to 4500 points for free-tier to avoid hitting API limits
    if len(df) > FREE_TIER_LIMIT:
        df = df.sample(FREE_TIER_LIMIT, random_state=42).reset_index(drop=True)
        print(f"Limited to {FREE_TIER_LIMIT} points due to TomTom free-tier limit")

    batches = math.ceil(len(df) / batch_size)
    print(f"Fetching {batches} batches of {batch_size} points each using {max_workers} threads")

    all_results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_points = df.iloc[start_idx:end_idx].to_dict("records")
            futures.append(executor.submit(fetch_batch, batch_points))

        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            all_results.extend(res)
            if i % 10 == 0:
                print(f"Completed {i}/{batches} batches ({len(all_results)} points fetched)")

    total_time = time.time() - start
    print(f"Done in {total_time/60:.1f} min, fetched {len(all_results)} records")

    out_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved live traffic data to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="Path to edge midpoints parquet file")
    parser.add_argument("--out", required=True, help="Path to save live traffic parquet file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of points per API call")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS, help="Number of threads to use")
    args = parser.parse_args()

    fetch_live_speeds(args.points, args.out, args.batch_size, args.max_workers)
