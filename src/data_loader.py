import time
from datetime import datetime, timedelta

import pandas as pd
import requests

NASA_API_KEY = "DEMO_KEY"  # Replace with your key from api.nasa.gov


# ── NASA NeoWs (Near Earth Object Web Service) ────────────────────────────────

def fetch_neows_live(days: int = 7, api_key: str = NASA_API_KEY) -> pd.DataFrame:
    """Fetch NEOs from the last N days via NASA NeoWs API (max 7 days per call)."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=min(days, 7))

    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "api_key": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return _parse_neows(resp.json())


def fetch_neows_historical(months: int = 3, api_key: str = NASA_API_KEY) -> pd.DataFrame:
    """Fetch several months of NEO data in 7-day chunks for ML training."""
    all_frames = []
    end = datetime.today()
    start = end - timedelta(days=months * 30)
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7), end)
        url = "https://api.nasa.gov/neo/rest/v1/feed"
        params = {
            "start_date": cursor.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "api_key": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            df = _parse_neows(resp.json())
            all_frames.append(df)
            time.sleep(0.5)  # be kind to the API
        except Exception as e:
            print(f"  Warning: chunk {cursor.date()} failed — {e}")
        cursor = chunk_end + timedelta(days=1)

    combined = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset="id")
    print(f"Total historical NEOs fetched: {len(combined)}")
    return combined


def _parse_neows(payload: dict) -> pd.DataFrame:
    """Parse the NeoWs JSON response into a flat DataFrame."""
    records = []
    for neos in payload["near_earth_objects"].values():
        for neo in neos:
            ca = neo["close_approach_data"][0] if neo["close_approach_data"] else {}
            diam = neo["estimated_diameter"]["kilometers"]
            records.append({
                "id": neo["id"],
                "name": neo["name"],
                "h": neo["absolute_magnitude_h"],
                "diameter_min_km": diam["estimated_diameter_min"],
                "diameter_max_km": diam["estimated_diameter_max"],
                "is_potentially_hazardous": neo["is_potentially_hazardous_asteroid"],
                "dist": float(ca.get("miss_distance", {}).get("astronomical", 0) or 0),
                "v_rel": float(ca.get("relative_velocity", {}).get("kilometers_per_second", 0) or 0),
                "close_approach_date": ca.get("close_approach_date", ""),
                "orbiting_body": ca.get("orbiting_body", "Earth"),
            })
    return pd.DataFrame(records)


# ── JPL Sentry (Impact Risk) ──────────────────────────────────────────────────

def fetch_sentry_data() -> pd.DataFrame:
    """Fetch potential Earth-impactor list from JPL Sentry API."""
    url = "https://ssd-api.jpl.nasa.gov/sentry.api"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame(payload["data"])
    numeric_cols = ["ip", "ps_max", "ts_max", "diameter", "h", "last_obs_jd"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Total Sentry impact-risk objects: {len(df)}")
    return df


# ── JPL CAD API (large historical dataset fallback) ───────────────────────────

def fetch_cad_data(date_min: str = "2015-01-01", date_max: str = "2024-01-01") -> pd.DataFrame:
    """Fetch large historical close-approach dataset from JPL CAD API."""
    url = "https://ssd-api.jpl.nasa.gov/cad.api"
    params = {"date-min": date_min, "date-max": date_max, "dist-max": "0.2", "fullname": True}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    print(f"Total CAD records fetched: {len(df)}")
    return df
