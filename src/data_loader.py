import requests
import pandas as pd


def fetch_asteroid_data(date_min="2000-01-01", date_max="2024-01-01", dist_max="0.2"):
    """Fetch asteroid close approach data from NASA JPL API."""
    url = "https://ssd-api.jpl.nasa.gov/cad.api"
    params = {
        "date-min": date_min,
        "date-max": date_max,
        "dist-max": dist_max,
        "fullname": True,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    payload = response.json()

    fields = payload["fields"]
    data = payload["data"]

    df = pd.DataFrame(data, columns=fields)
    print(f"Total records fetched: {len(df)}")
    return df
