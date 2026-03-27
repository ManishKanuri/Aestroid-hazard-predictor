import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features. Works with both NeoWs and CAD DataFrames."""
    df = df.copy()

    numeric = ["dist", "v_rel", "h", "diameter_min_km", "diameter_max_km"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["dist", "v_rel", "h"], inplace=True)
    df = df[df["dist"] > 0]

    # Use NASA's own hazard flag when available (NeoWs), else derive from orbital params (CAD)
    if "is_potentially_hazardous" in df.columns:
        df["hazardous"] = df["is_potentially_hazardous"].astype(int)
    else:
        df["hazardous"] = ((df["dist"] < 0.05) & (df["h"] < 22)).astype(int)

    # Derived features
    if "diameter_min_km" in df.columns and "diameter_max_km" in df.columns:
        df["diameter_avg_km"] = (df["diameter_min_km"] + df["diameter_max_km"]) / 2
    else:
        # Approximate diameter from absolute magnitude (Bowell formula simplified)
        df["diameter_avg_km"] = 10 ** (3.1236 - 0.5 * df["h"]) / 1000

    df["threat_score"] = (1 / df["dist"].clip(lower=1e-6)) * df["v_rel"] / df["h"].clip(lower=1)

    print(f"Records after cleaning: {len(df)}")
    print(f"Hazardous: {df['hazardous'].sum()} | Safe: {(df['hazardous'] == 0).sum()}")
    return df
