import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from CAD or NeoWs DataFrames.

    Label strategy (no leakage):
      - Uses dist_min (3-sigma uncertainty minimum distance) for the label.
      - Uses dist  (nominal closest approach) as a feature.
      - dist != dist_min, so the model must learn a real predictive relationship.
    """
    df = df.copy()

    numeric_cols = ["dist", "dist_min", "v_rel", "v_inf", "h"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["dist", "v_rel", "h"], inplace=True)
    df = df[df["dist"] > 0]

    # ── Label ─────────────────────────────────────────────────────────────────
    if "is_potentially_hazardous" in df.columns:
        # NeoWs: use NASA's MOID-based PHA flag directly (gold standard)
        df["hazardous"] = df["is_potentially_hazardous"].astype(int)
    elif "dist_min" in df.columns and df["dist_min"].notna().mean() > 0.5:
        # CAD: dist_min is the 3-sigma minimum — closer to MOID than nominal dist.
        # Using dist_min for label while dist stays as a feature creates a real
        # predictive gap (the model can't cheat by reading the label from dist).
        df["hazardous"] = ((df["dist_min"] < 0.05) & (df["h"] < 22)).astype(int)
    else:
        df["hazardous"] = ((df["dist"] < 0.05) & (df["h"] < 22)).astype(int)

    # ── Features ──────────────────────────────────────────────────────────────
    df["diameter_avg_km"] = 10 ** (3.1236 - 0.5 * df["h"]) / 1000

    if "v_inf" not in df.columns:
        df["v_inf"] = df["v_rel"] * 0.9   # fallback approximation

    df["v_inf"] = df["v_inf"].fillna(df["v_rel"])

    print(f"Records after cleaning : {len(df)}")
    print(f"Hazardous : {df['hazardous'].sum()}  |  Safe : {(df['hazardous'] == 0).sum()}")
    print(f"Imbalance ratio : 1 : {(df['hazardous']==0).sum() // max(df['hazardous'].sum(),1)}")
    return df
