import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw asteroid DataFrame."""
    df = df.copy()

    for col in ["dist", "v_rel", "h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["dist", "v_rel", "h"], inplace=True)

    df["hazardous"] = ((df["dist"] < 0.05) & (df["h"] < 22)).astype(int)

    print(f"Records after cleaning: {len(df)}")
    print(f"Hazardous count: {df['hazardous'].sum()}")
    return df
