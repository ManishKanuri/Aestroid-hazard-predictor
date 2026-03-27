from src.data_loader import fetch_neows_historical
from src.model import train
from src.preprocessing import preprocess


def main():
    print("=== Space Intelligence Platform ===\n")

    print("[1/3] Fetching 3 months of NEO data from NASA NeoWs API...")
    df_raw = fetch_neows_historical(months=3)

    print("\n[2/3] Preprocessing...")
    df_clean = preprocess(df_raw)

    print("\n[3/3] Training GradientBoosting model...")
    train(df_clean)

    print("\nDone. Run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
