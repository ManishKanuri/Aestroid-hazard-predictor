from src.data_loader import fetch_cad_data
from src.model import train
from src.preprocessing import preprocess


def main():
    print("=== Space Intelligence Platform ===\n")

    print("[1/3] Fetching historical data from JPL CAD API...")
    print("      (No API key or rate limits — single call, 50k+ records)\n")
    df_raw = fetch_cad_data(date_min="2015-01-01", date_max="2024-01-01")

    print("\n[2/3] Preprocessing...")
    df_clean = preprocess(df_raw)

    print("\n[3/3] Training model...")
    train(df_clean)

    print("\nDone. Run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
