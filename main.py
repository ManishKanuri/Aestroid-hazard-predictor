from src.data_loader import fetch_asteroid_data
from src.model import train
from src.preprocessing import preprocess


def main():
    print("=== Space Intelligence Platform ===\n")

    print("[1/3] Fetching asteroid data from NASA JPL API...")
    df_raw = fetch_asteroid_data()

    print("\n[2/3] Preprocessing data...")
    df_clean = preprocess(df_raw)

    print("\n[3/3] Training model...")
    train(df_clean)

    print("\nDone.")


if __name__ == "__main__":
    main()
