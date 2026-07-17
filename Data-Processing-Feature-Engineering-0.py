import pandas as pd
from feature_library import create_features

# --- Configuration ---
RAW_DATA_PATH = "RawData/Dataset_NQ_1min_2022_2025.csv"
PROCESSED_DATA_PATH = 'processed_features.csv'

def main():
    """Main function to run the data processing pipeline."""
    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH, delimiter=',')

    # --- Data Cleaning ---
    df.columns = df.columns.str.strip().str.title()

    required_cols = ['Timestamp Et', 'Open', 'High', 'Low', 'Close']

    # --- Data Validation ---
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
        print(f"Available columns are: {df.columns.tolist()}")
        exit(1)

    df = df[required_cols].copy()

    df['DateTime'] = pd.to_datetime(df['Timestamp Et'])
    df = df.sort_values('DateTime').drop(columns=['Timestamp Et'])
    df.set_index('DateTime', inplace=True)

    # --- Feature Engineering ---
    df = create_features(df)

    # --- Save enhanced dataset ---
    df.dropna(inplace=True)
    df.to_csv(PROCESSED_DATA_PATH, index=True)

    print(f"Feature engineering complete. Saved as '{PROCESSED_DATA_PATH}'.")

if __name__ == "__main__":
    main()