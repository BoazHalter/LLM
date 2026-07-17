import pandas as pd
from .feature_library import engineer_features # Import the centralized function

# --- Configuration ---
RAW_DATA_PATH = "RawData/Dataset_NQ_1min_2022_2025.csv"
PROCESSED_DATA_PATH = 'processed_features.csv'

# Load the dataset
df = pd.read_csv(RAW_DATA_PATH, delimiter=',')

# --- Data Cleaning ---
# Print original columns for debugging and clean them up
df.columns = df.columns.str.strip().str.title()

# Explicitly select the columns you need to avoid non-numeric data like ticker symbols.
# Adjust these names if they are different in your CSV file.
required_cols = ['Timestamp Et', 'Open', 'High', 'Low', 'Close']

# --- Data Validation ---
# Check if the required columns exist after cleaning. This prevents errors if the CSV is malformed.
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
    print(f"Available columns are: {df.columns.tolist()}")
    exit(1) # Exit with an error code

df = df[required_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

# Convert the timestamp column to a proper DateTime object
df['DateTime'] = pd.to_datetime(df['Timestamp Et'])
# Sort by DateTime and drop the original timestamp column
df = df.sort_values('DateTime').drop(columns=['Timestamp Et'])

# Set DateTime as Index
df.set_index('DateTime', inplace=True)

# --- Feature Engineering ---
# Call the centralized function from the library to create all features
df = engineer_features(df)

# Save enhanced dataset
df.dropna(inplace=True)
df.to_csv(PROCESSED_DATA_PATH, index=True) # Save with the DateTime index

print(f"Feature engineering complete. Saved as '{PROCESSED_DATA_PATH}'.")