import pandas as pd
import joblib
from chatgpt.feature_library import engineer_features # Import the centralized function

# --- Configuration ---
MODEL_PATH = 'chatgpt_rf_model.joblib'
SCALER_PATH = 'scaler.gz'
RAW_DATA_PATH = "RawData/Dataset_NQ_1min_2022_2025.csv"

# The number of recent rows to load for feature calculation.
# This must be larger than the longest lookback window (e.g., 60 for Stoch_K_60_10).
DATA_LOOKBACK = 200

# --- Prediction Logic ---

# 1. Load the trained model and scaler
print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 2. Load the most recent data
print(f"Loading last {DATA_LOOKBACK} rows from raw data...")
df_raw = pd.read_csv(RAW_DATA_PATH).tail(DATA_LOOKBACK)

# 3. Clean and prepare the data (must match training)
df_raw.columns = df_raw.columns.str.strip().str.title()
df_raw['DateTime'] = pd.to_datetime(df_raw['Timestamp Et'])
df_raw = df_raw.set_index('DateTime').sort_index()

# 4. Engineer features
print("Engineering features for new data...")
df_features = engineer_features(df_raw[['Open', 'High', 'Low', 'Close']].copy())

# 5. Get the last valid row for prediction
last_row = df_features.dropna().iloc[-1]
current_price = last_row['Close']
print(f"\nMaking prediction based on data from: {last_row.name} (Current Close: {current_price:.2f})")

# 6. Select the same features used in training
features_for_model = [
    'Close', 'SMA_20', 'EMA_20', 'BB_Upper', 'BB_Lower', 'RSI_14', 'MACD', 'MACD_Signal',
    'Stoch_K_9_3', 'Stoch_D_9_3', 'Stoch_K_14_3', 'Stoch_D_14_3', 'Stoch_K_44_4', 'Stoch_D_44_4',
    'Stoch_K_60_10', 'Stoch_D_60_10', 'Stoch_K_9_3_Crossover', 'Stoch_K_9_3_Level',
    'Stoch_K_14_3_Crossover', 'Stoch_K_14_3_Level', 'Stoch_K_44_4_Crossover', 'Stoch_K_44_4_Level',
    'Stoch_K_60_10_Crossover', 'Stoch_K_60_10_Level', 'Stoch_K_9_3_Momentum', 'Stoch_K_14_3_Momentum',
    'Stoch_K_44_4_Momentum', 'Stoch_K_60_10_Momentum'
]
# Convert the single row (a Series) to a DataFrame to preserve feature names for the scaler
X_pred_df = last_row[features_for_model].to_frame().T

# 7. Scale the features
X_pred_scaled = scaler.transform(X_pred_df)

# 8. Make the prediction (predicts the return)
predicted_return = model.predict(X_pred_scaled)[0]

# 9. Convert return to price
predicted_price = current_price * (1 + predicted_return)

# --- Display Result ---
print("\n--- Prediction Result ---")
print(f"Predicted Return for next minute: {predicted_return:+.4%}")
print(f"Predicted Close Price for next minute: {predicted_price:.2f}")

if predicted_price > current_price:
    print("Signal: Price is predicted to INCREASE.")
else:
    print("Signal: Price is predicted to DECREASE.")