import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = 'chatgpt_rf_model.joblib'
SCALER_PATH = 'scaler.gz'
RAW_DATA_PATH = r"C:\Users\bh_ya\Documents\LLM\RawData\Dataset_NQ_1min_2022_2025.csv"

# The number of recent rows to load for feature calculation.
# This must be larger than the longest lookback window (e.g., 60 for Stoch_K_60_10).
DATA_LOOKBACK = 200

# --- Feature Engineering Functions (must match training script) ---

# Technical Indicator Parameters from training
SMA_WINDOW = 20
EMA_WINDOW = 20
BB_WINDOW = 20
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_stochastic(df, n, d_period, k_smooth=3):
    low_n = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()
    raw_k = 100 * (df['Close'] - low_n) / (high_n - low_n)
    slow_k = raw_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()
    return slow_k, slow_d

def add_stochastic_signals(df, k_col, d_col):
    crossover_signal = np.sign(df[k_col] - df[d_col])
    df[f'{k_col}_Crossover'] = crossover_signal.diff().apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))
    df[f'{k_col}_Level'] = np.where(df[k_col] > 80, 1, np.where(df[k_col] < 20, -1, 0))
    df[f'{k_col}_Momentum'] = df[k_col].diff()

def engineer_features(df):
    """Applies all feature engineering steps to the dataframe."""
    df[f'SMA_{SMA_WINDOW}'] = df['Close'].rolling(window=SMA_WINDOW).mean()
    df[f'EMA_{EMA_WINDOW}'] = df['Close'].ewm(span=EMA_WINDOW, adjust=False).mean()
    rolling_std = df['Close'].rolling(window=BB_WINDOW).std()
    df['BB_Upper'] = df[f'SMA_{SMA_WINDOW}'] + (rolling_std * 2)
    df['BB_Lower'] = df[f'SMA_{SMA_WINDOW}'] - (rolling_std * 2)
    df[f'RSI_{RSI_WINDOW}'] = compute_rsi(df['Close'], period=RSI_WINDOW)
    ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()

    stochastic_configs = [(9, 3), (14, 3), (44, 4), (60, 10)]
    for n, d in stochastic_configs:
        k_col_name = f'Stoch_K_{n}_{d}'
        d_col_name = f'Stoch_D_{n}_{d}'
        if n == 44: # Special case from your script
            df[k_col_name], df[d_col_name] = calculate_stochastic(df, n, d, k_smooth=4)
        else:
            df[k_col_name], df[d_col_name] = calculate_stochastic(df, n, d)
        add_stochastic_signals(df, k_col_name, d_col_name)

    return df

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