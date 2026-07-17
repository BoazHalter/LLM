import pandas as pd
import numpy as np

# --- Configuration ---
RAW_DATA_PATH = r"C:\Users\bh_ya\Documents\LLM\RawData\Dataset_NQ_1min_2022_2025.csv"
PROCESSED_DATA_PATH = 'processed_features.csv'

# Technical Indicator Parameters
SMA_WINDOW = 20
EMA_WINDOW = 20
BB_WINDOW = 20
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Load the dataset
df = pd.read_csv(RAW_DATA_PATH, delimiter=',')

# --- Data Cleaning ---
# Print original columns for debugging and clean them up
print("Original columns from CSV:", df.columns.tolist())
df.columns = df.columns.str.strip().str.title()
print("Cleaned columns:", df.columns.tolist())

# Explicitly select the columns you need to avoid non-numeric data like ticker symbols.
# Adjust these names if they are different in your CSV file.
required_cols = ['Timestamp Et', 'Open', 'High', 'Low', 'Close']
df = df[required_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

# Convert the timestamp column to a proper DateTime object
df['DateTime'] = pd.to_datetime(df['Timestamp Et'])
# Sort by DateTime and drop the original timestamp column
df = df.sort_values('DateTime').drop(columns=['Timestamp Et'])

# Set DateTime as Index
df.set_index('DateTime', inplace=True)

# Moving Averages
df[f'SMA_{SMA_WINDOW}'] = df['Close'].rolling(window=SMA_WINDOW).mean()  # Simple Moving Average (SMA)
df[f'EMA_{EMA_WINDOW}'] = df['Close'].ewm(span=EMA_WINDOW, adjust=False).mean()  # Exponential Moving Average (EMA)

# Bollinger Bands (20-period SMA ± 2 std dev)
rolling_std = df['Close'].rolling(window=BB_WINDOW).std()
df['BB_Upper'] = df[f'SMA_{SMA_WINDOW}'] + (rolling_std * 2)
df['BB_Lower'] = df[f'SMA_{SMA_WINDOW}'] - (rolling_std * 2)

# RSI Calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Add a small epsilon to prevent division by zero
    rs = gain / (loss + 1e-10) 
    return 100 - (100 / (1 + rs))

df[f'RSI_{RSI_WINDOW}'] = compute_rsi(df['Close'], period=RSI_WINDOW)

# MACD Calculation
ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
df['MACD'] = ema_fast - ema_slow
df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()

# Stochastic Oscillator Calculation
def calculate_stochastic(df, n, d_period, k_smooth=3):
    """
    Calculates the Slow Stochastic Oscillator (%K and %D).
    n: The lookback period for high/low.
    d_period: The period for the %D line (SMA of %K).
    k_smooth: The smoothing period for the %K line.
    """
    low_n = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()
    
    # Raw %K
    raw_k = 100 * (df['Close'] - low_n) / (high_n - low_n)
    
    # Slow %K (smoothed raw %K)
    slow_k = raw_k.rolling(window=k_smooth).mean()
    # Slow %D (smoothed slow %K)
    slow_d = slow_k.rolling(window=d_period).mean()
    return slow_k, slow_d

df['Stoch_K_9_3'], df['Stoch_D_9_3'] = calculate_stochastic(df, 9, 3)
df['Stoch_K_14_3'], df['Stoch_D_14_3'] = calculate_stochastic(df, 14, 3)
df['Stoch_K_44_4'], df['Stoch_D_44_4'] = calculate_stochastic(df, 44, 4)
df['Stoch_K_60_10'], df['Stoch_D_60_10'] = calculate_stochastic(df, 60, 10)

# --- Advanced Feature Engineering from Stochastics ---
def add_stochastic_signals(df, k_col, d_col):
    # Crossover Signal: 1 for bullish crossover, -1 for bearish
    crossover_signal = np.sign(df[k_col] - df[d_col])
    df[f'{k_col}_Crossover'] = crossover_signal.diff().apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))

    # Level Signal: 1 for overbought (>80), -1 for oversold (<20)
    df[f'{k_col}_Level'] = np.where(df[k_col] > 80, 1, np.where(df[k_col] < 20, -1, 0))

    # Momentum/Slope Signal: The rate of change of the %K line
    df[f'{k_col}_Momentum'] = df[k_col].diff()

stochastic_configs = [
    ('Stoch_K_9_3', 'Stoch_D_9_3'),
    ('Stoch_K_14_3', 'Stoch_D_14_3'),
    ('Stoch_K_44_4', 'Stoch_D_44_4'),
    ('Stoch_K_60_10', 'Stoch_D_60_10')
]

for k_col, d_col in stochastic_configs:
    add_stochastic_signals(df, k_col, d_col)

# Save enhanced dataset
df.dropna(inplace=True)
df.to_csv(PROCESSED_DATA_PATH, index=True) # Save with the DateTime index

print(f"Feature engineering complete. Saved as '{PROCESSED_DATA_PATH}'.")