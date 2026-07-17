import pandas as pd
import numpy as np

def calculate_stochastic_oscillator(df, k_period, d_period, column_suffix):
    """Calculates the Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df[f'Stoch_K_{column_suffix}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df[f'Stoch_D_{column_suffix}'] = df[f'Stoch_K_{column_suffix}'].rolling(window=d_period).mean()

def add_stochastic_signals(df, k_period, d_period, column_suffix):
    """Adds crossover, level, and momentum signals for Stochastic Oscillator."""
    stoch_k_col = f'Stoch_K_{column_suffix}'
    stoch_d_col = f'Stoch_D_{column_suffix}'

    # Crossover Signal
    df[f'{stoch_k_col}_Crossover'] = np.where(
        (df[stoch_k_col].shift(1) < df[stoch_d_col].shift(1)) & (df[stoch_k_col] > df[stoch_d_col]), 1,
        np.where((df[stoch_k_col].shift(1) > df[stoch_d_col].shift(1)) & (df[stoch_k_col] < df[stoch_d_col]), -1, 0)
    )

    # Level Signal (Overbought/Oversold)
    df[f'{stoch_k_col}_Level'] = np.where(df[stoch_k_col] > 80, -1, np.where(df[stoch_k_col] < 20, 1, 0))

    # Momentum Signal
    df[f'{stoch_k_col}_Momentum'] = df[stoch_k_col].diff()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the given financial time-series data.

    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close', and 'Volume' columns.

    Returns:
        DataFrame with added feature columns.
    """
    # Use a dictionary to build features for performance and memory efficiency
    features = {}
    
    # --- Basic Features ---
    features['PriceRange'] = df['High'] - df['Low']
    features['Return'] = df['Close'].pct_change()

    # --- Moving Averages ---
    features['SMA_10'] = df['Close'].rolling(window=10).mean()
    features['SMA_20'] = df['Close'].rolling(window=20).mean()
    features['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # --- Bollinger Bands ---
    bb_window = 20
    bb_middle = df['Close'].rolling(window=bb_window).mean()
    std_dev = df['Close'].rolling(window=bb_window).std()
    features['BB_Upper'] = bb_middle + (std_dev * 2)
    features['BB_Lower'] = bb_middle - (std_dev * 2)

    # --- Relative Strength Index (RSI) ---
    rsi_period = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    features['RSI_14'] = 100 - (100 / (1 + rs))

    # --- Moving Average Convergence Divergence (MACD) ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()

    # Assign all calculated features to the original DataFrame in one go
    df_with_features = df.assign(**features)

    # --- Stochastic Oscillators ---
    stoch_params = [
        (9, 3, '9_3'),
        (14, 3, '14_3'),
        (44, 4, '44_4'),
        (60, 10, '60_10')
    ]
    
    for k, d, suffix in stoch_params:
        # These functions modify the DataFrame in-place, so we pass the new one
        calculate_stochastic_oscillator(df_with_features, k, d, suffix)
        add_stochastic_signals(df_with_features, k, d, suffix)

    # --- Final Cleanup ---
    expected_features = [
        'Open', 'High', 'Low', 'Close', 'PriceRange', 'Return',
        'SMA_10', 'SMA_20', 'EMA_20', 'BB_Upper', 'BB_Lower',
        'RSI_14', 'MACD', 'MACD_Signal',
        'Stoch_K_9_3', 'Stoch_D_9_3',
        'Stoch_K_14_3', 'Stoch_D_14_3',
        'Stoch_K_44_4', 'Stoch_D_44_4',
        'Stoch_K_60_10', 'Stoch_D_60_10',
        'Stoch_K_9_3_Crossover', 'Stoch_K_9_3_Level', 'Stoch_K_9_3_Momentum',
        'Stoch_K_14_3_Crossover', 'Stoch_K_14_3_Level', 'Stoch_K_14_3_Momentum',
        'Stoch_K_44_4_Crossover', 'Stoch_K_44_4_Level', 'Stoch_K_44_4_Momentum',
        'Stoch_K_60_10_Crossover', 'Stoch_K_60_10_Level', 'Stoch_K_60_10_Momentum'
    ]
    
    for col in expected_features:
        if col not in df_with_features.columns:
            df_with_features[col] = np.nan
            
    final_cols = [col for col in expected_features if col in df_with_features.columns]

    return df_with_features[final_cols]