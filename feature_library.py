import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the given financial time-series data.

    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        DataFrame with added feature columns.
    """
    # Make a copy to avoid modifying the original DataFrame in place
    df_copy = df.copy()

    # --- Add your feature engineering logic here ---

    # Example Feature 1: Price Range
    df_copy['PriceRange'] = df_copy['High'] - df_copy['Low']

    # Example Feature 2: Moving Average
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean()

    # Example Feature 3: Return
    df_copy['Return'] = df_copy['Close'].pct_change()

    # ------------------------------------------------

    return df_copy