import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = 'chatgpt_rf_model.joblib'
SCALER_PATH = 'scaler.gz'
PROCESSED_DATA_PATH = 'processed_features.csv'

# --- Strategy Parameters ---
# The minimum predicted return required to enter a trade.
# This helps filter out low-confidence predictions.
TRADE_THRESHOLD = 0.00005  # e.g., 0.005%

# --- Backtesting Logic ---

print("--- Starting Backtest ---")

# 1. Load model, scaler, and data
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(PROCESSED_DATA_PATH, index_col='DateTime', parse_dates=True)

# 2. Define the feature set (must match training)
features = [
    'Close', 'SMA_20', 'EMA_20', 'BB_Upper', 'BB_Lower', 'RSI_14', 'MACD', 'MACD_Signal',
    'Stoch_K_9_3', 'Stoch_D_9_3', 'Stoch_K_14_3', 'Stoch_D_14_3', 'Stoch_K_44_4', 'Stoch_D_44_4',
    'Stoch_K_60_10', 'Stoch_D_60_10', 'Stoch_K_9_3_Crossover', 'Stoch_K_9_3_Level',
    'Stoch_K_14_3_Crossover', 'Stoch_K_14_3_Level', 'Stoch_K_44_4_Crossover', 'Stoch_K_44_4_Level',
    'Stoch_K_60_10_Crossover', 'Stoch_K_60_10_Level', 'Stoch_K_9_3_Momentum', 'Stoch_K_14_3_Momentum',
    'Stoch_K_44_4_Momentum', 'Stoch_K_60_10_Momentum'
]

# 3. Prepare data for backtesting
df['Target'] = df['Close'].pct_change().shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df['Target']

# 4. Isolate the test set (same split as in training)
train_size = int(len(X) * 0.8)
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

print(f"Backtesting on {len(X_test)} data points (test set).")

# 5. Scale the test features
X_test_scaled = scaler.transform(X_test)

# 6. Get model predictions for the entire test set
predicted_returns = model.predict(X_test_scaled)

# 7. Simulate the trading strategy
positions = []  # 1 for long, -1 for short, 0 for flat
for pred_return in predicted_returns:
    if pred_return > TRADE_THRESHOLD:
        positions.append(1)  # Go Long
    elif pred_return < -TRADE_THRESHOLD:
        positions.append(-1) # Go Short
    else:
        positions.append(0)  # Stay Flat

# Calculate the strategy returns
# We assume we enter based on the prediction and hold for one period.
# The return is the actual return of that period, multiplied by our position (long or short).
# We shift positions by 1 because we make the decision at time t to capture the return at t+1.
strategy_returns = y_test * pd.Series(positions, index=y_test.index).shift(1)
strategy_returns = strategy_returns.fillna(0)

# --- Performance Evaluation ---

# Calculate cumulative returns for our strategy and for buy-and-hold
cumulative_strategy_returns = (1 + strategy_returns).cumprod()
buy_and_hold_returns = (1 + y_test).cumprod()

# Calculate performance metrics
total_trades = (pd.Series(positions).diff() != 0).sum()
win_rate = (strategy_returns > 0).sum() / total_trades if total_trades > 0 else 0

print("\n--- Backtest Results ---")
print(f"Total Trades Executed: {total_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Final Strategy Return: {(cumulative_strategy_returns.iloc[-1] - 1):.2%}")
print(f"Final Buy-and-Hold Return: {(buy_and_hold_returns.iloc[-1] - 1):.2%}")

# Calculate max drawdown
running_max = cumulative_strategy_returns.cummax()
drawdown = (cumulative_strategy_returns - running_max) / running_max
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown:.2%}")

# --- Visualization ---
plt.figure(figsize=(15, 7))
plt.plot(cumulative_strategy_returns, label='Model Strategy')
plt.plot(buy_and_hold_returns, label='Buy and Hold')
plt.title('Strategy Performance vs. Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.savefig('backtest_equity_curve.png') # Save the plot to a file
print("Saved backtest equity curve plot to 'backtest_equity_curve.png'")