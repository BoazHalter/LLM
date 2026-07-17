import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the processed features dataset
df = pd.read_csv('processed_features.csv', index_col='DateTime', parse_dates=True)

# --- Base Data Information ---
print("--- Base Data Information ---")
print(f"Dataset shape: {df.shape}")
print("\nData Info (columns, non-null counts, data types):")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("-----------------------------\n")

# Select features for training
features = [
    'Close', 
    'SMA_20', 
    'EMA_20', 
    'BB_Upper', 
    'BB_Lower', 
    'RSI_14', 
    'MACD', 
    'MACD_Signal',
    'Stoch_K_9_3', 'Stoch_D_9_3',
    'Stoch_K_14_3', 'Stoch_D_14_3',
    'Stoch_K_44_4', 'Stoch_D_44_4',
    'Stoch_K_60_10', 'Stoch_D_60_10',
    'Stoch_K_9_3_Crossover', 'Stoch_K_9_3_Level',
    'Stoch_K_14_3_Crossover', 'Stoch_K_14_3_Level',
    'Stoch_K_44_4_Crossover', 'Stoch_K_44_4_Level',
    'Stoch_K_60_10_Crossover', 'Stoch_K_60_10_Level',
    'Stoch_K_9_3_Momentum', 'Stoch_K_14_3_Momentum',
    'Stoch_K_44_4_Momentum', 'Stoch_K_60_10_Momentum'
]
df = df[features]

# --- Data Preparation ---

# We want to predict the future return, not the absolute price.
# Return = (Price[t+1] - Price[t]) / Price[t]
# This is calculated using pandas' percentage change function and shifted.
df['Target'] = df['Close'].pct_change().shift(-1)

# Drop rows with NaN values (from the shift and from initial indicator calculations)
df.dropna(inplace=True)

# Separate features (X) from the target (y)
X = df[features]
y = df['Target']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use (e.g., in production)
joblib.dump(scaler, 'scaler.gz')

# Split into training (80%) and testing (20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- Model Training ---
# Define the RandomForestRegressor model. `n_estimators` is the number of trees in the forest.
# We are now using the optimized hyperparameters found by the `tune_model.py` script.
# Replace these values with the actual output from your tuning script if they are different.
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    n_jobs=-1)

# Train the model
print("--- Training RandomForest Model ---")
model.fit(X_train, y_train)
print("--- Training Complete ---")

# --- Model Evaluation and Visualization ---

# 1. Predict Close prices on the test set
predicted_returns = model.predict(X_test)
actual_returns = y_test.values

# To make the plot intuitive, we convert the predicted returns back into predicted prices.
# Predicted Price[t+1] = Price[t] * (1 + Predicted_Return[t+1])
# We get Price[t] from the original unscaled test data.
X_test_unscaled = X.iloc[train_size:]
close_prices_t = X_test_unscaled['Close'].values

predicted_prices = close_prices_t * (1 + predicted_returns)
actual_prices = close_prices_t * (1 + actual_returns)

# 2. Plot predicted vs actual prices
plt.figure(figsize=(15, 7))
plt.plot(actual_prices, label="Actual Close Price", color='blue')
plt.plot(predicted_prices, label="Predicted Close Price", color='red', linestyle='--')
plt.title('Actual vs. Predicted Close Prices')
plt.xlabel('Time Steps (in test set)')
plt.ylabel('Price')
plt.legend()
plt.savefig('actual_vs_predicted_prices.png') # Save the plot to a file
print("Saved price plot to 'actual_vs_predicted_prices.png'")

# 3. Calculate and print regression metrics on the price predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

print("--- Model Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- Feature Importance Analysis (on predicting returns) ---
print("\n--- Calculating Feature Importance ---")

# Get feature importances from the trained model
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png') # Save the plot to a file
print("Saved feature importance plot to 'feature_importance.png'")

# Saving the model
joblib.dump(model, 'chatgpt_rf_model.joblib')
print("Model saved as 'chatgpt_rf_model.joblib'")