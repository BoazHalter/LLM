import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

print("--- Starting Hyperparameter Tuning ---")

# 1. Load and prepare data
df = pd.read_csv('processed_features.csv', index_col='DateTime', parse_dates=True)

features = [
    'Close', 'SMA_20', 'EMA_20', 'BB_Upper', 'BB_Lower', 'RSI_14', 'MACD', 'MACD_Signal',
    'Stoch_K_9_3', 'Stoch_D_9_3', 'Stoch_K_14_3', 'Stoch_D_14_3', 'Stoch_K_44_4', 'Stoch_D_44_4',
    'Stoch_K_60_10', 'Stoch_D_60_10', 'Stoch_K_9_3_Crossover', 'Stoch_K_9_3_Level',
    'Stoch_K_14_3_Crossover', 'Stoch_K_14_3_Level', 'Stoch_K_44_4_Crossover', 'Stoch_K_44_4_Level',
    'Stoch_K_60_10_Crossover', 'Stoch_K_60_10_Level', 'Stoch_K_9_3_Momentum', 'Stoch_K_14_3_Momentum',
    'Stoch_K_44_4_Momentum', 'Stoch_K_60_10_Momentum'
]

df['Target'] = df['Close'].pct_change().shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df['Target']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# We will use the full dataset for cross-validated search
# but you could also use just the training split for faster results.
train_size = int(len(X) * 0.8)
X_train, y_train = X_scaled[:train_size], y.iloc[:train_size]

print(f"Tuning on {len(X_train)} data points.")

# 2. Define the hyperparameter grid
# These are the settings RandomizedSearchCV will try.
param_grid = {
    'n_estimators': [100, 200, 300], # Number of trees in the forest
    'max_features': ['sqrt', 'log2'],      # Number of features to consider at every split
    'max_depth': [10, 20, 30, None],       # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]          # Minimum number of samples required at each leaf node
}

# 3. Set up RandomizedSearchCV
# n_iter: Number of different parameter combinations to try.
# cv: Number of cross-validation folds.
# n_jobs=-1: Use all available CPU cores.
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,  # Increase for a more thorough search, decrease for speed
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# 4. Run the search
print("\nFitting models with RandomizedSearchCV...")
random_search.fit(X_train, y_train)

# 5. Display the best parameters and save the best model
print("\n--- Tuning Complete ---")
print("Best parameters found: ")
print(random_search.best_params_)

# The best model found by the search
best_model = random_search.best_estimator_

# Save the best model so you can use it in your training and backtesting scripts
joblib.dump(best_model, 'chatgpt_rf_model_tuned.joblib')
print("\nBest model saved as 'chatgpt_rf_model_tuned.joblib'")

print("\nNext Steps:")
print("1. Update 'Model-Training-1.py' with these new hyperparameters.")
print("2. Re-run 'Model-Training-1.py' to train on the full training set and save the final model.")
print("3. Re-run 'backtest.py' to see if performance has improved.")

```

### How to Use It

1.  **Run the Tuning Script:** Execute this new script from your terminal. It may take some time to run as it's training multiple models.
    ```shell
    py tune_model.py
    ```
2.  **Update Your Training Script:** Once the script finishes, it will print the best combination of parameters it found. You should then update the `RandomForestRegressor` line in `Model-Training-1.py` with these new, optimized parameters.
3.  **Re-train and Re-backtest:** After updating the training script, run it again to create a new, more powerful `chatgpt_rf_model.joblib`. Finally, run `backtest.py` again to see how the optimized model performs.

This iterative process of tuning, training, and backtesting is the core loop of developing a quantitative trading model.

<!--
[PROMPT_SUGGESTION]I have the best parameters. How do I update my training script and re-run the backtest?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]The backtest is still not profitable. What other features could I engineer?[/PROMPT_SUGGESTION]
-->