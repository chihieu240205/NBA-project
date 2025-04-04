import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

def simulate_lagged_data():
    players = [f"Player {i}" for i in range(150)]
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    rows = []

    for season in seasons:
        for player in players:
            row = {
                "player": player,
                "season": season,
                "mp_regular": np.random.uniform(10, 36),
                "fgpct_regular": np.random.uniform(0.3, 0.6),
                "pts_regular": np.random.uniform(5, 30),
                "ast_regular": np.random.uniform(0, 10),
                "reb_regular": np.random.uniform(1, 12),
                "pts_playoff": np.random.uniform(5, 30),
            }
            rows.append(row)
    return pd.DataFrame(rows)

def run_linear_regression(threshold=2.0):
    df = simulate_lagged_data()

    train_df = df[df['season'] != "2024-25"].copy()
    test_df = df[df['season'] == "2024-25"].copy()

    # Target variable
    train_df["point_change"] = train_df["pts_playoff"] - train_df["pts_regular"]
    test_df["point_change"] = test_df["pts_playoff"] - test_df["pts_regular"]

    features = ["mp_regular", "fgpct_regular", "pts_regular", "ast_regular", "reb_regular"]
    target = "point_change"

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_baseline = baseline.predict(X_test)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Add predictions and errors
    test_df = test_df.copy()
    test_df["predicted_change"] = y_pred
    test_df["error"] = test_df["predicted_change"] - test_df["point_change"]
    test_df["accurate"] = test_df["error"].abs() <= threshold

    # Metrics
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    baseline_r2 = r2_score(y_test, y_baseline)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    lr_r2 = r2_score(y_test, y_pred)

    # Output results
    print(f"Baseline RMSE: {baseline_rmse:.2f}, R²: {baseline_r2:.2f}")
    print(f"Linear Regression RMSE: {lr_rmse:.2f}, R²: {lr_r2:.2f}")

    # Print top 10 accurate predictions
    print("\nAccurate Predictions (within ±2 points):")
    print(test_df[test_df["accurate"]][[
        "player", "pts_regular", "pts_playoff", "point_change", "predicted_change", "error"
    ]].head(10).round(2))