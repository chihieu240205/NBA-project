import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_prepare_data():
    regular_df = pd.read_csv("player_avg_stats_regular.xls")
    playoff_df = pd.read_csv("player_avg_stats_playoff.xls")

    regular_df.columns = regular_df.columns.str.strip().str.upper()
    playoff_df.columns = playoff_df.columns.str.strip().str.upper()

    # Merge on player name and season
    merged = pd.merge(
        regular_df, playoff_df,
        on=["PLAYER_NAME", "SEASON"],
        suffixes=("_REG", "_PLAY")
    )

    merged = merged.dropna(subset=["FG_PCT_REG", "FG_PCT_PLAY"])

    merged["FGPCT_DROPOFF"] = merged["FG_PCT_PLAY"] - merged["FG_PCT_REG"]

    good_stats = [
        "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "REB", "AST", "TOV", "PTS", "PLUS_MINUS"
    ]
    feature_cols = [f"{stat}_REG" for stat in good_stats if f"{stat}_REG" in merged.columns]

    X = merged[feature_cols].dropna()
    y = merged.loc[X.index, "FGPCT_DROPOFF"]

    return X, y

def train_random_forest_with_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline Dummy Regressor
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, y_dummy)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_dummy))

    # Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print sample predictions
    rf_results = pd.DataFrame({
        "Actual_Dropoff": y_test.values,
        "RF_Predicted": y_pred,
        "Error": y_pred - y_test.values
    })
    print("\n Random Forest Predictions (sample):")
    print(rf_results.head(5).round(4))

    return model, baseline_mae, baseline_rmse, model_mae, model_rmse