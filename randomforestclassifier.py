import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

def load_classification_data(threshold=-0.02):
    reg = pd.read_csv("player_avg_stats_regular.xls")
    play = pd.read_csv("player_avg_stats_playoff.xls")
    reg.columns = reg.columns.str.strip().str.upper()
    play.columns = play.columns.str.strip().str.upper()

    # Merge on player and season
    merged = pd.merge(reg, play, on=["PLAYER_NAME", "SEASON"], suffixes=("_REG", "_PLAY"))
    merged["FGPCT_DROPOFF"] = merged["FG_PCT_PLAY"] - merged["FG_PCT_REG"]

    # Classification label: 1 if dropoff exceeds threshold
    merged["DROPOFF_LABEL"] = (merged["FGPCT_DROPOFF"] <= threshold).astype(int)

    features = [
        "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "PTS", "AST", "REB", "TOV", "PLUS_MINUS"
    ]
    feature_cols = [f"{f}_REG" for f in features if f"{f}_REG" in merged.columns]

    X = merged[feature_cols].dropna()
    y = merged.loc[X.index, "DROPOFF_LABEL"]

    return X, y

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Baseline: Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy_pred = dummy.predict(X_test)
    print("\n Baseline DummyClassifier Report:")
    print(classification_report(y_test, y_dummy_pred, digits=3, zero_division=0))

    # Random Forest
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(" Random Forest Classifier Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return clf
