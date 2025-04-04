import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from nba_api.stats.endpoints import playergamelog

# Ensure required packages are installed (add this in your main script if needed)
# !pip install numpy pandas scikit-learn tensorflow nba_api

def fetch_last_n_games(player_id, season, n, season_type):
    """
    Fetches the last N completed games of a player from NBA API for a given season.
    """
    try:
        time.sleep(1)  # Prevent hitting the rate limit
        logs = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
        logs_df = logs.get_data_frames()[0]
        logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'])
        logs_df = logs_df.sort_values(by='GAME_DATE', ascending=False)
        completed_games = logs_df[logs_df['MIN'].notna()]
        return completed_games.head(n)
    except Exception as e:
        print(f"Error retrieving last {n} completed games for player {player_id}: {e}")
        return pd.DataFrame()

def prepare_rnn_data(df, features=['PTS', 'AST'], sequence_length=10):
    """
    Prepares input and target sequences for RNN model from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing player game logs.
        features (list): List of features to use for prediction.
        sequence_length (int): Number of past games to consider.

    Returns:
        tuple: Scaled input (X), target (y), and fitted scaler object.
    """
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])

    return np.array(X), np.array(y), scaler

def build_rnn_model(input_shape):
    """
    Builds and compiles an RNN (LSTM) model for time-series prediction.

    Parameters:
        input_shape (tuple): Shape of input data (timesteps, features).

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict_rnn(df, epochs=50, batch_size=32):
    """
    Trains an LSTM model on player stats and predicts the next game's PTS and AST.

    Parameters:
        df (pd.DataFrame): Player game log with 'PTS' and 'AST'.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.

    Returns:
        tuple: (Predicted Points, Predicted Assists)
    """
    sequence_length = 10
    X, y, scaler = prepare_rnn_data(df, sequence_length=sequence_length)
    model = build_rnn_model((sequence_length, X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    latest_sequence = scaler.transform(df[['PTS', 'AST']].values[-sequence_length:])
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    predicted_scaled = model.predict(latest_sequence)[0]
    predicted = scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]

    return predicted[0], predicted[1]
