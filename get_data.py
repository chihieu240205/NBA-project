import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguegamefinder


def fetch_regular_season_schedule(season="2024-25"):
    time.sleep(1)
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_nullable=season,
            season_type_nullable="Regular Season",
        )
        schedule_df = gamefinder.get_data_frames()[0]
        schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"])
        # Filter to include only games that have already ended
        schedule_df = schedule_df[schedule_df["WL"].isin(["W", "L"])]
        return schedule_df
    except Exception as e:
        print("Error retrieving schedule:", e)
        return pd.DataFrame()


def fetch_playoffs_schedule(season="2023-24"):
    time.sleep(1)
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_nullable=season,
            season_type_nullable="Playoffs",
        )
        schedule_df = gamefinder.get_data_frames()[0]
        schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"])
        return schedule_df
    except Exception as e:
        print("Error retrieving playoffs schedule:", e)
        return pd.DataFrame()


def fetch_bbr_player_avg_stats(season="2024-25"):
    try:
        parts = season.split("-")
        end_year = "20" + parts[1] if len(parts[1]) == 2 else parts[1]
        url = (
            f"https://www.basketball-reference.com/leagues/NBA_{end_year}_per_game.html"
        )
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", id="per_game_stats")
        df = pd.read_html(str(table))[0]
        df = df[df.Player != "Player"]
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print("Error retrieving player stats:", e)
        return pd.DataFrame()


def fetch_bbr_playoffs_stats(season):
    try:
        parts = season.split("-")
        end_year = "20" + parts[1] if len(parts[1]) == 2 else parts[1]
        url = f"https://www.basketball-reference.com/playoffs/NBA_{end_year}_per_game.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", id="per_game_stats")
        df = pd.read_html(str(table))[0]
        df = df[df.Player != "Player"]
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print("Error retrieving playoff stats:", e)
        return pd.DataFrame()


def preprocess_player_stats(df):
    df = df.copy()
    if "Awards" in df.columns:
        df = df.drop(columns=["Awards"])
    if "Rk" in df.columns:
        df = df.drop(columns=["Rk"])
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("%", "pct") for col in df.columns
    ]
    percent_cols = ["fgpct", "3pct", "2pct", "efgpct", "ftpct"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in percent_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
    df.reset_index(drop=True, inplace=True)
    return df
