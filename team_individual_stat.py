import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time
from nba_api.stats.endpoints import leaguedashplayerstats
import warnings
warnings.filterwarnings('ignore')

def fetch_regular_season_schedule(season):
    """
    Fetches all regular season fixtures for the specified season with all available attributes.
    
    Parameters:
        season (str): NBA season in 'YYYY-YY' format (default is '2024-25').
        
    Returns:
        pd.DataFrame: A DataFrame containing all regular season games for the specified season.
    """
    try:
        # Pause briefly in case of rate limiting
        time.sleep(1)
        # Use leaguegamefinder to retrieve games.
        # The parameter season_type_nullable is set to 'Regular Season' to filter only regular season games.
        gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00',season_nullable=season, season_type_nullable='Regular Season')
        schedule_df = gamefinder.get_data_frames()[0]
        return schedule_df
    except Exception as e:
        print("Error retrieving schedule:", e)
        return pd.DataFrame()
    
def fetch_playoffs_schedule(season):
    """
    Fetches all playoff fixtures for the specified season using LeagueGameFinder.
    
    Parameters:
        season (str): NBA season in 'YYYY-YY' format (default is '2022-23').
        
    Returns:
        pd.DataFrame: A DataFrame containing all playoff games for the specified season.
    """
    try:
        time.sleep(1)  # Respect API rate limits
        # Use LeagueGameFinder with season_type_nullable set to 'Playoffs'
        gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00',season_nullable=season, season_type_nullable='Playoffs')
        schedule_df = gamefinder.get_data_frames()[0]
        schedule_df['GAME_DATE'] = pd.to_datetime(schedule_df['GAME_DATE'])
        return schedule_df
    except Exception as e:
        print("Error retrieving playoffs schedule:", e)
        return pd.DataFrame()
    

def fetch_player_per_game_stats(season, season_type):
    """
    Fetches per game NBA player stats for a given season and season type.
    
    Args:
        season (str): Season string, e.g., '2021-22'.
        season_type (str): Type of season ('Regular Season' or 'Playoffs').
        
    Returns:
        pd.DataFrame: DataFrame with per game player statistics for the season.
    """
    try:
        # Pause briefly to mitigate potential rate limiting issues
        time.sleep(1)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='PerGame'  # Ensure stats are per game
        )
        df = stats.get_data_frames()[0]
        # Add a column to record the season for clarity
        df['Season'] = season
        return df
    except Exception as e:
        print(f"Error retrieving {season_type} stats for {season}: {e}")
        return pd.DataFrame()
    
def wrangle_player_performance(df):
    """
    Cleans and wrangles the player per game stats data for reviewing individual performance.
    
    Cleaning steps performed:
      1. Convert all column names to uppercase.
      2. Remove duplicate rows.
      3. Drop irrelevant columns: ranking columns (columns containing 'RANK') and extra identifiers (e.g., 'NICKNAME', 'W', 'L', 'W_PCT').
      4. Handle missing values by dropping rows with missing data.
      5. Convert applicable columns to numeric types.
      6. Standardize column names (ensuring they remain uppercase and stripped).
      7. Retain only needed attributes.
    """
    # 1. Convert all column names to uppercase and strip whitespace
    df.columns = [col.strip().upper() for col in df.columns]

    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # 2. Remove duplicate rows
    df_clean.drop_duplicates(inplace=True)

    # 3. Drop irrelevant columns
    rank_cols = [col for col in df_clean.columns if 'RANK' in col]
    extra_irrelevant_cols = ['NICKNAME', 'W', 'L', 'W_PCT']
    cols_to_drop = rank_cols + [col for col in extra_irrelevant_cols if col in df_clean.columns]
    df_clean.drop(columns=cols_to_drop, inplace=True)

    # 4. Handle missing values
    df_clean.dropna(inplace=True)

    # 5. Convert to numeric where applicable
    non_numeric_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON']
    for col in df_clean.columns:
        if col not in non_numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')

    # 6. Ensure column names are standardized
    df_clean.columns = [col.strip().upper() for col in df_clean.columns]

    # 7. Keep only needed attributes
    needed_cols = [
        'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'MIN',
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS',
        'NBA_FANTASY_PTS', 'DD2', 'TD3', 'SEASON'
    ]
    df_clean = df_clean[[col for col in needed_cols if col in df_clean.columns]]

    return df_clean