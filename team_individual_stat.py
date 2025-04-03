import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time
from nba_api.stats.endpoints import leaguedashplayerstats

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
    
def wrangle_player_stats(df):
    """
    Cleans and wrangles the NBA player per game stats data.
    
    Steps performed:
      - Remove duplicate rows.
      - Drop ranking columns (any column that includes '_RANK').
      - Optionally drop extra identifier columns (e.g., 'NICKNAME').
      - Standardize column names by converting them to lowercase.
    
    Args:
        df (pd.DataFrame): DataFrame containing player per game stats.
        
    Returns:
        pd.DataFrame: The cleaned and wrangled DataFrame.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Drop ranking columns (any column that contains 'RANK')
    rank_columns = [col for col in df_clean.columns if 'RANK' in col]
    df_clean.drop(columns=rank_columns, inplace=True)
    
    # Optionally drop extra identifier columns that might not be useful for modeling
    # For example, you may decide to drop 'NICKNAME'
    if 'NICKNAME' in df_clean.columns:
        df_clean.drop(columns=['NICKNAME'], inplace=True)
    
    # Standardize column names to lowercase for consistency
    df_clean.columns = [col.lower() for col in df_clean.columns]
    
    return df_clean