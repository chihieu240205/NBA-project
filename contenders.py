
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# 1. Helper: Fetch and aggregate regular-season stats
def get_season_team_stats(season='2024-25'):
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    games = finder.get_data_frames()[0]

    # Home/away and win flags
    games['is_home'] = games['MATCHUP'].str.contains(' vs\. ')
    games['is_away'] = games['MATCHUP'].str.contains('@')
    games['is_win']  = games['WL'] == 'W'
    games['home_win'] = games['is_home'] & games['is_win']
    games['away_win'] = games['is_away'] & games['is_win']

    # Aggregate counts per team
    agg = games.groupby('TEAM_ID').agg(
        games_played     = ('GAME_ID',    'nunique'),
        wins             = ('is_win',     'sum'),
        pts_for          = ('PTS',        'sum'),
        fgm              = ('FGM',        'sum'),
        fga              = ('FGA',        'sum'),
        fg3m             = ('FG3M',       'sum'),
        fg3a             = ('FG3A',       'sum'),
        ftm              = ('FTM',        'sum'),
        fta              = ('FTA',        'sum'),
        oreb             = ('OREB',       'sum'),
        dreb             = ('DREB',       'sum'),
        tov              = ('TOV',        'sum'),
        personal_fouls   = ('PF',         'sum'),
        avg_plus_minus   = ('PLUS_MINUS', 'mean'),
        home_games       = ('is_home',    'sum'),
        home_wins        = ('home_win',   'sum'),
        away_games       = ('is_away',    'sum'),
        away_wins        = ('away_win',   'sum')
    ).reset_index()

    # Derive percentages
    agg['win_pct']      = agg['wins']       / agg['games_played']
    agg['fg_pct']       = agg['fgm']        / agg['fga']
    agg['fg3_pct']      = agg['fg3m']       / agg['fg3a']
    agg['ft_pct']       = agg['ftm']        / agg['fta']
    agg['home_win_pct'] = agg['home_wins']  / agg['home_games']
    agg['away_win_pct'] = agg['away_wins']  / agg['away_games']

    return agg

# 2. Label map: historical playoff performance → tier
label_map = {
    'Finals':       2,    # Title Contender
    'Conf. Finals': 1,    # Dark Horse
    'R1 Exit':      0     # Fringe
}

# 3. Build the training set across seasons
def build_training_set(seasons=None):
    if seasons is None:
        seasons = ['2021-22', '2022-23', '2023-24']
    all_frames = []
    for season in seasons:
        # 3.1: Get regular-season features
        stats = get_season_team_stats(season)

        # 3.2: Retrieve playoff results for that season
        #    Assume `playoff_results_df` exists with columns ['SEASON_ID','TEAM_ID','playoff_result']
        pr = playoff_results_df.copy()
        pr = pr[pr['SEASON_ID'] == season][['TEAM_ID', 'playoff_result']]

        # 3.3: Merge features with labels
        df = stats.merge(pr, on='TEAM_ID', how='inner')

        # 3.4: Map textual results to numeric labels
        df['label'] = df['playoff_result'].map(label_map)
        all_frames.append(df)

    # 3.5: Concatenate all seasons into one DataFrame
    df_train = pd.concat(all_frames, ignore_index=True)
    return df_train
