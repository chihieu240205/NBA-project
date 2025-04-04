import pandas as pd
import matplotlib.pyplot as plt

team_abbreviation_map = {
    # Western Conference Teams
    'OKC': 'Oklahoma City Thunder',
    'HOU': 'Houston Rockets',
    'DEN': 'Denver Nuggets',
    'LAL': 'Los Angeles Lakers',
    'GSW': 'Golden State Warriors',
    'MEM': 'Memphis Grizzlies',
    'DAL': 'Dallas Mavericks',
    'SAC': 'Sacramento Kings',
    'MIN': 'Minnesota Timberwolves',
    'LAC': 'Los Angeles Clippers',

    # Eastern Conference Teams
    'CLE': 'Cleveland Cavaliers',
    'BOS': 'Boston Celtics',
    'NYK': 'New York Knicks',
    'IND': 'Indiana Pacers',
    'DET': 'Detroit Pistons',
    'MIL': 'Milwaukee Bucks',
    'MIA': 'Miami Heat',
    'CHI': 'Chicago Bulls',
    'ORL': 'Orlando Magic',
    'ATL': 'Atlanta Hawks'
}

playoff_teams = list(team_abbreviation_map.values())

western_teams = [
    'Oklahoma City Thunder', 'Houston Rockets', 'Denver Nuggets', 'Los Angeles Lakers',
    'Golden State Warriors', 'Memphis Grizzlies', 'Dallas Mavericks', 'Sacramento Kings',
    'Minnesota Timberwolves', 'Los Angeles Clippers'
]

eastern_teams = [
    'Cleveland Cavaliers', 'Boston Celtics', 'New York Knicks', 'Indiana Pacers',
    'Detroit Pistons', 'Milwaukee Bucks', 'Miami Heat', 'Chicago Bulls',
    'Orlando Magic', 'Atlanta Hawks'
]

def analyze_win_percentages(df_schedule, team_abbreviation_map, playoff_teams, conference_teams, conference_name):
    df_schedule = df_schedule[df_schedule['WL'].isin(['W', 'L'])]

    def extract_opponent(row):
        matchup = row['MATCHUP']
        if 'vs.' in matchup:
            opponent_abbr = matchup.split('vs. ')[1]
        elif '@' in matchup:
            opponent_abbr = matchup.split('@ ')[1]
        else:
            return None
        return team_abbreviation_map.get(opponent_abbr, None) if team_abbreviation_map.get(opponent_abbr, None) in playoff_teams else None

    df_schedule['OPPONENT'] = df_schedule.apply(extract_opponent, axis=1)

    filtered_df = df_schedule[df_schedule['TEAM_NAME'].isin(conference_teams) & df_schedule['OPPONENT'].isin(conference_teams)]

    matchup_results = filtered_df.groupby(['TEAM_NAME', 'OPPONENT', 'WL']).size().reset_index(name='Count')
    matchup_pivot = matchup_results.pivot(index=['TEAM_NAME', 'OPPONENT'], columns='WL', values='Count').fillna(0)
    matchup_pivot['Total Games'] = matchup_pivot.sum(axis=1)
    matchup_pivot['Win Rate'] = (matchup_pivot.get('W', 0) / matchup_pivot['Total Games']) * 100
    matchup_pivot.reset_index(inplace=True)

    team_wins = filtered_df[filtered_df['WL'] == 'W'].groupby('TEAM_NAME').size()
    team_total_games = filtered_df.groupby('TEAM_NAME').size()
    playoff_win_percentage = (team_wins / team_total_games) * 100
    playoff_win_percentage = playoff_win_percentage.fillna(0).sort_values(ascending=False)
    playoff_win_df = playoff_win_percentage.reset_index()
    playoff_win_df.columns = ['TEAM_NAME', 'Win% Against Playoff Teams']

    total_wins = df_schedule[(df_schedule['TEAM_NAME'].isin(conference_teams)) & (df_schedule['WL'] == 'W')].groupby('TEAM_NAME').size()
    total_games = df_schedule[df_schedule['TEAM_NAME'].isin(conference_teams)].groupby('TEAM_NAME').size()
    overall_win_percentage = (total_wins / total_games) * 100
    overall_win_percentage = overall_win_percentage.fillna(0).sort_values(ascending=False)
    overall_win_df = overall_win_percentage.reset_index()
    overall_win_df.columns = ['TEAM_NAME', 'Overall Win%']

    comparison_df = pd.merge(playoff_win_df, overall_win_df, on='TEAM_NAME')
    comparison_df = comparison_df.sort_values(by='Overall Win%', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    bar_width = 0.4

    # Western Conference
    west_df = comparison_df[comparison_df['TEAM_NAME'].isin(western_teams)]
    indices = range(len(west_df))
    axes[0].bar(indices, west_df['Overall Win%'], width=bar_width, label='Overall Win%', color='blue')
    axes[0].bar([i + bar_width for i in indices], west_df['Win% Against Playoff Teams'], width=bar_width, label='Win% vs Playoff Teams', color='orange')
    axes[0].set_title('Western Conference')
    axes[0].set_xticks([i + bar_width / 2 for i in indices])
    axes[0].set_xticklabels(west_df['TEAM_NAME'], rotation=90, fontsize=8)

    # Eastern Conference
    east_df = comparison_df[comparison_df['TEAM_NAME'].isin(eastern_teams)]
    indices = range(len(east_df))
    axes[1].bar(indices, east_df['Overall Win%'], width=bar_width, label='Overall Win%', color='blue')
    axes[1].bar([i + bar_width for i in indices], east_df['Win% Against Playoff Teams'], width=bar_width, label='Win% vs Playoff Teams', color='orange')
    axes[1].set_title('Eastern Conference')
    axes[1].set_xticks([i + bar_width / 2 for i in indices])
    axes[1].set_xticklabels(east_df['TEAM_NAME'], rotation=90, fontsize=8)

    for ax in axes:
        ax.set_ylabel('Win Percentage (%)')
        ax.legend()

    plt.suptitle('Overall vs Playoff Win% by Conference', fontsize=16)
    plt.tight_layout()
    plt.show()
