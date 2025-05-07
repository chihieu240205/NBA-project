pip install numpy pandas scikit-learn xgboost lightgbm requests tqdm joblib

pip install nba_api

pip install nba_api pandas scikit-learn numpy


from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
import pandas as pd
games = LeagueGameFinder(
    season_nullable="2024-25",
    league_id_nullable="00",
    season_type_nullable="Regular Season"
).get_data_frames()[0]

games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games[games['MIN'] > 0]
games = games[~games['MATCHUP'].str.contains('All-Star', na=False)]
games


from nba_api.stats.static import teams
from pprint import pprint

all_teams = teams.get_teams()
pprint(all_teams, width=200)

games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games[games['MIN'] > 0]
games = games[~games['MATCHUP'].str.contains('All-Star', na=False)]

playoff_teams = [
    # West
    1610612760,  # OKC Thunder
    1610612745,  # HOU Rockets
    1610612747,  # LAL Lakers
    1610612743,  # DEN Nuggets
    1610612746,  # LAC Clippers
    1610612750,  # MIN Timberwolves
    1610612744,  # GSW Warriors
    1610612763,  # MEM Grizzlies

    # East
    1610612739,  # CLE Cavaliers
    1610612738,  # BOS Celtics
    1610612752,  # NYK Knicks
    1610612754,  # IND Pacers
    1610612749,  # MIL Bucks
    1610612765,  # DET Pistons
    1610612753,  # ORL Magic
    1610612748   # MIA Heat
]

games = games[games['TEAM_ID'].isin(playoff_teams)]

games['a_home'] = games['MATCHUP'].str.contains(r' vs\.').astype(int)

a = games.rename(columns=lambda c: 'a_'+c if c not in ['GAME_ID','GAME_DATE'] else c)
b = games.rename(columns=lambda c: 'b_'+c if c not in ['GAME_ID','GAME_DATE'] else c)

merged = (
    a.merge(b, on=['GAME_ID','GAME_DATE'])
     .query('a_TEAM_ID != b_TEAM_ID')
)
merged['a_win'] = (merged['a_WL'] == 'W').astype(int)

merged['a_efg'] = (merged['a_FGM'] + 0.5 * merged['a_FG3M']) / merged['a_FGA']
merged['b_efg'] = (merged['b_FGM'] + 0.5 * merged['b_FG3M']) / merged['b_FGA']
merged['a_ts']  = merged['a_PTS'] / (2 * merged['a_FGA'] + 0.44 * merged['a_FTA'])
merged['b_ts']  = merged['b_PTS'] / (2 * merged['b_FGA'] + 0.44 * merged['b_FTA'])

# 2) 30-game rolling means of efg and ts
for stat in ['efg','ts']:
    merged[f'a_{stat}_30gm'] = (
        merged.groupby('a_TEAM_ID')[f'a_{stat}']
              .rolling(30, min_periods=1).mean()
              .reset_index(level=0, drop=True)
    )
    merged[f'b_{stat}_30gm'] = (
        merged.groupby('b_TEAM_ID')[f'b_{stat}']
              .rolling(30, min_periods=1).mean()
              .reset_index(level=0, drop=True)
    )
    merged[f'diff_{stat}_30gm'] = merged[f'a_{stat}_30gm'] - merged[f'b_{stat}_30gm']

# 3) Rebounding & assist/turnover differentials (season‐to‐date)
merged['a_rim'] = merged['a_OREB'] + merged['a_DREB']
merged['b_rim'] = merged['b_OREB'] + merged['b_DREB']
merged['a_ast_to'] = merged['a_AST'] / merged['a_TOV'].replace(0,1)
merged['b_ast_to'] = merged['b_AST'] / merged['b_TOV'].replace(0,1)

for stat in ['rim','ast_to']:
    merged[f'a_{stat}_30gm'] = (
        merged.groupby('a_TEAM_ID')[f'a_{stat}']
              .rolling(30, min_periods=1).mean()
              .reset_index(level=0, drop=True)
    )
    merged[f'b_{stat}_30gm'] = (
        merged.groupby('b_TEAM_ID')[f'b_{stat}']
              .rolling(30, min_periods=1).mean()
              .reset_index(level=0, drop=True)
    )
    merged[f'diff_{stat}_30gm'] = merged[f'a_{stat}_30gm'] - merged[f'b_{stat}_30gm']

# 4) Rest‐day differential
merged['a_rest'] = (merged.groupby('a_TEAM_ID')['GAME_DATE']
                          .diff().dt.days.fillna(1))
merged['b_rest'] = (merged.groupby('b_TEAM_ID')['GAME_DATE']
                          .diff().dt.days.fillna(1))
merged['diff_rest'] = merged['a_rest'] - merged['b_rest']

# 5) Head‑to‑head win% so far this season
h2h = (merged.groupby(['a_TEAM_ID','b_TEAM_ID'])['a_win']
           .expanding().mean()
           .reset_index()
           .rename(columns={'a_win':'h2h_pct'}))
merged = merged.merge(
    h2h[['a_TEAM_ID','b_TEAM_ID','h2h_pct']],
    on=['a_TEAM_ID','b_TEAM_ID'],
    how='left'
)
merged['h2h_pct'].fillna(0.5, inplace=True)

# 6) Assemble the enhanced feature matrix with updated column names
feature_cols = [
    'a_a_home',
    'diff_efg_30gm', 'diff_ts_30gm',
    'diff_rim_30gm', 'diff_ast_to_30gm',
    'diff_rest',   'h2h_pct'
]
X = merged[feature_cols]
y = merged['a_win']


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
model = LogisticRegression().fit(X_train, y_train)
preds = model.predict_proba(X_val)[:,1]
print("AUC =", roc_auc_score(y_val, preds))

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
clf = LogisticRegression(max_iter=1000)
grid = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("best C:", grid.best_params_['C'])
print("cv AUC:", grid.best_score_)
model = grid.best_estimator_

# 1. Lock in your best‐C model and check hold‑out AUC
model = grid.best_estimator_
val_preds = model.predict_proba(X_val)[:,1]
print("Hold‑out AUC =", roc_auc_score(y_val, val_preds))

# 2. (Optional but recommended) Calibrate if you see mis‑calibration
from sklearn.calibration import CalibratedClassifierCV
calib = CalibratedClassifierCV(model, method='isotonic', cv=5)
calib.fit(X_train, y_train)
val_preds_cal = calib.predict_proba(X_val)[:,1]
print("Calibrated AUC =", roc_auc_score(y_val, val_preds_cal))

# 3. Compute a best‑of‑7 series win‑probability function
from scipy.stats import binom

def series_win_prob(p):
    # P(win ≥4 out of 7) = ∑_{k=4}^7 C(7,k) p^k (1−p)^(7−k)
    return sum(binom.pmf(k, 7, p) for k in range(4, 8))

# 4. Example: get per‑game p for Team A vs B, then series p
#    (assuming you've built a feature row `x_feat` for A @ home first)
# Here we define a sample x_feat using the input features used in the model.
x_feat = merged[feature_cols].iloc[-1].values
p_game = model.predict_proba([x_feat])[0,1]   # or calib.predict_proba
p_series = series_win_prob(p_game)
print(f"P(game) = {p_game:.3f}, P(series) = {p_series:.3f}")


# Given playoff_teams = [West seeds 1–8, then East seeds 1–8]
west = playoff_teams[:8]
east = playoff_teams[8:]

# Round 1: 1 vs 8, 2 vs 7, 3 vs 6, 4 vs 5 in each conference
first_round = [
    # East matchups
    (east[0], east[7]),  # 1 vs 8
    (east[1], east[6]),  # 2 vs 7
    (east[2], east[5]),  # 3 vs 6
    (east[3], east[4]),  # 4 vs 5

    # West matchups
    (west[0], west[7]),  # 1 vs 8
    (west[1], west[6]),  # 2 vs 7
    (west[2], west[5]),  # 3 vs 6
    (west[3], west[4]),  # 4 vs 5
]

print("Playoffs:", playoff_teams)

raw_games = leaguegamefinder.LeagueGameFinder(
    season_nullable="2024-25",
    league_id_nullable="00"
).get_data_frames()[0]

# 1) Drop DNPs
df = raw_games[raw_games['MIN'] > 0].copy()

# 2) Flag home/away
df['home'] = df['MATCHUP'].str.contains(r' vs\.').astype(int)

# 3) Merge in opponent points
opp = df[['GAME_ID','PTS']].rename(columns={'PTS':'opp_PTS'})
df = df.merge(opp, on='GAME_ID')

# 4) Per‑game diff & 5‑game rolling mean
df['diff'] = df['PTS'] - df['opp_PTS']
df = df.sort_values(['TEAM_ID','GAME_DATE'])
df['diff_30gm'] = (
    df
    .groupby('TEAM_ID')['diff']
    .rolling(window=30, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# 5) Final rolling diff for every team
last_diff = df.groupby('TEAM_ID')['diff_30gm'].last().to_dict()

from scipy.stats import binom

def series_win_prob(p):
    return sum(binom.pmf(k, 7, p) for k in range(4, 8))

last_diff = merged.groupby('a_TEAM_ID')['diff_ts_30gm'].last().to_dict()

series_p = {}
for A, B in first_round:
    d = last_diff[A] - last_diff[B]
    # Construct feature vector with dummy/default values for the missing features.
    # Our feature_cols are:
    # [a_a_home, diff_efg_10gm, diff_ts_10gm, diff_rim_10gm, diff_ast_to_10gm, diff_rest, h2h_pct]
    # We'll assume a_a_home=1, diff_efg_10gm=0, diff_ts_10gm=d, diff_rim_10gm=0, diff_ast_to_10gm=0,
    # diff_rest=0, and h2h_pct=0.5 as default values.
    features = [1, 0, d, 0, 0, 0, 0.5]
    p_game = model.predict_proba([features])[0, 1]
    series_p[(A, B)] = series_win_prob(p_game)

import warnings

# ignore only the "X does not have valid feature names" message
warnings.filterwarnings(
    action='ignore',
    message='X does not have valid feature names',
    category=UserWarning
)

import random
from collections import Counter

import numpy as np

seed_map = {
    # West seeds
    1610612760: 1, 1610612745: 2, 1610612747: 3, 1610612743: 4,
    1610612746: 5, 1610612750: 6, 1610612744: 7, 1610612763: 8,
    # East seeds
    1610612739: 1, 1610612738: 2, 1610612752: 3, 1610612754: 4,
    1610612749: 5, 1610612765: 6, 1610612753: 7, 1610612748: 8
}


# 2) Penalize underdogs via a logit shift
beta = 0.05  # larger = harsher penalty
def penalize_by_seed(p, A, B, beta=beta):
    seedA, seedB = seed_map[A], seed_map[B]
    diff = seedA - seedB
    logit = np.log(p / (1 - p))
    logit_adj = logit - beta * diff
    return 1 / (1 + np.exp(-logit_adj))

# 3) Best‐of‐7 simulation with per‐game penalty
home_seq = [1,1,0,0,1,0,1]
def simulate_series(A, B):
    winsA = winsB = 0
    for home in home_seq:
        d = last_diff[A] - last_diff[B]
        features = [1, 0, d, 0, 0, 0, 0.5]
        p_game = model.predict_proba([features])[0,1]
        p_game = penalize_by_seed(p_game, A, B)
        if random.random() < p_game:
            winsA += 1
        else:
            winsB += 1
        if winsA == 4 or winsB == 4:
            return A if winsA == 4 else B
    return A

# simulate an entire bracket
def simulate_bracket():
    # Round 1: eight series in first_round order
    r1 = [simulate_series(A, B) for A, B in first_round]

    # Round 2: East winners vs each other, then West
    east_sf = [simulate_series(r1[0], r1[3]), simulate_series(r1[1], r1[2])]
    west_sf = [simulate_series(r1[4], r1[7]), simulate_series(r1[5], r1[6])]

    # Conference Finals
    east_f = simulate_series(east_sf[0], east_sf[1])
    west_f = simulate_series(west_sf[0], west_sf[1])

    # NBA Finals
    return simulate_series(east_f, west_f)

counts_r2 = Counter()       # winners of Round 1 → make Round 2 (Conf. Semis)
counts_r3 = Counter()       # winners of Round 2 → make Round 3 (Conf. Finals)
counts_finals = Counter()   # winners of Round 3 → make NBA Finals
counts_champs = Counter()   # champions

N = 20000
for _ in range(N):
    # Round 1: simulate eight series (16 teams → 8 winners)
    r1 = [simulate_series(A, B) for A, B in first_round]
    counts_r2.update(r1)

    # Round 2: separate conference semis
    east_sf = [simulate_series(r1[i], r1[i+1]) for i in (0, 2)]
    west_sf = [simulate_series(r1[i], r1[i+1]) for i in (4, 6)]
    r2 = east_sf + west_sf
    counts_r3.update(r2)

    # Round 3: Conference Finals
    east_f = simulate_series(east_sf[0], east_sf[1])
    west_f = simulate_series(west_sf[0], west_sf[1])
    finals = [east_f, west_f]
    counts_finals.update(finals)

    # Championship
    champ = simulate_series(east_f, west_f)
    counts_champs[champ] += 1

# Build probability DataFrame
probs = []
for tid in playoff_teams:
    probs.append({
        'team_id':            tid,
        'make_R2 (8)':        counts_r2[tid] / N,
        'make_R3 (4)':        counts_r3[tid] / N,
        'make_Finals (2)':    counts_finals[tid] / N,
        'champion (1)':       counts_champs[tid] / N
    })

df_probs = pd.DataFrame(probs)
try:
    id_to_name
except NameError:
    from nba_api.stats.static import teams
    all_teams = teams.get_teams()
    id_to_name = {team['id']: team['full_name'] for team in all_teams}

df_probs['team_name'] = df_probs['team_id'].map(id_to_name)
df_probs = df_probs[['team_name','team_id','make_R2 (8)','make_R3 (4)','make_Finals (2)','champion (1)']]

print(df_probs.sort_values('champion (1)', ascending=False))

# Explanation:
# The simulation starts from the first round defined in cell 10, which consists of 8 matchups (16 teams).
# Only one team per matchup advances, so only the 8 winners from round 1 (and any teams that advance further)
# can become champions. That is why the final counters (champs) include only around 8 teams.

df_probs

def classify_team(seed, champ_prob):
    # Define tiers based on seed and champion probability.
    if seed in [1, 2, 3]:
        if champ_prob >= 0.10:
            return "Title Contender"
        elif champ_prob >= 0.05:
            return "Contender"
        else:
            return "Underperformer"
    elif seed in [4, 5, 6]:
        if champ_prob >= 0.10:
            return "Contender"
        elif champ_prob >= 0.05:
            return "Dark Horse"
        else:
            return "Sleeper"
    else:  # seed in [7, 8]
        if champ_prob >= 0.10:
            return "Dark Horse"
        elif champ_prob >= 0.05:
            return "Sleeper"
        else:
            return "Long Shot"

# Add seed column using the existing seed_map dictionary
df_probs['seed'] = df_probs['team_id'].map(seed_map)

# The column 'champion (1)' holds the champion probability.
df_probs['tier'] = df_probs.apply(lambda row: classify_team(row['seed'], row['champion (1)']), axis=1)

print(df_probs[['team_name', 'seed', 'champion (1)', 'tier']])