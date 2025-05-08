import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'iframe_connected' 
from analyze_win_percentages import analyze_win_percentages
import get_data
from get_data import fetch_bbr_player_avg_stats, fetch_bbr_playoffs_stats

def custom_preprocess(df):
    df = df.copy()

    if "Awards" in df.columns:
        df = df.drop(columns=["Awards"])

    if "Rk" in df.columns:
        df = df.drop(columns=["Rk"])

    df.columns = [
        col.strip().lower().replace(" ", "_").replace("%", "pct") for col in df.columns
    ]

    pct_cols = ["fgpct", "3pct", "2pct", "efgpct", "ftpct"]
    for col in pct_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip("%")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in df.columns:
        if col not in pct_cols and col not in ["player", "tm", "pos"]:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def load_all_stats():
    seasons = ["2021-22", "2022-23", "2023-24"]
    all_regular_data = []
    all_playoff_data = []

    print("Fetching data for multiple seasons...")
    for season in seasons:
        regular_season = fetch_bbr_player_avg_stats(season)
        regular_season['season'] = season
        all_regular_data.append(custom_preprocess(regular_season))

        playoffs = fetch_bbr_playoffs_stats(season)
        playoffs['season'] = season
        all_playoff_data.append(custom_preprocess(playoffs))

    regular_stats = pd.concat(all_regular_data)
    playoff_stats = pd.concat(all_playoff_data)

    print(f"Combined regular season data shape: {regular_stats.shape}")
    print(f"Combined playoff data shape: {playoff_stats.shape}")

    return regular_stats, playoff_stats

def compare_fg_percentage(regular_stats, playoff_stats, min_minutes=15):
    if "player" in regular_stats.columns and "player" in playoff_stats.columns:
        merged_data = pd.merge(
            regular_stats, playoff_stats, on=["player", "season"], suffixes=("_regular", "_playoff")
        )

        print(f"Number of player-seasons in both regular season and playoffs: {len(merged_data)}")

        filtered_data = merged_data[
            (merged_data["mp_regular"] >= min_minutes) & (merged_data["mp_playoff"] >= min_minutes)
        ]

        fg_diff = filtered_data["fgpct_playoff"] - filtered_data["fgpct_regular"]
        avg_diff = fg_diff.mean()
        print(f"Average difference in FG% (Playoff - Regular): {avg_diff:.4f}")

        improved = (fg_diff > 0).sum()
        declined = (fg_diff < 0).sum()
        same = (fg_diff == 0).sum()

        print(f"Players with improved FG% in playoffs: {improved} ({improved/len(fg_diff):.1%})")
        print(f"Players with declined FG% in playoffs: {declined} ({declined/len(fg_diff):.1%})")
        print(f"Players with unchanged FG% in playoffs: {same} ({same/len(fg_diff):.1%})")
    else:
        print("Missing 'player' column in one or both datasets.")

def plot_fg_percentage_scatter(regular_stats, playoff_stats, min_minutes=15):
    if "player" in regular_stats.columns and "player" in playoff_stats.columns:
        merged_data = pd.merge(
            regular_stats, playoff_stats, on=["player", "season"], suffixes=("_regular", "_playoff")
        )

        filtered_data = merged_data[
            (merged_data["mp_regular"] >= min_minutes) & (merged_data["mp_playoff"] >= min_minutes)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=filtered_data["fgpct_regular"],
            y=filtered_data["fgpct_playoff"],
            mode='markers',
            hovertemplate="<b>Player:</b> %{text}<br>" +
                          "Mins Played: %{customdata[0]}<br>" +
                          "Mins Played Playoffs: %{customdata[1]}<br>" +
                          "Regular Season FG%: %{x:.3f}<br>" +
                          "Playoff FG%: %{y:.3f}<br>" +
                          "<extra></extra>",
            text=filtered_data["player"],
            customdata=np.column_stack((filtered_data["mp_regular"], filtered_data["mp_playoff"])),
            name='Players'
        ))

        min_val = min(filtered_data["fgpct_regular"].min(), filtered_data["fgpct_playoff"].min())
        max_val = max(filtered_data["fgpct_regular"].max(), filtered_data["fgpct_playoff"].max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Equal Performance',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title="Regular Season vs Playoff Field Goal Percentage (2021-24 Seasons)",
            xaxis_title="Regular Season FG%",
            yaxis_title="Playoff FG%",
            width=900,
            height=500,
            showlegend=True,
            template='plotly_white',
            hovermode='closest',
            autosize=True
        )

        fig.show()
    else:
        print("Missing 'player' column in one or both datasets.")
