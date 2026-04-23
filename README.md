# NBA Playoff Match Day Performance Predictor

A data science and machine learning project that analyzes how NBA players and teams perform in the playoffs compared to the regular season. This project combines data collection, cleaning, visualization, and predictive modeling to study playoff shooting dropoff, team performance against playoff-caliber opponents, and first-round series outcomes.

## Team

This was a collaborative course project completed by:

- Chi Hieu Nguyen
- Jesus Rojas
- Daniel Rodriguez
- Dinh Dang Khoa Tran
- Duc Tam Nguyen

## Overview

Playoff basketball is different from the regular season: the pace slows down, defenses become tougher, and teams rely more heavily on scouting and matchup adjustments. Because of that, regular-season averages do not always transfer well into playoff settings.

Our project studies this gap in three ways:

1. **Player performance analysis** – comparing regular-season and playoff shooting efficiency  
2. **Team performance analysis** – comparing overall win rate to win rate against playoff-caliber opponents  
3. **Predictive modeling** – testing machine learning models for player shooting dropoff and team playoff outcomes  

The goal was to better understand which players and teams remain effective under playoff pressure, and how much can realistically be predicted using box-score-level data. :contentReference[oaicite:0]{index=0}

## Project Questions

- Do NBA players shoot differently in the playoffs than in the regular season?
- Are overall regular-season win percentages enough to judge playoff readiness?
- Can regular-season box score data predict playoff shooting dropoff?
- Can team-level data be used to estimate playoff or title-contender strength?

## Data Sources

We used data collected from:

- **NBA API** for schedules and game logs
- **Basketball Reference-based stats** for player averages across seasons

The cleaned data spans multiple regular seasons and playoff seasons, including current-season data used for comparison and forecasting. The report states that the combined dataset included **2,229 regular-season rows** and **648 playoff rows** after preprocessing. :contentReference[oaicite:1]{index=1}

## Data Preparation

Our preprocessing pipeline included:

- fetching regular-season and playoff data across multiple seasons
- cleaning column names
- removing unnecessary columns such as ranking and awards fields when present
- converting percentage columns to numeric values
- imputing missing values with medians
- organizing the data at the **player-season** level for analysis and modeling :contentReference[oaicite:2]{index=2}

## Exploratory Data Analysis

### 1. Regular Season vs. Playoff Shooting

We compared player field goal percentage in the regular season versus the playoffs.

### Key finding
- **63.6%** of players saw a decline in FG% in the playoffs
- **35.7%** improved their FG% in the playoffs
- the average playoff-minus-regular-season FG% difference was **-0.0180**, meaning players shot about **1.8% worse on average** in the playoffs :contentReference[oaicite:3]{index=3}

### 2. Team Performance vs. Playoff-Caliber Opponents

We also compared each team’s overall win percentage against its win percentage against playoff-level teams.

### Key finding
Most teams performed worse against playoff-caliber opponents than they did overall, though stronger teams such as the **Cavaliers** and **Thunder** held up better in tougher matchups. :contentReference[oaicite:4]{index=4}

## Machine Learning Models

### A. Predicting Player Shooting Dropoff

We tested regression and classification approaches using regular-season box-score features.

#### Models used
- Random Forest Regression
- XGBoost Regression
- Random Forest Classifier
- Dummy baseline classifier

#### Regression results
- **Random Forest:** MAE = **0.1134**, RMSE = **0.1615**
- **XGBoost:** MAE = **0.1160**, RMSE = **0.1705**
- **Baseline:** MAE = **0.1095**, RMSE = **0.1493** :contentReference[oaicite:5]{index=5}

These results showed that basic box-score features were **not strong enough** to predict playoff shooting decline accurately. The report concludes that factors such as defensive pressure, shot contest quality, usage, and lineup context likely matter more than standard counting stats alone. :contentReference[oaicite:6]{index=6}

#### Classification results
We also built a classifier to identify players with larger shooting declines. The Random Forest classifier achieved **60.5% accuracy**, but still struggled to identify dropoff cases reliably. :contentReference[oaicite:7]{index=7}

### B. Predicting First-Round Series / Team Outlook

We also built a team-level playoff forecasting pipeline using:

- Logistic Regression
- GridSearchCV for tuning
- probability calibration
- Monte Carlo best-of-7 series simulation

#### Team model results
- **AUC = 0.7462**
- **best C = 100**
- **CV AUC = 0.8034**
- **Hold-out AUC = 0.7370**
- **Calibrated AUC = 0.7323** :contentReference[oaicite:8]{index=8}

This model grouped teams into tiers such as **Contender**, **Dark Horse**, **Sleeper**, and **Long Shot**. The report highlights teams such as the **Cavaliers** and **Celtics** as strong contenders, while the **Pacers** emerged as a possible dark-horse team. :contentReference[oaicite:9]{index=9}

## Key Takeaways

- Most NBA players shoot worse in the playoffs than in the regular season
- Team success against playoff-caliber opponents gives a better signal than overall win percentage alone
- Basic box-score data is not enough to predict individual playoff shooting dropoff well
- Team-level playoff forecasting worked better than player-level dropoff prediction
- Better future models would likely need richer data such as matchup difficulty, shot quality, defensive pressure, injuries, and player-tracking information :contentReference[oaicite:10]{index=10}

## Visuals

Add screenshots here from your notebook or report, for example:

- regular season vs playoff FG% scatter plot
- overall win% vs playoff-opponent win% bar chart
- model output tables
- team contender tier chart

Example markdown:
```md
![FG Percentage Scatter](images/fg_scatter.png)
![Playoff Opponent Win Percentage](images/team_win_pct.png)
![Model Results](images/model_results.png)
```

## Repository Structure
```md
.
├── get_data.py
├── player_visuals.py
├── team_visuals.py
├── randomforestreg.py
├── randomforestclassifier.py
├── xgboostmodel.py
├── mc_module.py
├── CS418Final_report.ipynb
└── README.md
```

## How to Run
Clone the repository
Install dependencies
Run the data-fetching and analysis scripts or notebook

Example:
```md
git clone https://github.com/chihieu240205/NBA-project.git
cd NBA-project
pip install -r requirements.txt
jupyter notebook
```

Then open the main notebook and run the cells in order.

## Future Improvements
incorporate player-tracking and shot-quality data
add injury and lineup context
include defensive matchup features
improve model calibration and feature engineering
expand visualizations and interactive dashboards

## Course Context
This project was developed as a team course project focused on applying data science and machine learning methods to real sports analytics questions.
