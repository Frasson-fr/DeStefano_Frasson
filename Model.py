import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


def load_and_train():
    try:
        df = pd.read_csv("data/Matches.csv", low_memory=False)

        # Mappatura colonne
        rename_dict = {'FTHome': 'FTHG', 'FTAway': 'FTAG', 'FTResult': 'FTResult'}
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

        # Filtro leghe inglesi
        df = df[df['Division'].isin(['E0', 'E1', 'E2'])].copy()
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTResult'])

        # Gestione Date
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        df = df.sort_values('MatchDate')

        # Encoding squadre
        encoder = LabelEncoder()
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        encoder.fit(all_teams)

        df_train = df.copy()
        df_train['HomeTeam_Enc'] = encoder.transform(df['HomeTeam'])
        df_train['AwayTeam_Enc'] = encoder.transform(df['AwayTeam'])

        # Feature Selection
        for col in ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']:
            if col not in df_train.columns:
                df_train[col] = 1500 if 'Elo' in col else 0

        # --- LOGICA PESI TEMPORALI ---
        ref_date = df_train['MatchDate'].max()
        df_train['Days_Diff'] = (ref_date - df_train['MatchDate']).dt.days
        # Pesi: le partite recenti pesano vicino a 1.0, quelle del 2000 pesano circa 0.2
        sample_weights = np.exp(-df_train['Days_Diff'] / 5000)

        features = ['HomeTeam_Enc', 'AwayTeam_Enc', 'HomeElo', 'AwayElo',
                    'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
        X = df_train[features]
        y = df_train['FTResult']

        # Training Random Forest pesato
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        model.fit(X, y, sample_weight=sample_weights)

        # Statistiche per Poisson e GUI
        last_stats = {}
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
            goals_made = (team_matches[team_matches['HomeTeam'] == team]['FTHG'].sum() +
                          team_matches[team_matches['AwayTeam'] == team]['FTAG'].sum()) / 10

            last_m = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].iloc[-1]

            last_stats[team] = {
                'Elo': last_m['HomeElo'] if last_m['HomeTeam'] == team else last_m['AwayElo'],
                'AvgGoals': goals_made if not np.isnan(goals_made) else 1.2,
                'Form3': last_m['Form3Home'] if last_m['HomeTeam'] == team else last_m['Form3Away'],
                'Form5': last_m['Form5Home'] if last_m['HomeTeam'] == team else last_m['Form5Away']
            }

        return model, encoder, sorted(all_teams), last_stats

    except Exception as e:
        return None, None, [], str(e)