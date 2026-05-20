import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Elenco delle informazioni che l'IA userà per studiare le squadre
FEATURES = [
    'HomeTeam_Enc', 'AwayTeam_Enc', 'HomeElo', 'AwayElo', 'EloDiff',
    'H_ScoredHome', 'H_ConcededHome', 'A_ScoredAway', 'A_ConcededAway',
    'HomeForm', 'AwayForm'
]


def load_and_train():
    """Carica il dataset, esegue il feature engineering e addestra il modello."""
    try:
        df = pd.read_csv("data/Matches.csv", low_memory=False).rename(columns={'FTHome': 'FTHG', 'FTAway': 'FTAG'})
        df = df[df['Division'].isin(['E0', 'E1', 'E2'])].dropna(
            subset=['HomeTeam', 'AwayTeam', 'FTResult', 'FTHG', 'FTAG'])

        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        df = df.sort_values('MatchDate').reset_index(drop=True)

        encoder = LabelEncoder()
        all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
        encoder.fit(all_teams)

        df['HomeTeam_Enc'], df['AwayTeam_Enc'] = encoder.transform(df['HomeTeam']), encoder.transform(df['AwayTeam'])
        df['EloDiff'] = df.get('HomeElo', 1500) - df.get('AwayElo', 1500)

        # Calcola quanti gol fanno/subiscono in media le squadre nelle ultime 5 partite
        df['H_ScoredHome'] = df.groupby('HomeTeam')['FTHG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.2)
        df['H_ConcededHome'] = df.groupby('HomeTeam')['FTAG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.2)
        df['A_ScoredAway'] = df.groupby('AwayTeam')['FTAG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)
        df['A_ConcededAway'] = df.groupby('AwayTeam')['FTHG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.2)

        df['HomePts'] = df['FTResult'].map({'H': 3, 'D': 1, 'A': 0})
        df['AwayPts'] = df['FTResult'].map({'A': 3, 'D': 1, 'H': 0})

        df['HomeForm'] = df.groupby('HomeTeam')['HomePts'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)
        df['AwayForm'] = df.groupby('AwayTeam')['AwayPts'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)

        # Importanza del tempo: fa pesare di più i match recenti e meno quelli vecchi
        days_diff = (df['MatchDate'].max() - df['MatchDate']).dt.days.fillna(9999)
        weights = np.exp(-days_diff / 5000)

        model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(df[FEATURES], df['FTResult'], sample_weight=weights)

        last_stats = {}
        for t in all_teams:
            team_matches = df[(df['HomeTeam'] == t) | (df['AwayTeam'] == t)].sort_values('MatchDate')
            last_match = team_matches.iloc[-1]

            is_home = last_match['HomeTeam'] == t
            current_elo = last_match['HomeElo'] if is_home else last_match['AwayElo']

            pts_list = []
            for _, row in team_matches.tail(5).iterrows():
                if row['HomeTeam'] == t:
                    pts_list.append(3 if row['FTResult'] == 'H' else (1 if row['FTResult'] == 'D' else 0))
                else:
                    pts_list.append(3 if row['FTResult'] == 'A' else (1 if row['FTResult'] == 'D' else 0))

            form_mean = np.mean(pts_list) if pts_list else 1.0

            last_stats[t] = {
                'Elo': current_elo if pd.notna(current_elo) else 1500,
                'h_scored': df[df['HomeTeam'] == t]['FTHG'].tail(5).mean() or 1.2,
                'h_conceded': df[df['HomeTeam'] == t]['FTAG'].tail(5).mean() or 1.2,
                'a_scored': df[df['AwayTeam'] == t]['FTAG'].tail(5).mean() or 1.0,
                'a_conceded': df[df['AwayTeam'] == t]['FTHG'].tail(5).mean() or 1.2,
                'form': form_mean
            }

        return model, encoder, all_teams, last_stats
    except Exception as e:
        return None, None, [], str(e)


def predict_match(model, encoder, s, t1, t2):
    """Prende due squadre e calcola le probabilità 1X2 e i 3 punteggi esatti coerenti."""
    data = pd.DataFrame([[
        encoder.transform([t1])[0], encoder.transform([t2])[0],
        s[t1]['Elo'], s[t2]['Elo'], s[t1]['Elo'] - s[t2]['Elo'],
        s[t1]['h_scored'], s[t1]['h_conceded'], s[t2]['a_scored'], s[t2]['a_conceded'],
        s[t1]['form'], s[t2]['form']
    ]], columns=FEATURES)

    # Chiede all'IA una prima percentuale di stima su chi vincerà la partita
    probs = model.predict_proba(data)[0]
    classes = list(model.classes_)
    raw_h = probs[classes.index('H')] if 'H' in classes else 0.34
    raw_a = probs[classes.index('A')] if 'A' in classes else 0.33

    # Unisce la media dei gol storici con le percentuali dell'IA per trovare i gol attesi della partita
    l_h = max(0.4, ((s[t1]['h_scored'] + s[t2]['a_conceded']) / 2) * (1 + (raw_h - raw_a) * 0.2))
    l_a = max(0.3, ((s[t2]['a_scored'] + s[t1]['h_conceded']) / 2) * (1 - (raw_h - raw_a) * 0.2))

    scores = []
    p_h, p_d, p_a = 0.0, 0.0, 0.0

    # Calcola la probabilità di ogni singolo risultato esatto (da 0-0 fino a 6-6)
    for h in range(7):
        for a in range(7):
            p_cell = poisson.pmf(h, l_h) * poisson.pmf(a, l_a)
            scores.append({'score': f"{h} - {a}", 'p': p_cell})

            # Somma le percentuali dei risultati per decidere se vince la casa, il pareggio o l'ospite
            if h > a:
                p_h += p_cell
            elif h == a:
                p_d += p_cell
            else:
                p_a += p_cell

    # Aggiusta i decimali per fare in modo che la somma totale delle tre percentuali sia esattamente 100%
    total_p = p_h + p_d + p_a
    p_h, p_d, p_a = p_h / total_p, p_d / total_p, p_a / total_p

    top_scores = [sc['score'] for sc in sorted(scores, key=lambda x: x['p'], reverse=True)[:3]]

    home_goals_dist = [poisson.pmf(i, l_h) for i in range(6)]
    away_goals_dist = [poisson.pmf(i, l_a) for i in range(6)]

    # Adesso restituiamo TUTTI gli 8 elementi richiesti da app.py
    return p_h, p_d, p_a, l_h, l_a, top_scores, home_goals_dist, away_goals_dist