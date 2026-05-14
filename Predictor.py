import pandas as pd
from scipy.stats import poisson

def get_exact_score(h_elo, a_elo, h_avg, a_avg):
    elo_diff = (h_elo - a_elo) / 400
    l_h = max(0.8, h_avg + (elo_diff * 0.5))
    l_a = max(0.6, a_avg - (elo_diff * 0.5))

    scores = []
    for h in range(5):
        for a in range(5):
            prob = poisson.pmf(h, l_h) * poisson.pmf(a, l_a)
            scores.append({'score': f"{h} - {a}", 'p': prob})

    # Ordina per probabilità e prendi i primi 3
    top_3 = sorted(scores, key=lambda x: x['p'], reverse=True)[:3]
    return [s['score'] for s in top_3]

def predict_match(model, encoder, last_stats, t1, t2):
    data = pd.DataFrame([[
        encoder.transform([t1])[0], encoder.transform([t2])[0],
        last_stats[t1]['Elo'], last_stats[t2]['Elo'],
        last_stats[t1]['Form3'], last_stats[t1]['Form5'],
        last_stats[t2]['Form3'], last_stats[t2]['Form5']
    ]], columns=['HomeTeam_Enc', 'AwayTeam_Enc', 'HomeElo', 'AwayElo',
                 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away'])

    probs = model.predict_proba(data)[0]
    classes = list(model.classes_)

    # Ottieni i top 3 risultati
    top_results = get_exact_score(
        last_stats[t1]['Elo'], last_stats[t2]['Elo'],
        last_stats[t1]['AvgGoals'], last_stats[t2]['AvgGoals']
    )

    return {
        'prob_home': probs[classes.index('H')] * 100,
        'prob_draw': probs[classes.index('D')] * 100,
        'prob_away': probs[classes.index('A')] * 100,
        'top_scores': top_results
    }