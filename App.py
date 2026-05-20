import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from Predictor import predict_match

def run_interface(model, encoder, teams, last_stats):
    """Sviluppa l'interfaccia grafica partendo dai dati del modello già addestrato."""
    st.set_page_config(page_title="Football Predictor PRO", page_icon="⚽", layout="centered")

    # Modifiche grafiche per nascondere i menu di sistema e pulire l'interfaccia
    st.markdown("""
        <style>
        #MainMenu, header, footer, .stDeployButton, [data-testid="stHeader"] {
            visibility: hidden !important;
            display: none !important;
        }
        .block-container {
            max-width: 900px !important;
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        div[data-testid="stMetricSimpleValue"] {
            font-size: 28px !important;
            font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚽ Match Predictor PRO")

    st.subheader("Configurazione Partita")

    col_team1, col_team2 = st.columns(2)
    with col_team1:
        t1 = st.selectbox("Squadra in Casa:", teams, index=0)
    with col_team2:
        t2 = st.selectbox("Squadra in Trasferta:", teams, index=min(1, len(teams) - 1))

    predict_btn = st.button("PREDICI RISULTATO", use_container_width=True)

    if predict_btn:
        if t1 == t2:
            st.warning("Seleziona due squadre differenti per effettuare il calcolo!")
        else:
            # Recupera i calcoli dal file Predictor.py quando l'utente clicca sul bottone
            p_h, p_d, p_a, l_h, l_a, top_scores, h_dist, a_dist = predict_match(model, encoder, last_stats, t1, t2)

            st.markdown("---")

            # Divide la schermata in due macro-colonne per organizzare i risultati
            main_col1, main_col2 = st.columns([1, 1], gap="large")

            # Colonna Sinistra: Mostra le percentuali di vittoria (1X2) con le barre grafiche
            with main_col1:
                st.subheader("🔮 Probabilità Esito")

                st.write(f"*Casa ({t1}):* {p_h * 100:.1f}%")
                st.progress(float(p_h))

                st.write(f"*Pareggio:* {p_d * 100:.1f}%")
                st.progress(float(p_d))

                st.write(f"*Trasferta ({t2}):* {p_a * 100:.1f}%")
                st.progress(float(p_a))

            # Colonna Destra: Mostra i tre risultati esatti migliori e la media dei gol attesi
            with main_col2:
                st.subheader("📊 Risultati Esatti")

                c1, c2, c3 = st.columns(3)
                c1.metric("1° Opzione", top_scores[0])
                c2.metric("2° Opzione", top_scores[1])
                c3.metric("3° Opzione", top_scores[2])

                st.markdown("<br>", unsafe_allow_html=True)

                st.info(f"Gol attesi **{t1}**: {l_h:.2f}\n\nGol attesi **{t2}**: {l_a:.2f}")

            # st.markdown("---")
            # st.subheader("📈 Grafico di Confronto dei Gol Attesi")

            # # Imposta le dimensioni e la struttura delle barre
            # fig, ax = plt.subplots(figsize=(10, 4.5))
            # x = np.arange(6)
            # width = 0.35

            # # Genera le barre colorate per ciascuna squadra
            # ax.bar(x - width/2, [v * 100 for v in h_dist], width, label=t1, color='#1f77b4')
            # ax.bar(x + width/2, [v * 100 for v in a_dist], width, label=t2, color='#ff7f0e')

            # # Personalizza le etichette del grafico e mostra la griglia
            # ax.set_ylabel('Probabilità (%)', fontsize=11)
            # ax.set_xticks(x)
            # ax.set_xticklabels([f'{i} Gol' for i in x])
            # ax.legend(fontsize=10)
            # ax.grid(axis='y', linestyle='--', alpha=0.5)

            # # Mostra il grafico finale sulla pagina Streamlit
            # st.pyplot(fig)