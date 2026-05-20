import streamlit as st
from Predictor import load_and_train
from App import run_interface


@st.cache_resource
def init_app():
    # Sfrutta la cache di Streamlit per non riaddestrare il modello ad ogni click dell'utente
    return load_and_train()


result = init_app()

if result[0] is None:
    st.error(f"Errore nel caricamento dati:\n{result[3]}")
else:
    model, encoder, teams, last_stats = result

    # Passa il modello IA e lo snapshot delle statistiche all'interfaccia grafica
    run_interface(model, encoder, teams, last_stats)