import streamlit as st
from Predictor import load_and_train
from App import run_interface


@st.cache_resource
def init_app():
    # Memorizza risultati per non riaddestrare il modello ad ogni click
    return load_and_train()


result = init_app()

if result[0] is None:
    st.error(f"Errore nel caricamento dati:\n{result[3]}")
else:
    model, encoder, teams, last_stats = result

    # Passa al programma i dati delle squadre già elaborati per poterli usare nell'interfaccia
    run_interface(model, encoder, teams, last_stats)