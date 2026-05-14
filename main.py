from Model import load_and_train
from Gui import start_gui
import tkinter as tk
from tkinter import messagebox

# --- ENTRY POINT ---
def main():
    print("⚽ Football Predictor PRO - Avvio in corso...")
    print("   Caricamento dataset e training modello con pesi temporali...")

    # Carichiamo il modello (restituisce 4 valori come definito nel nuovo Model.py)
    result = load_and_train()

    # Controllo se il caricamento è andato a buon fine
    # result[0] è il modello, result[2] è la lista squadre
    if result[0] is None:
        root = tk.Tk()
        root.withdraw()
        # Il quarto elemento (indice 3) contiene il messaggio d'errore
        messagebox.showerror("Errore Caricamento", f"Dati non compatibili:\n{result[3]}")
        return

    # Estraiamo i dati dal risultato
    model, encoder, teams, last_stats = result

    print(f"   ✓ Modello addestrato con successo su {len(teams)} squadre.")
    print("   ✓ Focus impostato sui dati recenti (2020-2025).")
    print("   ✓ Apertura interfaccia grafica...\n")

    # Avviamo la GUI
    start_gui(model, encoder, teams, last_stats)

if __name__ == "__main__":
    main()