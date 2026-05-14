import tkinter as tk
from tkinter import ttk, messagebox
from Predictor import predict_match

def start_gui(model, encoder, teams, last_stats):
    root = tk.Tk()
    root.title("Football Predictor PRO")
    root.geometry("500x700")
    root.configure(bg="#f0f0f0", padx=30, pady=20)

    tk.Label(root, text="⚽ Match Predictor PRO", font=("Helvetica", 20, "bold"), bg="#f0f0f0").pack(pady=20)

    # Selezione Squadre
    tk.Label(root, text="Squadra Casa:", bg="#f0f0f0").pack()
    h_cb = ttk.Combobox(root, values=teams, width=45, state="readonly", font=("Helvetica", 10))
    h_cb.pack(pady=5); h_cb.current(0)

    tk.Label(root, text="Squadra Trasferta:", bg="#f0f0f0").pack()
    a_cb = ttk.Combobox(root, values=teams, width=45, state="readonly", font=("Helvetica", 10))
    a_cb.pack(pady=5); a_cb.current(1)

    # Bottone
    tk.Button(root, text="PREDICI RISULTATO", command=lambda: predict(),
              bg="#2e7d32", fg="white", font=("Helvetica", 12, "bold"),
              pady=15, cursor="hand2").pack(fill="x", pady=25)

    # Box Risultati
    res_box = tk.Frame(root, bg="white", highlightbackground="#cccccc", highlightthickness=1, pady=20)
    res_box.pack(pady=10, fill="x")

    tk.Label(res_box, text="RISULTATI PIÙ PROBABILI", font=("Helvetica", 9, "bold"), bg="white", fg="#888888").pack()

    # Label per i 3 punteggi (Senza scritte sotto)
    score_labels = []
    for i in range(3):
        lbl = tk.Label(res_box, text="-", font=("Helvetica", 22, "bold"), bg="white", fg="#1a1a1a")
        if i == 0: lbl.config(font=("Helvetica", 42, "bold")) # Il primo è più grande
        lbl.pack(pady=5)
        score_labels.append(lbl)

    # Statistiche in fondo
    prob_lab = tk.Label(root, text="", font=("Courier New", 10, "bold"), bg="#f0f0f0")
    prob_lab.pack(side="bottom", pady=20)

    def predict():
        t1, t2 = h_cb.get(), a_cb.get()
        if t1 == t2:
            messagebox.showwarning("Attenzione", "Seleziona squadre diverse")
            return

        res = predict_match(model, encoder, last_stats, t1, t2)

        # Aggiorna i 3 punteggi
        for i, score in enumerate(res['top_scores']):
            score_labels[i].config(text=score)
            if i > 0: score_labels[i].config(fg="#777777") # Secondari più chiari

        # Aggiorna stringa statistiche con nomi reali
        prob_text = f"{t1}: {res['prob_home']:.1f}% | Pareggio: {res['prob_draw']:.1f}% | {t2}: {res['prob_away']:.1f}%"
        prob_lab.config(text=prob_text)

    root.mainloop()