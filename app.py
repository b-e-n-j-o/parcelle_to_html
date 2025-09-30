import streamlit as st
import subprocess
import re
import json

# ======================
# TON APP DE BASE ICI
# ======================
st.title("Kerelia – Tableau de bord")
st.write("⚙️ Ici toutes tes fonctionnalités habituelles : upload CERFA, cartes, etc.")

# Séparateur visuel
st.divider()
st.subheader("📊 Suivi intersections (temps réel)")

# Charger ton catalogue JSON pour compter les couches
with open("catalogue_layers.json", "r") as f:
    catalogue = json.load(f)

total_couches = len(catalogue.keys())

progress_bar = st.progress(0)
status_text = st.empty()
log_box = st.empty()

# Bouton pour lancer le script
if st.button("🚀 Lancer intersections"):
    SCRIPT_PATH = "pipeline_intersections.py"

    process = subprocess.Popen(
        ["python3", SCRIPT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    couches_done = 0
    logs = []

    for line in process.stdout:
        logs.append(line.strip())
        log_box.text("\n".join(logs[-20:]))  # dernières lignes de logs

        if "→ Couche" in line:
            couches_done += 1
            progress_bar.progress(couches_done / total_couches)
            status_text.text(f"⚡ {couches_done}/{total_couches} couches traitées")

    process.wait()

    if process.returncode == 0:
        st.success("🎉 Intersections terminées avec succès !")
    else:
        st.error(f"❌ Erreur (code {process.returncode})")
