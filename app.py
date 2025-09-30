# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import tempfile
from pathlib import Path
import subprocess
import re
import json

from intersections import run_intersections
from interpretation import load_catalog, render_html

# =========================
# Config de base (chemins relatifs au projet)
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "CONFIG" / "v_commune_2025.csv"
MAPPING_PATH = BASE_DIR / "CONFIG" / "nouveau_catalogue_29_09.json"

st.set_page_config("Analyse parcelle")
st.title("🔎 Analyse parcelle")

# =========================
# Partie haute : Analyse parcelle (ton app existante)
# =========================
parcel = st.text_input("Référence parcellaire (ex: AD 0598)", "")

if st.button("Lancer l’analyse") and parcel:
    with st.spinner("Calcul des intersections en cours..."):
        # Fichiers temporaires
        tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name

        # 1. Lancer intersections
        report = run_intersections(
            commune="Latresne",
            departement="33",
            parcels=parcel,
            csv=str(CSV_PATH),
            mapping=str(MAPPING_PATH),
            out_json=tmp_json,
        )

        # 2. Charger catalogue + interprétation HTML
        catalog = load_catalog(str(MAPPING_PATH))
        html = render_html(report, catalog)

        # 3. Sauver dans un fichier temporaire
        tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        with open(tmp_html, "w", encoding="utf-8") as f:
            f.write(html)

    st.success("✅ Rapport généré")
    with open(tmp_html, "rb") as f:
        st.download_button("📥 Télécharger le rapport HTML", f, file_name="rapport_parcelle.html")

# =========================
# Partie basse : Suivi intersections en temps réel
# =========================
st.divider()
st.subheader("📊 Suivi intersections (temps réel)")

progress_bar = st.progress(0)
status_text = st.empty()
log_box = st.empty()

if st.button("🚀 Lancer toutes les intersections (streaming logs)"):
    # Charger le catalogue pour compter les couches
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
    total_couches = len(catalogue.keys())

    SCRIPT_PATH = BASE_DIR / "pipeline_intersections.py"

    process = subprocess.Popen(
        ["python3", str(SCRIPT_PATH)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    couches_done = 0
    logs = []

    for line in process.stdout:
        logs.append(line.strip())
        # Affiche les 20 dernières lignes de logs
        log_box.text("\n".join(logs[-20:]))

        # Détecter quand une nouvelle couche démarre
        if re.search(r"→ Couche", line):
            couches_done += 1
            progress_bar.progress(couches_done / total_couches)
            status_text.text(f"⚡ {couches_done}/{total_couches} couches traitées")

    process.wait()

    if process.returncode == 0:
        st.success("🎉 Intersections terminées avec succès !")
    else:
        st.error(f"❌ Erreur (code {process.returncode})")
