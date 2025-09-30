# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import tempfile
from pathlib import Path

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
