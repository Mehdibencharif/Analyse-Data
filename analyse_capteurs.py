import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

uploaded_files = st.file_uploader(
    "Choisissez un ou plusieurs fichiers Excel à analyser",
    type=[".xlsx", ".xls"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.header(f"Fichier : {file.name}")

        try:
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            sheet_to_use = st.selectbox(
                f"Choisissez une feuille pour {file.name}",
                sheet_names,
                key=file.name
            )

            df = pd.read_excel(xls, sheet_name=sheet_to_use)
            df.columns = [str(c).strip() for c in df.columns]

            # Détection automatique de la colonne timestamp
            df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            st.markdown(f"**Nombre total de lignes :** {len(df)}")
            st.markdown(f"**Période couverte :** {df['timestamp'].min()} ➡ {df['timestamp'].max()}")

            # === Méthode simple Présentes vs Manquantes ===
            st.subheader("📌 Présentes vs Manquantes – Méthode simple (pas de resampling)")

            summary_simple = []
            nb_total_lignes = len(df)

            for col in df.columns:
                if col.lower() in ['timestamp', 'notes']:
                    continue

                nb_presente = df[col].notna().sum()
                nb_manquante = nb_total_lignes - nb_presente

                pct_presente = 100 * nb_presente / nb_total_lignes
                pct_manquante = 100 - pct_presente

                summary_simple.append({
                    "Capteur": col,
                    "Présentes": nb_presente,
                    "% Présentes": round(pct_presente, 2),
                    "Manquantes": nb_manquante,
                    "% Manquantes": round(pct_manquante, 2),
                })

            df_simple = pd.DataFrame(summary_simple)

            st.dataframe(df_simple)

            # Graphique empilé
            fig, ax = plt.subplots(figsize=(14, 6))
            df_simple.set_index("Capteur")[["% Présentes", "% Manquantes"]].plot(
                kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"]
            )
            plt.ylabel("%")
            plt.title("Pourcentage de données présentes et manquantes par capteur")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {str(e)}")
