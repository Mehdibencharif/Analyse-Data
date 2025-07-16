import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers Excel à analyser", type=[".xlsx", ".xls"], accept_multiple_files=True)

st.markdown("## Paramètres d'analyse")
mode_frequence = st.radio("Méthode de calcul de fréquence attendue :", ["Automatique (fréquence médiane)", "Manuelle"], index=0)

if mode_frequence == "Manuelle":
    freq_attendue = st.selectbox("Fréquence attendue pour le calcul des données manquantes :", ["1min", "5min", "10min", "15min", "1h", "1D"], index=1)
else:
    freq_attendue = None  # Sera calculée automatiquement

seuil_manquantes = st.slider("Seuil critique de données manquantes (%)", 0, 100, 30)

if uploaded_files:
    for file in uploaded_files:
        st.header(f"Fichier : {file.name}")

        try:
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            sheet_to_use = st.selectbox(f"Choisissez une feuille pour {file.name}", sheet_names, key=file.name)

            df = pd.read_excel(xls, sheet_name=sheet_to_use)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            df_time = df.set_index("timestamp")

            if mode_frequence == "Automatique":
                freq_median = df["timestamp"].diff().median().total_seconds() / 60
                st.markdown(f"**Fréquence médiane détectée :** {freq_median:.2f} minutes")
                if freq_median < 2:
                    freq_attendue = "1min"
                elif freq_median < 7:
                    freq_attendue = "5min"
                elif freq_median < 12:
                    freq_attendue = "10min"
                elif freq_median < 20:
                    freq_attendue = "15min"
                elif freq_median < 90:
                    freq_attendue = "1h"
                else:
                    freq_attendue = "1D"
                st.markdown(f"**Fréquence utilisée pour l'analyse :** `{freq_attendue}`")

            df_resampled = df_time.resample(freq_attendue).mean(numeric_only=True)
            total = len(df_resampled)

            summary_freq = []
            for col in df_resampled.columns:
                if col.lower() in ['notes']:
                    continue
                serie = df_resampled[col].dropna()
                non_na = len(serie)
                missing = total - non_na

                summary_freq.append({
                    "Capteur": col,
                    "Présentes": non_na,
                    "Manquantes": missing,
                    "% Manquantes": round(100 * missing / total if total else 0, 2)
                })

            df_freq_summary = pd.DataFrame(summary_freq)
            st.subheader("Pourcentage de données manquantes")
            st.dataframe(df_freq_summary.style.background_gradient(cmap="coolwarm", subset=["% Manquantes"]))

            # Barplot empilé
            st.subheader("Données présentes vs manquantes")
            df_stacked = df_freq_summary.set_index("Capteur")[["Présentes", "Manquantes"]]
            fig, ax = plt.subplots(figsize=(14, 6))
            df_stacked.plot(kind='bar', stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
            plt.axhline(df_stacked.sum(axis=1).max(), color="grey", linestyle="--", linewidth=0.8)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Nombre de données")
            plt.title(f"Présentes vs manquantes – fréquence {freq_attendue}")
            plt.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {str(e)}")
