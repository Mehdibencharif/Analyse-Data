import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers Excel à analyser", type=[".xlsx", ".xls"], accept_multiple_files=True)

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

            # Fréquence réelle détectée
            freq_median = df["timestamp"].diff().median().total_seconds() / 60
            st.markdown(f"**Fréquence médiane d'échantillonnage :** {freq_median:.2f} minutes")

            # Resample automatique selon fréquence détectée
            detected_freq = f"{int(round(freq_median))}min"
            df_time = df.set_index("timestamp")
            df_resampled = df_time.resample(detected_freq).mean(numeric_only=True)

            total = len(df_resampled)
            missing_summary = []

            for col in df_resampled.columns:
                if col.lower() in ["notes"]:
                    continue
                valides = df_resampled[col].notna().sum()
                missing = total - valides
                pct_missing = 100 * missing / total

                missing_summary.append({
                    "Capteur": col,
                    "Présentes": valides,
                    "Manquantes": missing,
                    "% Manquantes": round(pct_missing, 2)
                })

            df_missing_summary = pd.DataFrame(missing_summary).sort_values("% Manquantes", ascending=False)

            st.subheader(f"Résumé des données manquantes – fréquence détectée : {detected_freq}")
            st.dataframe(df_missing_summary)

            # Barplot des données manquantes
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.barplot(data=df_missing_summary, x="Capteur", y="% Manquantes", ax=ax, palette="coolwarm")
            plt.axhline(seuil_manquantes, color='red', linestyle='--', label=f'Seuil critique {seuil_manquantes}%')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("% de données manquantes")
            plt.title("Pourcentage de données manquantes par capteur")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # Graphique Présentes vs Manquantes (empilé)
            df_stacked = df_missing_summary.set_index("Capteur")[["Présentes", "Manquantes"]]
            fig, ax = plt.subplots(figsize=(14, 6))
            df_stacked.plot(kind='bar', stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
            plt.axhline(df_stacked.sum(axis=1).max(), color="grey", linestyle="--", linewidth=0.8)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Nombre de données")
            plt.title(f"Données présentes vs manquantes – fréquence {detected_freq}")
            plt.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {str(e)}")
