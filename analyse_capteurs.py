import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers Excel à analyser", type=[".xlsx", ".xls"], accept_multiple_files=True)

seuil_manquantes = st.slider("Seuil critique de données manquantes (%)", 0, 100, 30)
frequence_attendue_minutes = st.number_input("Fréquence attendue (en minutes)", min_value=1, max_value=1440, value=1)

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

            freq_median = df["timestamp"].diff().median().total_seconds() / 60
            st.markdown(f"**Fréquence médiane d'échantillonnage :** {freq_median:.2f} minutes")

            frequences_test = ['1min', '5min', '10min', '15min', '1h', '1D']
            df_time = df.set_index("timestamp")
            comparaison = {}

            for freq in frequences_test:
                df_resampled = df_time.resample(freq).mean(numeric_only=True)
                stats_freq = {}

                for col in df_resampled.columns:
                    if col.lower() in ['notes']:
                        continue
                    total = len(df_resampled)
                    valides = df_resampled[col].notna().sum()
                    pct_missing = 100 * (total - valides) / total
                    stats_freq[col] = round(pct_missing, 2)

                comparaison[freq] = stats_freq

            df_comparaison = pd.DataFrame(comparaison).T
            df_comparaison.index.name = "Fréquence"

            st.subheader("Pourcentage de données manquantes selon la fréquence")
            st.dataframe(df_comparaison.style.background_gradient(cmap="coolwarm", axis=None))

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.heatmap(
                df_comparaison,
                annot=True,
                fmt=".1f",
                cmap="coolwarm",
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': '% de données manquantes'}
            )
            plt.title("Pourcentage de données manquantes par capteur selon la fréquence", fontsize=14)
            plt.ylabel("Fréquence")
            plt.xlabel("Capteurs")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)

            # === Présentes vs Manquantes selon grille attendue (version Excel-like) ===
            st.subheader("% de Données Manquantes par Variable (grille théorique)")
            start = df["timestamp"].min()
            end = df["timestamp"].max()
            expected_index = pd.date_range(start=start, end=end, freq=f"{frequence_attendue_minutes}min")
            df_full = df.set_index("timestamp").reindex(expected_index)

            resultats = []
            for col in df_full.columns:
                if col.lower() in ['notes']:
                    continue
                total_theorique = len(df_full)
                valides = df_full[col].notna().sum()
                pct_pres = 100 * valides / total_theorique
                pct_manq = 100 - pct_pres
                resultats.append({
                    "Capteur": col,
                    "% manquantes": round(pct_manq, 2)
                })

            df_resultats_excel = pd.DataFrame(resultats)
            st.dataframe(df_resultats_excel)

            # Graphe barre simple (% manquantes)
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.barplot(data=df_resultats_excel, x="Capteur", y="% manquantes", color="darkblue")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("% Manquant")
            plt.title("% de Données Manquantes par Variable")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {str(e)}")
