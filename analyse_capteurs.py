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

      # === Visualisation 4 : Présentes vs Manquantes (Barplot empilé) ===
if "Présentes" in df_freq_summary.columns and "Manquantes" in df_freq_summary.columns:
    df_stacked = df_freq_summary[["Capteur", "Présentes", "Manquantes"]].copy()
    df_stacked.set_index("Capteur", inplace=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    df_stacked.plot(kind='bar', stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
    plt.axhline(total, color="grey", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nombre de points")
    plt.title(f"Données présentes vs manquantes – fréquence {freq}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
