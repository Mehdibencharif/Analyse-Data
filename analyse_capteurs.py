import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

# Upload des fichiers
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers Excel à analyser", type=[".xlsx", ".xls"], accept_multiple_files=True)

# Slider de seuil critique
seuil_manquantes = st.slider("Seuil critique de données manquantes (%)", 0, 100, 30)

if uploaded_files:
    for file in uploaded_files:
        st.header(f"Fichier : {file.name}")

        try:
            # Lecture des feuilles
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            sheet_to_use = st.selectbox(f"Choisissez une feuille pour {file.name}", sheet_names, key=file.name)

            # Chargement des données
            df = pd.read_excel(xls, sheet_name=sheet_to_use)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            # Fréquence médiane
            freq_median = df["timestamp"].diff().median().total_seconds() / 60
            st.markdown(f"**Fréquence médiane d'échantillonnage :** {freq_median:.2f} minutes")

            # Test des différentes fréquences
            frequences_test = ['1min', '5min', '10min', '15min', '1h', '1D']
            df_time = df.set_index("timestamp")
            comparaison = {}

            for freq in frequences_test:
                df_resampled = df_time.resample(freq).mean(numeric_only=True)
                stats_freq = {}
                for col in df_resampled.columns:
                    if col.lower() == 'notes':
                        continue
                    total = len(df_resampled)
                    valides = df_resampled[col].notna().sum()
                    pct_missing = 100 * (total - valides) / total
                    stats_freq[col] = round(pct_missing, 2)
                comparaison[freq] = stats_freq

            # Création du DataFrame final
            df_comparaison = pd.DataFrame(comparaison).T
            df_comparaison.index.name = "Fréquence"

            # Affichage tableau
            st.subheader("Pourcentage de données manquantes selon la fréquence")
            st.dataframe(df_comparaison.style.background_gradient(cmap="coolwarm", axis=None))

            # Affichage heatmap
            fig, ax = plt.subplots(figsize=(16, 8))
            sns.heatmap(
                df_comparaison,
                annot=True,
                fmt=".2f",
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

            # Affichage barplot vertical des données manquantes à la fréquence 1min (comme Excel)
            st.subheader("📉 % de Données Manquantes par Variable (fréquence 1min)")
            if '1min' in df_comparaison.index:
                data_1min = df_comparaison.loc['1min']
                fig2, ax2 = plt.subplots(figsize=(14, 6))
                data_1min.sort_values(ascending=False).plot(kind='bar', color='steelblue', ax=ax2)
                plt.ylabel("% Manquantes")
                plt.xlabel("Variables")
                plt.title("% de Données Manquantes par Variable (fréquence 1min)")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Erreur avec le fichier {file.name} : {e}")
