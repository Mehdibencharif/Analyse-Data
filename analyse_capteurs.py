import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Streamlit
st.set_page_config(page_title="Analyse données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs – % manquants selon fréquence attendue")

uploaded_files = st.file_uploader("Téléversez un ou plusieurs fichiers Excel", type=[".xlsx", ".xls"], accept_multiple_files=True)
frequence = st.selectbox("Fréquence attendue", ["1min", "5min", "10min", "15min", "1h"], index=0)

if uploaded_files:
    for file in uploaded_files:
        st.header(f"Fichier : {file.name}")
        try:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox(f"Choisissez une feuille pour {file.name}", xls.sheet_names, key=file.name)
            df = pd.read_excel(xls, sheet_name=sheet)

            # Nettoyage
            df.columns = [str(c).strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            # Définir grille temporelle complète
            start = df["timestamp"].min()
            end = df["timestamp"].max()
            full_index = pd.date_range(start=start, end=end, freq=frequence)
            df_full = df.set_index("timestamp").reindex(full_index)

            # Calcul pour chaque capteur
            resultats = []
            total_attendu = len(full_index)

            for col in df_full.columns:
                if col.lower() in ["notes"]:
                    continue
                valides = df_full[col].notna().sum()
                pct_pres = 100 * valides / total_attendu
                pct_manq = 100 - pct_pres
                resultats.append({
                    "Capteur": col,
                    "Présentes": valides,
                    "Attendues": total_attendu,
                    "% Manquantes": round(pct_manq, 2),
                    "% Présentes": round(pct_pres, 2)
                })

            df_resultats = pd.DataFrame(resultats)

            st.subheader("📋 Résumé des % de données manquantes")
            st.dataframe(df_resultats)

            st.subheader("📉 Graphique % Manquantes par Capteur")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=df_resultats, x="Capteur", y="% Manquantes", color="crimson")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("% de données manquantes")
            plt.title(f"Données manquantes par capteur – fréquence {frequence}")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {e}")
