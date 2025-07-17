import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

# --- Fréquence d'analyse ---
st.sidebar.header("Paramètres d'analyse")
frequence = st.sidebar.selectbox("Choisissez la fréquence d'analyse :", ["1min", "5min", "10min", "15min", "1H"])
rule_map = {"1min": "1min", "5min": "5min", "10min": "10min", "15min": "15min", "1H": "1H"}

# --- Fichiers à téléverser ---
st.sidebar.subheader("Téléversement des fichiers")
main_file = st.sidebar.file_uploader("Fichier principal (obligatoire)", type=[".xlsx", ".xls", ".xlsm"], key="main")
compare_file = st.sidebar.file_uploader("Fichier de comparaison (facultatif)", type=[".xlsx", ".xls", ".xlsm"], key="compare")

# --- Fonction d'importation et prétraitement ---
def charger_et_resampler(fichier, nom):
    xls = pd.ExcelFile(fichier)
    feuille = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.selectbox(f"Feuille à utiliser pour {nom}", xls.sheet_names, key=nom)
    df = pd.read_excel(xls, sheet_name=feuille)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample(rule_map[frequence]).mean()
    return df

# --- Analyse de complétude ---
def analyser_completude(df):
    total = len(df)
    resultat = []
    for col in df.columns:
        presentes = df[col].notna().sum()
        pct = 100 * presentes / total if total > 0 else 0
        statut = "🟢" if pct == 100 else ("🟠" if pct > 0 else "🔴")
        resultat.append({"Capteur": col, "% Données présentes": round(pct, 2), "Statut": statut})
    return pd.DataFrame(resultat)

# --- Affichage ---
if main_file:
    st.subheader("📂 Analyse du fichier principal")
    df_main = charger_et_resampler(main_file, "Fichier principal")
    stats_main = analyser_completude(df_main)
    st.dataframe(stats_main, use_container_width=True)

    # Graphique
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=stats_main, x="Capteur", y="% Données présentes", hue="Statut", dodge=False, palette={"🟢": "green", "🟠": "orange", "🔴": "red"}, ax=ax1)
    plt.title("Complétude des capteurs - Fichier principal")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- Comparaison ---
    if compare_file:
        st.subheader("🔁 Comparaison avec un deuxième fichier")
        df_compare = charger_et_resampler(compare_file, "Fichier comparaison")
        stats_compare = analyser_completude(df_compare)

        df_merged = pd.merge(stats_main, stats_compare, on="Capteur", how="outer", suffixes=(" (Principal)", " (Comparaison)"))
        df_merged = df_merged.fillna({"% Données présentes (Principal)": 0, "% Données présentes (Comparaison)": 0})

        # Recalculer le statut général
        def statut_global(row):
            if row['% Données présentes (Principal)'] == 0 and row['% Données présentes (Comparaison)'] == 0:
                return "🔴"
            elif row['% Données présentes (Principal)'] == 100 and row['% Données présentes (Comparaison)'] == 100:
                return "🟢"
            else:
                return "🟠"

        df_merged["Statut global"] = df_merged.apply(statut_global, axis=1)
        st.dataframe(df_merged, use_container_width=True)

        # Graphique comparatif
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        df_melted = df_merged.melt(id_vars=["Capteur", "Statut global"],
                                   value_vars=["% Données présentes (Principal)", "% Données présentes (Comparaison)"],
                                   var_name="Fichier", value_name="% Présentes")
        sns.barplot(data=df_melted, x="Capteur", y="% Présentes", hue="Fichier", ax=ax2)
        plt.title("Comparaison de complétude par capteur")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig2)

    # --- Export CSV ---
    st.subheader("📤 Export des résultats")
    export_df = df_merged if compare_file else stats_main
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger le rapport (CSV)", csv, file_name="rapport_capteurs.csv", mime="text/csv")

else:
    st.info("Veuillez téléverser au minimum un fichier pour lancer l'analyse.")
