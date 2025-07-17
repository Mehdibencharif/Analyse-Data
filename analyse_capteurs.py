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
    return df

# --- Vérification du fichier principal ---
if not main_file:
    st.warning("📁 Veuillez téléverser un fichier principal pour commencer l’analyse.")
    st.stop()
    
# --- Analyse simple ---
def analyse_simplifiee(df):
    st.subheader("📌 Présentes vs Manquantes – Méthode simple")
    total = len(df)
    resume = []
    for col in df.columns:
        if col.lower() in ['timestamp', 'notes']:
            continue
        presente = df[col].notna().sum()
        pct = 100 * presente / total if total > 0 else 0
        statut = "🟢" if pct == 100 else ("🟠" if pct > 0 else "🔴")
        resume.append({"Capteur": col, "Présentes": presente, "% Présentes": round(pct, 2), "Statut": statut})
    df_resume = pd.DataFrame(resume)
    st.dataframe(df_resume, use_container_width=True)

    # Graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    df_resume.set_index("Capteur")["% Présentes"].plot(kind="bar", ax=ax, color="skyblue")
    plt.ylabel("% Présentes")
    plt.title("Pourcentage de données présentes par capteur")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # 🔍 Vérification : est-ce que chaque capteur analysé est dans la référence ?
    if capteurs_reference is not None:
        df_resume["Dans la référence"] = df_resume["Capteur"].apply(
            lambda capteur: "✅ Oui" if capteur in capteurs_reference else "❌ Non"
        )
        st.subheader("📋 Validation des capteurs analysés")
        st.dataframe(df_resume[["Capteur", "Dans la référence"]], use_container_width=True)
    return df_resume

# --- Analyse complète ---
def analyser_completude(df):
    df = df.set_index("timestamp").resample(rule_map[frequence]).mean()
    total = len(df)
    resultat = []
    for col in df.columns:
        presentes = df[col].notna().sum()
        pct = 100 * presentes / total if total > 0 else 0
        statut = "🟢" if pct == 100 else ("🟠" if pct > 0 else "🔴")
        resultat.append({"Capteur": col, "% Données présentes": round(pct, 2), "Statut": statut})
    return pd.DataFrame(resultat)

# --- Traitement ---
if main_file:
    st.subheader("📂 Fichier principal : Analyse simplifiée")
    df_main = charger_et_resampler(main_file, "Fichier principal")
    df_simple = analyse_simplifiee(df_main)

    st.subheader("📈 Analyse rééchantillonnée selon la fréquence choisie")
    stats_main = analyser_completude(df_main.reset_index())
    st.dataframe(stats_main, use_container_width=True)

    # Graphique
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=stats_main, x="Capteur", y="% Données présentes", hue="Statut", dodge=False, palette={"🟢": "green", "🟠": "orange", "🔴": "red"}, ax=ax1)
    plt.title("Complétude des capteurs - Fichier principal")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig1)
# ----- Comparaison 
if compare_file:
    st.subheader("🔁 Comparaison avec un deuxième fichier")
    df_compare = charger_et_resampler(compare_file, "Fichier comparaison")

    if "timestamp" not in df_compare.columns:
        st.error("Le fichier de comparaison ne contient pas de colonne 'timestamp'.")
        st.stop()

    try:
        df_compare["timestamp"] = pd.to_datetime(df_compare["timestamp"], errors="coerce")
        df_compare = df_compare.dropna(subset=["timestamp"])
        df_compare_numeric = df_compare.select_dtypes(include="number")
        df_compare_resample = df_compare_numeric.set_index(df_compare["timestamp"]).resample(rule_map[frequence]).mean().reset_index()
        df_compare_resample["timestamp"] = pd.to_datetime(df_compare_resample["timestamp"])
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier de comparaison : {str(e)}")
        st.stop()

    stats_compare = analyser_completude(df_compare_resample)

    df_merged = pd.merge(stats_main, stats_compare, on="Capteur", how="outer", suffixes=(" (Principal)", " (Comparaison)"))
    df_merged = df_merged.fillna({"% Données présentes (Principal)": 0, "% Données présentes (Comparaison)": 0})

    def statut_global(row):
        if row['% Données présentes (Principal)'] == 0 and row['% Données présentes (Comparaison)'] == 0:
            return "🔴"
        elif row['% Données présentes (Principal)'] == 100 and row['% Données présentes (Comparaison)'] == 100:
            return "🟢"
        else:
            return "🟠"

    df_merged["Statut global"] = df_merged.apply(statut_global, axis=1)
    st.dataframe(df_merged, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df_melted = df_merged.melt(
        id_vars=["Capteur", "Statut global"],
        value_vars=["% Données présentes (Principal)", "% Données présentes (Comparaison)"],
        var_name="Fichier", value_name="% Présentes"
    )
    sns.barplot(data=df_melted, x="Capteur", y="% Présentes", hue="Fichier", ax=ax2)
    plt.title("Comparaison de complétude par capteur")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig2)

    # 🔍 Vérifier les capteurs manquants
    capteurs_main = set(stats_main["Capteur"])
    capteurs_compare = set(stats_compare["Capteur"])
    capteurs_manquants = capteurs_main.symmetric_difference(capteurs_compare)

    if capteurs_manquants:
        st.warning(f"⚠️ Capteurs non communs entre les deux fichiers : {', '.join(capteurs_manquants)}")
    else:
        st.success("✅ Tous les capteurs sont présents dans les deux fichiers.")

    export_df = df_merged

else:
    export_df = stats_main
    st.info("Veuillez téléverser un deuxième fichier si vous souhaitez effectuer une comparaison.")

# ✅ Export final (toujours affiché si fichier principal analysé)
st.subheader("📤 Export des résultats")
csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger le rapport (CSV)", csv, file_name="rapport_capteurs.csv", mime="text/csv")

