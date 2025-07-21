import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

#-------------Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")


# --- Paramètres Fréquence d'analyse ---
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
def analyse_simplifiee(df, capteurs_reference=None):
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
    
     # 🔁 Ajouter la colonne Doublon (capteurs dupliqués dans le tableau)
    df_resume["Doublon"] = df_resume["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

    # 🔍 Vérification : est-ce que chaque capteur est dans la référence ?
    if capteurs_reference is not None and len(capteurs_reference) > 0:
        df_resume["Dans la référence"] = df_resume["Capteur"].apply(
            lambda capteur: "✅ Oui" if capteur in capteurs_reference else "❌ Non"
        )
        st.subheader("📋 Validation des capteurs analysés")
        st.markdown("""
        ### 🧾 Légende des colonnes :
        - ✅ : Présence / Unicité confirmée  
        - ❌ : Capteur non trouvé dans la référence  
        - 🔁 : Capteur dupliqué
        """)
        st.dataframe(df_resume[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.subheader("📋 Validation des capteurs analysés")
        st.markdown("⚠️ Aucune référence fournie. Affichage des doublons uniquement.")
        st.dataframe(df_resume[["Capteur", "Doublon"]], use_container_width=True)

    return df_resume

# --- Analyse complète : rééchantillonnage temporel et complétude ---
def analyser_completude(df):
    if "timestamp" not in df.columns:
        st.error("❌ La colonne 'timestamp' est manquante pour effectuer le rééchantillonnage.")
        return pd.DataFrame()

    df = df.set_index("timestamp").resample(rule_map[frequence]).mean()

    total = len(df)
    resultat = []

    for col in df.columns:
        presentes = df[col].notna().sum()
        pct = 100 * presentes / total if total > 0 else 0
        statut = "🟢" if pct == 100 else ("🟠" if pct > 0 else "🔴")
        resultat.append({
            "Capteur": col,
            "% Données présentes": round(pct, 2),
            "Statut": statut
        })

    return pd.DataFrame(resultat)
    

# --- Traitement principal ---
st.subheader("📂 Fichier principal : Analyse simplifiée")
df_main = charger_et_resampler(main_file, "Fichier principal")

# --- Lecture de la liste de capteurs attendus (si fichier de comparaison fourni) ---
capteurs_reference = None
if compare_file:
    try:
        df_compare = pd.read_excel(compare_file)
        capteurs_reference = set(df_compare["Description"].astype(str).str.strip())
        st.success("✅ Fichier de comparaison chargé avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier de comparaison : {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Aucun fichier de comparaison n'a été téléversé. La validation ne sera pas effectuée.")
    capteurs_reference = set()



# --- Analyse simplifiée avec ou sans validation
df_simple = analyse_simplifiee(df_main, capteurs_reference)

# --- Analyse rééchantillonnée selon la fréquence choisie ---
st.subheader("📈 Analyse rééchantillonnée selon la fréquence choisie")
stats_main = analyser_completude(df_main.reset_index())
st.dataframe(stats_main, use_container_width=True)

# --- Graphique de complétude par capteur ---
fig1, ax1 = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=stats_main,
    x="Capteur",
    y="% Données présentes",
    hue="Statut",
    dodge=False,
    palette={"🟢": "green", "🟠": "orange", "🔴": "red"},
    ax=ax1
)
plt.title("Complétude des capteurs - Fichier principal")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.tight_layout()
st.pyplot(fig1)


# ✅ Export final
st.subheader("📤 Export des résultats")
csv = df_simple.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger le rapport (CSV)", csv, file_name="rapport_capteurs.csv", mime="text/csv")

