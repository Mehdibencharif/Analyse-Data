import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

#-------------Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("Analyse de données capteurs")


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
    st.subheader("Présentes vs Manquantes – Méthode simple")
    total = len(df)
    resume = []

    for col in df.columns:
        if col.lower() in ['timestamp', 'notes']:
            continue

        presente = df[col].notna().sum()
        manquantes = total - presente
        pct_presente = 100 * presente / total if total > 0 else 0
        pct_manquantes = 100 - pct_presente
        statut = "🟢" if pct_presente >= 80 else ("🟠" if pct_presente > 0 else "🔴")

        resume.append({
            "Capteur": col.strip(),
            "Présentes": presente,
            "% Présentes": round(pct_presente, 2),
            "Manquantes": manquantes,
            "% Manquantes": round(pct_manquantes, 2),
            "Statut": statut
        })

    df_resume = pd.DataFrame(resume)

    # 💡 Affichage du tableau
    st.dataframe(df_resume, use_container_width=True)

    # 📊 Graphique horizontal trié
    df_plot = df_resume.sort_values(by="% Présentes", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.25)))  # Hauteur dynamique
    sns.barplot(
        data=df_plot,
        y="Capteur",
        x="% Présentes",
        hue="Statut",
        dodge=False,
        palette={"🟢": "green", "🟠": "orange", "🔴": "red"},
        ax=ax
    )
    plt.title("Pourcentage de données présentes par capteur", fontsize=14)
    plt.xlabel("% Présentes")
    plt.ylabel("Capteur")
    plt.xlim(0, 100)
    plt.tight_layout()
    st.pyplot(fig)

    return df_resume
 

    # 🔁 Ajouter la colonne Doublon
    df_resume["Doublon"] = df_resume["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

    # 🔍 Validation des capteurs
    if capteurs_reference is not None and len(capteurs_reference) > 0:
        # Nettoyage
        df_resume["Capteur"] = df_resume["Capteur"].astype(str).str.strip()
        capteurs_reference_cleaned = {c.strip() for c in capteurs_reference}

        df_resume["Dans la référence"] = df_resume["Capteur"].apply(
            lambda capteur: "✅ Oui" if capteur in capteurs_reference_cleaned else "❌ Non"
        )

        st.subheader("📋 Validation des capteurs analysés")
        st.markdown("""
        ### 🧾 Légende des colonnes :
        - ✅ : Présence / Unicité confirmée  
        - ❌ : Capteur non trouvé dans la référence  
        - 🔁 : Capteur dupliqué
        """)
        st.dataframe(df_resume[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)

        # 🔎 Capteurs attendus mais absents
        capteurs_trouves = set(df_resume["Capteur"])
        manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
        if manquants:
            st.subheader("📌 Capteurs attendus non trouvés dans les données analysées")
            st.markdown("Voici les capteurs présents dans le fichier de référence mais absents du fichier principal :")
            st.write(manquants)
        else:
            st.markdown("✅ Tous les capteurs attendus sont présents dans les données.")
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
        presente = df[col].notna().sum()
        manquantes = total - presente
        pct_presente = 100 * presente / total if total > 0 else 0
        pct_manquantes = 100 - pct_presente
        statut = "🟢" if pct_presente >= 80 else ("🟠" if pct_presente > 0 else "🔴")

        resultat.append({
            "Capteur": col.strip(),
            "Présentes": presente,
            "% Présentes": round(pct_presente, 2),
            "Manquantes": manquantes,
            "% Manquantes": round(pct_manquantes, 2),
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
stats_main = analyser_completude(df_main.reset_index())  # 👈 définie ici
st.dataframe(stats_main, use_container_width=True)
fig, ax = plt.subplots(figsize=(10, max(6, len(stats_main) * 0.25)))  # Hauteur dynamique

df_plot = stats_main.sort_values(by="% Présentes", ascending=True)  # Tri du moins au plus complet

sns.barplot(
    data=df_plot,
    y="Capteur",               # ✅ Capteurs sur l'axe vertical
    x="% Présentes",           # ✅ Pourcentage sur l'axe horizontal
    hue="Statut",
    dodge=False,
    palette={"🟢": "green", "🟠": "orange", "🔴": "red"},
    ax=ax
)

plt.title("Complétude des capteurs - Fichier principal", fontsize=14)
plt.xlabel("% Données présentes")
plt.ylabel("Capteur")
plt.xlim(0, 100)
plt.tight_layout()
st.pyplot(fig)


# ✅ Export final
st.subheader("📤 Export des résultats")
csv = df_simple.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger le rapport (CSV)", csv, file_name="rapport_capteurs.csv", mime="text/csv")

