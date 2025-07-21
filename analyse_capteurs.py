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

    # 🔁 Ajouter la colonne Doublon
    df_resume["Doublon"] = df_resume["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

    # 🔍 Validation si référence disponible
    if capteurs_reference is not None and len(capteurs_reference) > 0:
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

    # 📊 Graphique horizontal
    df_plot = df_resume.sort_values(by="% Présentes", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.25)))
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

    # 🧾 Légende des statuts
    st.markdown("""
    ### 🧾 Légende des statuts :
    - 🟢 : Capteur exploitable (≥ 80 %)
    - 🟠 : Incomplet (entre 1 % et 79 %)
    - 🔴 : Données absentes (0 %)
    """)

    return df_resume  # ✅ Bien indenté dans la fonction


  # 🔁 Vérification des doublons
df_resume["Capteur"] = df_resume["Capteur"].astype(str).str.strip()
df_resume["Doublon"] = df_resume["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

# 🔍 Validation selon référence
if capteurs_reference is not None and len(capteurs_reference) > 0:
    capteurs_reference_cleaned = {c.strip() for c in capteurs_reference}

    df_resume["Dans la référence"] = df_resume["Capteur"].apply(
        lambda capteur: "✅ Oui" if capteur in capteurs_reference_cleaned else "❌ Non"
    )

    st.subheader("📋 Validation des capteurs analysés")
    st.markdown("""
    ### 🧾 Légende des colonnes :
    - ✅ : Présence confirmée dans la référence  
    - ❌ : Absent de la référence  
    - 🔁 : Capteur dupliqué dans le fichier principal
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


# --- Analyse de complétude sans rééchantillonnage ---
def analyser_completude(df):
    if "timestamp" not in df.columns:
        st.error("❌ La colonne 'timestamp' est manquante.")
        return pd.DataFrame()

    total = len(df)
    resultat = []

    for col in df.select_dtypes(include="number").columns:
        presente = df[col].notna().sum()
        manquantes = total - presente
        pct_presente = 100 * presente / total if total > 0 else 0
        pct_manquantes = 100 - pct_presente
        statut = "🟢" if pct_presente >= 80 else ("🟠" if pct_presente > 0 else "🔴")

        resultat.append({
            "Capteur": col.strip(),
            "Présentes": int(presente),
            "% Présentes": round(pct_presente, 2),
            "Manquantes": int(manquantes),
            "% Manquantes": round(pct_manquantes, 2),
            "Statut": statut
        })

    return pd.DataFrame(resultat)

# --- Traitement principal ---
st.subheader("📂 Fichier principal : Analyse brute (sans rééchantillonnage)")
df_main = charger_excel(main_file)  # 💡 suppose une fonction de chargement sans resampling

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
    st.warning("⚠️ Aucun fichier de comparaison n'a été téléversé.")

# --- Analyse simplifiée avec validation
df_simple = analyse_simplifiee(df_main, capteurs_reference)

# --- Analyse brute sans rééchantillonnage ---
st.subheader("📈 Analyse de complétude des données brutes")
stats_main = analyser_completude(df_main)
st.dataframe(stats_main, use_container_width=True)

# 🧾 Légende des statuts
st.markdown("""
### 🧾 Légende des statuts :
- 🟢 : Capteur exploitable (≥ 80 %)
- 🟠 : Incomplet (entre 1 % et 79 %)
- 🔴 : Données absentes (0 %)
""")

# 🔢 Résumé par statut
count_vert = stats_main["Statut"].value_counts().get("🟢", 0)
count_orange = stats_main["Statut"].value_counts().get("🟠", 0)
count_rouge = stats_main["Statut"].value_counts().get("🔴", 0)

st.markdown(f"""
**Résumé des capteurs :**
- ✔️ Capteurs exploitables (🟢) : `{count_vert}`
- ⚠️ Capteurs incomplets (🟠) : `{count_orange}`
- ❌ Capteurs vides (🔴) : `{count_rouge}`
""")

# 📊 Graphique horizontal
df_plot = stats_main.sort_values(by="% Présentes", ascending=True)
fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.25)))
sns.barplot(
    data=df_plot,
    y="Capteur",
    x="% Présentes",
    hue="Statut",
    dodge=False,
    palette={"🟢": "green", "🟠": "orange", "🔴": "red"},
    ax=ax
)
plt.title("Complétude des capteurs - Fichier brut", fontsize=14)
plt.xlabel("% Données présentes")
plt.ylabel("Capteur")
plt.xlim(0, 100)
plt.tight_layout()
st.pyplot(fig)



# ✅ Export final
st.subheader("📤 Export des résultats")
csv = df_simple.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger le rapport (CSV)", csv, file_name="rapport_capteurs.csv", mime="text/csv")

