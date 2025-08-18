import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

# ------------- Configuration de la page Streamlit -------------
st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données ")

# ------------- Paramètres de fréquence d'analyse -------------
st.sidebar.header("Paramètres d'analyse")
frequence = st.sidebar.selectbox(
    "Choisissez la fréquence d'analyse :",
    ["1min", "5min", "10min", "15min", "1H"]
)
rule_map = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "1H": "1H"
}

# ------------- Téléversement des fichiers -------------
st.sidebar.subheader("Téléversement des fichiers")
main_file = st.sidebar.file_uploader(
    "📂 Fichier principal (obligatoire)",
    type=[".xlsx", ".xls", ".xlsm"],
    key="main"
)
compare_file = st.sidebar.file_uploader(
    "📂 Fichier de comparaison (facultatif)",
    type=[".xlsx", ".xls", ".xlsm"],
    key="compare"
)

# ------------- Fonction de chargement de fichier -------------
def charger_et_resampler(fichier, nom_fichier):
    xls = pd.ExcelFile(fichier)
    feuille = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.selectbox(
        f"📄 Feuille à utiliser pour {nom_fichier}",
        xls.sheet_names,
        key=nom_fichier
    )
    df = pd.read_excel(xls, sheet_name=feuille)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

# ------------- Vérification de la présence du fichier principal -------------
if not main_file:
    st.warning("⚠️ Veuillez téléverser un fichier principal pour démarrer l’analyse.")
    st.stop()

# 📥 Chargement du fichier principal
df_main = charger_et_resampler(main_file, "Fichier principal")

import re

def nettoyer_nom_capteur(nom: str) -> str:
    """
    Supprime les unités entre crochets [] ou parenthèses () et les espaces inutiles.
    Exemples :
      'Temp-1 [°C]'      -> 'Temp-1'
      'Débit (Gpm)'      -> 'Débit'
      'Pression [bar] '  -> 'Pression'
    """
    s = str(nom)
    s = re.sub(r"\s*[\[\(].*?[\]\)]", "", s)  # enlève [ ... ] ou ( ... )
    return s.strip()

# 🧼 Copie "nettoyée" des colonnes du fichier principal (SEULEMENT pour la comparaison)
df_main_cleaned = df_main.copy()
df_main_cleaned.columns = [
    "timestamp" if c == "timestamp" else nettoyer_nom_capteur(c)
    for c in df_main_cleaned.columns
]

# 🧼 Ensemble de référence nettoyé (si un fichier de référence est fourni)
capteurs_reference_cleaned = None
if compare_file:
    capteurs_reference_cleaned = {
        nettoyer_nom_capteur(c)
        for c in df_compare["Description"].astype(str)
    }


# 📑 Lecture du fichier de comparaison (capteurs attendus)
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

    # Affichage tableau
    st.dataframe(df_resume, use_container_width=True)

    # Graphique horizontal
    #df_plot = df_resume.sort_values(by="% Présentes", ascending=True)
    #fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.25)))
    #sns.barplot(
    #    data=df_plot,
    #    y="Capteur",
    #    x="% Présentes",
    #    hue="Statut",
    #    dodge=False,
    #    palette={"🟢": "green", "🟠": "orange", "🔴": "red"},
    #    ax=ax
    #)
    #plt.title("Pourcentage de données présentes par capteur", fontsize=14)
    #plt.xlabel("% Présentes")
    #plt.ylabel("Capteur")
    #plt.xlim(0, 100)
    #plt.tight_layout()
    #st.pyplot(fig)

    return df_resume

# 📊 Analyse simple avec validation
df_simple = analyse_simplifiee(df_main, capteurs_reference)

# 🔁 Nettoyage et vérification des doublons
df_simple["Capteur"] = df_simple["Capteur"].astype(str).str.strip()
df_simple["Doublon"] = df_simple["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

# 🔍 Validation selon la référence (si fournie)
if capteurs_reference_cleaned and len(capteurs_reference_cleaned) > 0:
    # 1) Ajoute une colonne "Nom_nettoye" dans le récapitulatif simple
    df_simple["Nom_nettoye"] = (
        df_simple["Capteur"]
        .astype(str)
        .apply(nettoyer_nom_capteur)
    )

    # 2) Indique si le capteur figure dans la référence nettoyée
    df_simple["Dans la référence"] = df_simple["Nom_nettoye"].isin(capteurs_reference_cleaned) \
        .map({True: "✅ Oui", False: "❌ Non"})

    # 3) Tri : capteurs validés (✅) d’abord
    df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

    # 4) Affichages séparés
    st.subheader("✅ Capteurs trouvés dans la référence")
    df_valides = df_simple[df_simple["Dans la référence"] == "✅ Oui"]
    if not df_valides.empty:
        st.dataframe(df_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.markdown("Aucun capteur valide trouvé.")

    st.subheader("❌ Capteurs absents de la référence")
    df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]
    if not df_non_valides.empty:
        st.dataframe(df_non_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.markdown("Tous les capteurs sont présents dans la référence.")

    # 5) Liste brute des noms non reconnus (utile pour un copier/coller)
    if not df_non_valides.empty:
        st.subheader("Liste brute – Capteurs du fichier principal absents de la référence")
        st.write(df_non_valides["Capteur"].tolist())

    # 6) Capteurs attendus mais absents dans les données analysées
    capteurs_trouves = set(df_simple["Nom_nettoye"])
    manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
    if manquants:
        st.subheader("Capteurs attendus non trouvés dans les données analysées")
        st.markdown("Voici les capteurs présents dans le fichier de référence mais absents du fichier principal :")
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"])
        st.dataframe(df_manquants, use_container_width=True)
    else:
        st.markdown("✅ Tous les capteurs attendus sont présents dans les données.")

# --- Analyse de complétude sans rééchantillonnage ---
def resampler_df(df, frequence_str):
    if "timestamp" not in df.columns:
        st.warning("⚠️ Colonne 'timestamp' non trouvée dans le fichier.")
        return df

    # Affiche la fréquence choisie pour débogage
    st.info(f"⏱️ Fréquence sélectionnée : {frequence_str}")

    # Si fréquence est 1min (pas besoin de rééchantillonnage)
    if frequence_str == "1min":
        st.info("✅ Pas de rééchantillonnage nécessaire (1min).")
        return df.copy()

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")

        # Garde uniquement les colonnes numériques pour la moyenne
        df_numeric = df.select_dtypes(include="number")

        # Rééchantillonnage avec moyenne
        df_resampled = df_numeric.resample(rule_map[frequence_str]).mean().reset_index()

        st.success(f"✅ Données rééchantillonnées avec succès à {frequence_str}.")

        return df_resampled

    except Exception as e:
        st.error(f"❌ Erreur lors du rééchantillonnage : {e}")
        return df.reset_index()

# --- Analyse de complétude ---
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

# 📈 Analyse de complétude avec rééchantillonnage à la fréquence choisie
st.subheader(f"📈 Analyse de complétude des données brutes ({frequence})")
df_resample = resampler_df(df_main, frequence)
stats_main = analyser_completude(df_resample)
st.dataframe(stats_main, use_container_width=True)

# 📘 Légende des statuts
st.markdown("""
### 🧾 Légende des statuts :
- 🟢 : Capteur exploitable (≥ 80 %)
- 🟠 : Incomplet (entre 1 % et 79 %)
- 🔴 : Données absentes (0 %)
""")

# 📌 Résumé numérique des capteurs selon statut
count_vert = stats_main["Statut"].value_counts().get("🟢", 0)
count_orange = stats_main["Statut"].value_counts().get("🟠", 0)
count_rouge = stats_main["Statut"].value_counts().get("🔴", 0)
st.markdown(f"""
**Résumé des capteurs :**
-  Capteurs exploitables (🟢) : `{count_vert}`
-  Capteurs incomplets (🟠) : `{count_orange}`
-  Capteurs vides (🔴) : `{count_rouge}`
""")


# 📉 Graphique horizontal final
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

# ✅ Export Excel final avec couleurs
st.subheader("📤 Export des résultats (Excel)")

from io import BytesIO

output = BytesIO()

with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    # Écriture des feuilles
    df_simple.to_excel(writer, index=False, sheet_name="Résumé capteurs")
    stats_main.to_excel(writer, index=False, sheet_name="Complétude brute")

    if 'df_non_valides' in locals() and not df_non_valides.empty:
        df_non_valides.to_excel(writer, index=False, sheet_name="Capteurs non reconnus")

    if 'df_manquants' in locals() and not df_manquants.empty:
        df_manquants.to_excel(writer, index=False, sheet_name="Capteurs manquants")

    workbook  = writer.book

    #  Format couleur selon le statut
    format_vert = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_orange = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})
    format_rouge = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    # Appliquer le format à la feuille "Résumé capteurs"
    feuille = writer.sheets["Résumé capteurs"]
    statut_col = df_simple.columns.get_loc("Statut")  # colonne Statut

    # Appliquer la mise en forme conditionnelle à la colonne Statut
    feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
        'type':     'text',
        'criteria': 'containing',
        'value':    '🟢',
        'format':   format_vert
    })
    feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
        'type':     'text',
        'criteria': 'containing',
        'value':    '🟠',
        'format':   format_orange
    })
    feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
        'type':     'text',
        'criteria': 'containing',
        'value':    '🔴',
        'format':   format_rouge
    })

    #writer.save()

# Bouton de téléchargement
st.download_button(
    label="📥 Télécharger le rapport Excel ",
    data=output.getvalue(),
    file_name="rapport_capteurs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)





