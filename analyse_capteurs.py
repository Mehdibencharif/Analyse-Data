import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta
import re
import unicodedata

from pandas.api.types import is_numeric_dtype

def coerce_numeric_general(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    Force les colonnes majoritairement numériques en float.
    Les placeholders ou textes parasites deviennent NaN.
    """
    for col in df.columns:
        if col.lower() in ("timestamp", "notes"):
            continue
        s = df[col]
        if not is_numeric_dtype(s):
            s2 = s.astype(str).str.replace(",", ".", regex=False).str.strip()
            s2 = s2.replace(list(PLACEHOLDER_NULLS), pd.NA)
            numeric = pd.to_numeric(s2, errors="coerce")
            # si la majorité est numérique, on conserve
            if numeric.notna().mean() >= threshold:
                df[col] = numeric
    return df
    
# Valeurs considérées comme "vides" ou "nulles"
PLACEHOLDER_NULLS = {"", " ", "-", "—", "–", "NA", "N/A", "na", "n/a", "null", "None"}

def series_with_true_nans(s: pd.Series) -> pd.Series:
    """Transforme les placeholders en vrais NaN pour bien compter les manquants."""
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s.replace(list(PLACEHOLDER_NULLS), pd.NA)
        s = s.replace(r"^\s+$", pd.NA, regex=True)
    return s

# Motif pour reconnaître les colonnes de température
TEMP_NAME_RE = re.compile(r"(?i)(temp|temperature|°\s*c|degc|degre|°c|\[°c\])")

def coerce_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes de température en float (NaN si non numérique)."""
    for col in df.columns:
        if col.lower() in ("timestamp", "notes"):
            continue
        name = str(col)
        if TEMP_NAME_RE.search(name):
            s = df[col]
            if s.dtype == object:
                s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
                s = s.replace(list(PLACEHOLDER_NULLS), pd.NA)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df

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

# 🧼 Conversion des colonnes de température en numérique
df_main = coerce_temperature_columns(df_main)
df_main = coerce_numeric_general(df_main)  # 🔧 force toutes les colonnes majoritairement numériques

# -------- Nettoyage des noms de capteurs (pour la comparaison uniquement) --------
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

# 📑 Lecture du fichier de comparaison (capteurs attendus) + versions nettoyées
df_compare = None
capteurs_reference = None
capteurs_reference_cleaned = None

if compare_file:
    try:
        df_compare = pd.read_excel(compare_file)
        if "Description" not in df_compare.columns:
            st.error("❌ Le fichier de comparaison doit contenir une colonne 'Description'.")
            st.stop()

        # Ensemble brut + ensemble nettoyé
        df_compare["Description"] = df_compare["Description"].astype(str).str.strip()
        capteurs_reference = set(df_compare["Description"])
        capteurs_reference_cleaned = {nettoyer_nom_capteur(c) for c in capteurs_reference}

        st.success("✅ Fichier de comparaison chargé avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier de comparaison : {e}")
        st.stop()
else:
    st.info("ℹ️ Aucun fichier de comparaison n'a été téléversé (facultatif).")

# --- Analyse simple ---
def analyse_simplifiee(df):
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
            "Capteur": str(col).strip(),
            "Présentes": int(presente),
            "% Présentes": round(pct_presente, 2),
            "Manquantes": int(manquantes),
            "% Manquantes": round(pct_manquantes, 2),
            "Statut": statut
        })

    df_resume = pd.DataFrame(resume)

    # Ajout d'une colonne nettoyée pour les comparaisons (sans affecter les noms affichés)
    df_resume["Nom_nettoye"] = df_resume["Capteur"].astype(str).apply(nettoyer_nom_capteur)

    # Affichage tableau
    st.dataframe(df_resume, use_container_width=True)
    return df_resume

# 📊 Analyse simple (toujours basée sur df_main, non rééchantillonné)
df_simple = analyse_simplifiee(df_main)

# 🔁 Nettoyage et vérification des doublons
df_simple["Capteur"] = df_simple["Capteur"].astype(str).str.strip()
df_simple["Doublon"] = df_simple["Capteur"].duplicated(keep=False) \
    .map({True: "🔁 Oui", False: "✅ Non"})
# 🧹 Suppression des doublons basés sur le nom nettoyé (on garde la dernière version, la plus "propre")
df_simple = df_simple.drop_duplicates(subset=["Nom_nettoye"], keep="last").reset_index(drop=True)

# 🔍 Validation selon la référence (si fournie)
if capteurs_reference_cleaned and len(capteurs_reference_cleaned) > 0:
    # 1) Indication si le capteur figure dans la référence nettoyée
    df_simple["Dans la référence"] = df_simple["Nom_nettoye"].isin(capteurs_reference_cleaned) \
        .map({True: "✅ Oui", False: "❌ Non"})

    # 2) Tri : capteurs validés (✅) d’abord
    df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

    # 3) Affichages séparés
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

    # 4) Liste brute des noms non reconnus (utile pour copier/coller)
    if not df_non_valides.empty:
        st.subheader("Liste brute – Capteurs du fichier principal absents de la référence")
        st.write(df_non_valides["Capteur"].tolist())

    # 5) Capteurs attendus mais absents dans les données analysées
    capteurs_trouves = set(df_simple["Nom_nettoye"])
    manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
    if manquants:
        st.subheader("Capteurs attendus non trouvés dans les données analysées")
        st.markdown("Voici les capteurs présents dans le fichier de référence mais absents du fichier principal :")
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"])
        st.dataframe(df_manquants, use_container_width=True)
    else:
        st.markdown("✅ Tous les capteurs attendus sont présents dans les données.")      

# --- Analyse de complétude AVEC rééchantillonnage qui conserve toutes les colonnes ---
from pandas.api.types import is_numeric_dtype

def resampler_df(df, frequence_str):
    if "timestamp" not in df.columns:
        st.warning("⚠️ Colonne 'timestamp' non trouvée dans le fichier.")
        return df

    st.info(f"⏱️ Fréquence sélectionnée : {frequence_str}")

    # Pas de rééchantillonnage demandé
    if frequence_str == "1min":
        st.info("✅ Pas de rééchantillonnage nécessaire (1min).")
        return df.copy()

    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

        # Plan d'agrégation colonne par colonne :
        #  - numérique  -> mean
        #  - non num.   -> first (pour conserver la colonne et permettre l'analyse de complétude)
        agg_map = {}
        for col in df.columns:
            if col.lower() in ("notes",):   # on laisse vivre si tu as une colonne notes
                agg_map[col] = "first"
            else:
                agg_map[col] = "mean" if is_numeric_dtype(df[col]) else "first"

        df_resampled = df.resample(rule_map[frequence_str]).agg(agg_map).reset_index()
        st.success(f"✅ Données rééchantillonnées avec succès à {frequence_str}.")
        return df_resampled

    except Exception as e:
        st.error(f"❌ Erreur lors du rééchantillonnage : {e}")
        return df.reset_index()

# --- Analyse de complétude (par colonne, numérique ou non) ---
def analyser_completude(df):
    if "timestamp" not in df.columns:
        st.error("❌ La colonne 'timestamp' est manquante.")
        return pd.DataFrame()

    total = len(df)
    resultat = []

    # On analyse TOUTES les colonnes (sauf timestamp et notes)
    cols = [c for c in df.columns if c.lower() not in ("timestamp", "notes")]

    for col in cols:
        # présence = entrées non nulles (valable aussi pour objets/texte)
        presente = int(df[col].notna().sum())
        manquantes = int(total - presente)
        pct_presente = 100 * presente / total if total > 0 else 0.0
        pct_manquantes = 100 - pct_presente

        # statut visuel
        if pct_presente >= 80:
            statut = "🟢"
        elif pct_presente > 0:
            statut = "🟠"
        else:
            statut = "🔴"

        resultat.append({
            "Capteur": str(col).strip(),
            "Présentes": presente,
            "% Présentes": round(pct_presente, 2),
            "Manquantes": manquantes,
            "% Manquantes": round(pct_manquantes, 2),
            "Statut": statut
        })

    return pd.DataFrame(resultat)

# 📈 Analyse de complétude avec la fréquence choisie
st.subheader(f"📈 Analyse de complétude des données brutes ({frequence})")
df_resample = resampler_df(df_main, frequence)
stats_main = analyser_completude(df_resample)
st.dataframe(stats_main, use_container_width=True)
# 🧹 Suppression des doublons sur la colonne Capteur (ex: Énergie-Froid apparaît 2 fois)
stats_main = stats_main.drop_duplicates(subset=["Capteur"], keep="last").reset_index(drop=True)

# 📘 Légende des statuts (cohérente avec les seuils ci-dessus)
st.markdown("""
### 🧾 Légende des statuts :
- 🟢 : Capteur exploitable (≥ 80 % de valeurs présentes)
- 🟠 : Incomplet (entre 1 % et 79 %)
- 🔴 : Données absentes (0 %)
""")

# 📌 Résumé numérique des capteurs selon statut
count_vert = stats_main["Statut"].value_counts().get("🟢", 0)
count_orange = stats_main["Statut"].value_counts().get("🟠", 0)
count_rouge = stats_main["Statut"].value_counts().get("🔴", 0)
st.markdown(f"""
**Résumé des capteurs :**
- Capteurs exploitables (🟢) : `{count_vert}`
- Capteurs incomplets (🟠) : `{count_orange}`
- Capteurs vides (🔴) : `{count_rouge}`
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
plt.title("Complétude des capteurs", fontsize=14)
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













