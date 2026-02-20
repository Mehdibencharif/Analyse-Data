import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta
import re
import hashlib
import unicodedata
from pandas.api.types import is_numeric_dtype

# ----------------------------- Constantes / placeholders -----------------------------

# Valeurs considérées comme "vides" ou "nulles"
PLACEHOLDER_NULLS = {"", " ", "-", "—", "–", "NA", "N/A", "na", "n/a", "null", "None"}

# Motif pour reconnaître les colonnes de température
TEMP_NAME_RE = re.compile(r"(?i)(temp|temperature|°\s*c|degc|degre|°c|\[°c\])")

# ----------------------------- Utilitaires "qualité data" -----------------------------

def series_with_true_nans(s: pd.Series) -> pd.Series:
    """Transforme les placeholders en vrais NaN pour bien compter les manquants."""
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s.replace(list(PLACEHOLDER_NULLS), pd.NA)
        s = s.replace(r"^\s+$", pd.NA, regex=True)
    return s

def coerce_numeric_general(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    Force les colonnes majoritairement numériques en float.
    Les placeholders ou textes parasites deviennent NaN.
    """
    df = df.copy()
    for col in df.columns:
        if str(col).lower() in ("timestamp", "notes"):
            continue

        s = df[col]
        if not is_numeric_dtype(s):
            s2 = series_with_true_nans(s).astype(str).str.replace(",", ".", regex=False).str.strip()
            numeric = pd.to_numeric(s2, errors="coerce")

            # si la majorité est numérique, on conserve
            if numeric.notna().mean() >= threshold:
                df[col] = numeric

    return df

def coerce_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes de température en float (NaN si non numérique)."""
    df = df.copy()
    for col in df.columns:
        if str(col).lower() in ("timestamp", "notes"):
            continue

        if TEMP_NAME_RE.search(str(col)):
            s = series_with_true_nans(df[col])
            s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
            df[col] = pd.to_numeric(s, errors="coerce")

    return df

def stats_min_max_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule min / max / moyenne pour chaque capteur numérique (hors timestamp/notes).
    Retourne un DataFrame stable, même si df est vide.
    """
    cols_out = ["Capteur", "Min", "Max", "Moyenne", "Nb valeurs", "Nb manquantes", "% présentes"]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=cols_out)

    rows = []
    total = len(df)

    for col in df.columns:
        c = str(col).lower()
        if c in ("timestamp", "notes"):
            continue

        s = series_with_true_nans(df[col])

        # forcer en numérique (si ça ne convertit pas -> tout NaN)
        s_num = pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )

        nb_val = int(s_num.notna().sum())
        nb_nan = int(total - nb_val)
        pct = (100.0 * nb_val / total) if total > 0 else 0.0

        rows.append({
            "Capteur": str(col),
            "Min": float(s_num.min()) if nb_val > 0 else None,
            "Max": float(s_num.max()) if nb_val > 0 else None,
            "Moyenne": float(s_num.mean()) if nb_val > 0 else None,
            "Nb valeurs": nb_val,
            "Nb manquantes": nb_nan,
            "% présentes": round(pct, 2),
        })

    return pd.DataFrame(rows, columns=cols_out)
#----- Bloc 2 -------------#

# ----------------------------- Streamlit : page & paramètres -----------------------------

st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

# =========================
# Paramètres d'analyse
# =========================
st.sidebar.header("Paramètres d'analyse")

frequence = st.sidebar.selectbox(
    "Choisissez la fréquence d'analyse :",
    ["1min", "5min", "10min", "15min", "1H"],
    index=3  # 15 min par défaut (tu peux changer)
)

rule_map = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "1H": "1H"
}

# =========================
# Téléversement des fichiers
# =========================
st.sidebar.subheader("Téléversement des fichiers")

main_file = st.sidebar.file_uploader(
    "📂 Fichier principal (obligatoire)",
    type=["xlsx", "xls", "xlsm"],
    key="main"
)

compare_file = st.sidebar.file_uploader(
    "📂 Fichier de comparaison (facultatif)",
    type=["xlsx", "xls", "xlsm"],
    key="compare"
)

# =========================
# Hash & reset d'état (sélection feuille) si fichier change
# =========================
def file_sha1(uploaded) -> str:
    data = uploaded.getvalue() if uploaded is not None else b""
    return hashlib.sha1(data).hexdigest()[:10]

if main_file is not None:
    st.sidebar.caption(f"Hash fichier principal : `{file_sha1(main_file)}`")
if compare_file is not None:
    st.sidebar.caption(f"Hash fichier comparaison : `{file_sha1(compare_file)}`")

if "last_main_sha1" not in st.session_state:
    st.session_state["last_main_sha1"] = None

curr_sha1 = file_sha1(main_file) if main_file is not None else None

if curr_sha1 is not None and curr_sha1 != st.session_state["last_main_sha1"]:
    # On oublie les sélections de feuilles liées à l'ancien fichier
    for k in list(st.session_state.keys()):
        if str(k).startswith("Fichier principal_sheet_"):
            del st.session_state[k]

    st.session_state["last_main_sha1"] = curr_sha1
    
#----- Bloc 3 -------------#
# ----------------------------- Chargement fichier -----------------------------

def charger_fichier_excel(fichier, nom_fichier: str) -> pd.DataFrame:
    """
    Charge un fichier Excel uploadé (Streamlit) et retourne un DataFrame
    avec une colonne 'timestamp' (colonne 0 renommée), triée et nettoyée.
    """
    raw = fichier.getvalue()
    xls = pd.ExcelFile(BytesIO(raw))

    # clé UI unique (évite les conflits quand on change de fichier)
    sheet_key = f"{nom_fichier}_sheet_{hashlib.sha1(raw).hexdigest()[:8]}"

    # sélection de la feuille si plusieurs
    if len(xls.sheet_names) == 1:
        feuille = xls.sheet_names[0]
    else:
        feuille = st.selectbox(
            f"📄 Feuille à utiliser pour {nom_fichier}",
            xls.sheet_names,
            key=sheet_key
        )

    df = pd.read_excel(xls, sheet_name=feuille)

    # nettoyage des noms de colonnes
    df.columns = [str(c).strip() for c in df.columns]

    # sécurité : au moins 1 colonne
    if df.shape[1] < 1:
        return pd.DataFrame()

    # 1ère colonne = timestamp
    first_col = df.columns[0]
    if str(first_col).lower() != "timestamp":
        df = df.rename(columns={first_col: "timestamp"})

    # parsing timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # nettoyage
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df

# --- Stop si pas de fichier principal
if main_file is None:
    st.warning("⚠️ Veuillez téléverser un fichier principal pour démarrer l’analyse.")
    st.stop()

# ----------------------------- Filtre temporel (sidebar) -----------------------------
st.sidebar.subheader("Filtre temporel (optionnel)")
date_deb = st.sidebar.date_input("Début", value=None)
date_fin = st.sidebar.date_input("Fin", value=None)

def filtrer_periode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    if date_deb is not None:
        df = df[df["timestamp"] >= pd.Timestamp(date_deb)]

    if date_fin is not None:
        # inclut toute la journée de date_fin
        df = df[df["timestamp"] < (pd.Timestamp(date_fin) + pd.Timedelta(days=1))]

    return df

# ----------------------------- Chargement + conversions + filtre -----------------------------

df_main = charger_fichier_excel(main_file, "Fichier principal")

# sécurité : fichier vide ou mal lu
if df_main is None or df_main.empty:
    st.warning("⚠️ Aucune donnée lue dans le fichier (ou feuille vide).")
    st.stop()

# sécurité : timestamp obligatoire
if "timestamp" not in df_main.columns:
    st.error("❌ Colonne 'timestamp' introuvable (la 1ère colonne devrait être un temps/date).")
    st.stop()

# conversions (temp d'abord, puis général)
df_main = coerce_temperature_columns(df_main)
df_main = coerce_numeric_general(df_main)

# re-sécurise timestamp (au cas où)
df_main["timestamp"] = pd.to_datetime(df_main["timestamp"], errors="coerce")
df_main = df_main.dropna(subset=["timestamp"])

# filtre période
df_main = filtrer_periode(df_main)

# affichage période + pas détecté
if not df_main.empty:
    tmin = df_main["timestamp"].min()
    tmax = df_main["timestamp"].max()
    st.sidebar.caption(f"Période détectée : {tmin} → {tmax}")

    # détection pas (protégée)
    try:
        step_info = detect_sampling_step(df_main, "timestamp")
    except NameError:
        step_info = {"median_min": None, "mode_min": None, "summary": None}
        st.sidebar.error("❌ detect_sampling_step() n'est pas défini. Mets la fonction dans le bloc 1, avant cet appel.")
    except Exception as e:
        step_info = {"median_min": None, "mode_min": None, "summary": None}
        st.sidebar.error(f"❌ Erreur détection pas : {e}")

    if step_info.get("median_min") is not None:
        st.sidebar.success(
            f"⏱️ Pas détecté (médian) : {step_info['median_min']:.2f} min\n"
            f"📌 Pas le + fréquent : {step_info['mode_min']:.2f} min"
        )
        if step_info.get("summary"):
            st.sidebar.caption(f"Top pas : {step_info['summary']}")
    else:
        st.sidebar.warning("⏱️ Pas de remontée non détectable (timestamps insuffisants ou irréguliers).")

else:
    st.warning("⚠️ Aucune donnée valide après chargement/filtrage.")
    st.stop()

#----- Bloc 4 -------------#
# ----------------------------- Nettoyage noms (comparaison) -----------------------------

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
    s = re.sub(r"\s+", " ", s)                # normalise les espaces
    return s.strip()

# Colonnes nettoyées (sauf timestamp)
df_main_cleaned = df_main.copy()
df_main_cleaned.columns = [
    "timestamp" if str(c).lower() == "timestamp" else nettoyer_nom_capteur(c)
    for c in df_main_cleaned.columns
]

# ----------------------------- Fichier de référence (facultatif) -----------------------------

df_compare = None
capteurs_reference_cleaned = set()

if compare_file is not None:
    try:
        raw_ref = BytesIO(compare_file.getvalue())

        # Feuille: si plusieurs, on laisse l'utilisateur choisir
        xls_ref = pd.ExcelFile(raw_ref)
        if len(xls_ref.sheet_names) == 1:
            ref_sheet = xls_ref.sheet_names[0]
        else:
            ref_sheet = st.sidebar.selectbox(
                "📄 Feuille à utiliser pour le fichier de comparaison",
                xls_ref.sheet_names,
                key="compare_sheet_select"
            )

        df_compare = pd.read_excel(xls_ref, sheet_name=ref_sheet)

        if "Description" not in df_compare.columns:
            st.error("❌ Le fichier de comparaison doit contenir une colonne 'Description'.")
            st.stop()

        df_compare["Description"] = df_compare["Description"].astype(str).str.strip()

        # Ensemble des capteurs de référence (nettoyés)
        capteurs_reference_cleaned = {
            nettoyer_nom_capteur(c) for c in df_compare["Description"].dropna().tolist()
        }

        st.success(f"✅ Fichier de comparaison chargé ({len(capteurs_reference_cleaned)} capteurs).")

    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier de comparaison : {e}")
        st.stop()
else:
    st.info("ℹ️ Aucun fichier de comparaison n'a été téléversé (facultatif).")

#----- Bloc 5 -------------#
# ----------------------------- Analyse simple -----------------------------

def analyse_simplifiee(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Présentes vs Manquantes – Méthode simple")

    if df is None or df.empty:
        st.info("Aucune donnée à analyser.")
        return pd.DataFrame(columns=["Capteur", "Présentes", "% Présentes", "Manquantes", "% Manquantes", "Statut", "Nom_nettoye"])

    total = len(df)
    resume = []

    for col in df.columns:
        if str(col).lower() in ("timestamp", "notes"):
            continue

        # placeholders -> vrais NaN
        s = series_with_true_nans(df[col])

        presente = int(s.notna().sum())
        manquantes = int(total - presente)

        pct_presente = (100.0 * presente / total) if total > 0 else 0.0
        pct_manquantes = 100.0 - pct_presente

        statut = "🟢" if pct_presente >= 80 else ("🟠" if pct_presente > 0 else "🔴")

        resume.append({
            "Capteur": str(col).strip(),
            "Présentes": presente,
            "% Présentes": round(pct_presente, 2),
            "Manquantes": manquantes,
            "% Manquantes": round(pct_manquantes, 2),
            "Statut": statut
        })

    df_resume = pd.DataFrame(resume)

    # nom nettoyé (sert pour doublons + comparaison)
    df_resume["Nom_nettoye"] = df_resume["Capteur"].astype(str).apply(nettoyer_nom_capteur)

    st.dataframe(df_resume, use_container_width=True)
    return df_resume

# ✅ Utiliser df_main_cleaned pour être cohérent avec les noms nettoyés
df_simple = analyse_simplifiee(df_main_cleaned)

# ----------------------------- Doublons (basé sur nom nettoyé) -----------------------------

df_simple["Doublon"] = df_simple["Nom_nettoye"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})
df_simple = df_simple.drop_duplicates(subset=["Nom_nettoye"], keep="last").reset_index(drop=True)

# ----------------------------- Validation vs référence (si fournie) -----------------------------

df_valides = pd.DataFrame()
df_non_valides = pd.DataFrame()
df_manquants = pd.DataFrame()

if isinstance(capteurs_reference_cleaned, set) and len(capteurs_reference_cleaned) > 0:
    df_simple["Dans la référence"] = df_simple["Nom_nettoye"].isin(capteurs_reference_cleaned).map(
        {True: "✅ Oui", False: "❌ Non"}
    )

    # trier: capteurs reconnus en haut
    df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

    st.subheader("✅ Capteurs trouvés dans la référence")
    df_valides = df_simple[df_simple["Dans la référence"] == "✅ Oui"]
    if not df_valides.empty:
        st.dataframe(df_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.write("Aucun capteur valide trouvé.")

    st.subheader("❌ Capteurs absents de la référence")
    df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]
    if not df_non_valides.empty:
        st.dataframe(df_non_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
        st.subheader("Liste brute – Capteurs absents de la référence")
        st.write(df_non_valides["Capteur"].tolist())
    else:
        st.write("Tous les capteurs sont présents dans la référence.")

    # capteurs attendus mais absents des données
    capteurs_trouves = set(df_simple["Nom_nettoye"].dropna().tolist())
    manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)

    if manquants:
        st.subheader("Capteurs attendus non trouvés dans les données analysées")
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"])
        st.dataframe(df_manquants, use_container_width=True)
    else:
        st.write("✅ Tous les capteurs attendus sont présents dans les données.")

#----- Bloc 6 -------------#
# ----------------------------- Resample (utile pour d'autres vues, pas pour la complétude) -----------------------------

def resampler_df(df: pd.DataFrame, frequence_str: str, rule_map: dict) -> pd.DataFrame:
    """
    Rééchantillonne les DONNÉES (valeurs) selon la fréquence choisie.
    ⚠️ À ne pas utiliser pour calculer la complétude (utiliser analyser_completude_freq).
    """
    if df is None or df.empty:
        st.info("Aucune donnée à rééchantillonner.")
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        st.warning("⚠️ Colonne 'timestamp' non trouvée dans le fichier.")
        return df.copy()

    if frequence_str not in rule_map:
        st.warning("⚠️ Fréquence invalide.")
        return df.copy()

    st.info(f"⏱️ Fréquence sélectionnée : {frequence_str}")

    try:
        df2 = df.copy()
        df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
        df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp")

        # pas de resample si 1min
        if frequence_str == "1min":
            st.info("✅ Pas de rééchantillonnage nécessaire (1min).")
            return df2.reset_index(drop=True)

        freq = rule_map[frequence_str]

        # index temporel
        df2 = df2.set_index("timestamp")

        # mapping d'agrégation
        agg_map = {}
        for col in df2.columns:
            c = str(col).lower()
            if c == "notes":
                agg_map[col] = "first"
            else:
                agg_map[col] = "mean" if is_numeric_dtype(df2[col]) else "first"

        df_resampled = df2.resample(freq).agg(agg_map).reset_index()

        st.success(f"✅ Données rééchantillonnées avec succès à {frequence_str}.")
        return df_resampled

    except Exception as e:
        st.error(f"❌ Erreur lors du rééchantillonnage : {e}")
        return df.reset_index(drop=True) if isinstance(df.index, pd.RangeIndex) else df.reset_index()


# ----------------------------- Nouvelle complétude (INDICATEUR) -----------------------------

def build_presence_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un indicateur de présence (1 = présent, NaN = manquant)
    pour toutes les colonnes sauf timestamp/notes, à partir des données BRUTES.
    """
    cols = [c for c in df.columns if str(c).lower() not in ("timestamp", "notes")]
    ind = pd.DataFrame(index=df.index)

    for c in cols:
        s = series_with_true_nans(df[c])
        ind[c] = s.notna().astype("float")  # 1.0 = présent, NaN = manquant

    return ind


def analyser_completude_freq(df: pd.DataFrame, frequence_str: str, rule_map: dict) -> pd.DataFrame:
    """
    Calcule la complétude à partir d'un indicateur de présence.
    Retourne TOUJOURS un DataFrame.
    """
    base_cols = ["Capteur", "Présentes", "% Présentes", "Manquantes", "% Manquantes", "Statut"]

    # ✅ debug visible (tu peux enlever après)
    # st.caption("DEBUG: analyser_completude_freq() appelée")

    # garde-fous
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=base_cols)

    if "timestamp" not in df.columns:
        st.error("❌ La colonne 'timestamp' est manquante.")
        return pd.DataFrame(columns=base_cols)

    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if df2.empty:
        return pd.DataFrame(columns=base_cols)

    ind = build_presence_indicator(df2)
    ind.index = df2["timestamp"]

    if frequence_str != "1min":
        if frequence_str not in rule_map:
            st.error("❌ Fréquence invalide.")
            return pd.DataFrame(columns=base_cols)

        freq = rule_map[frequence_str]
        ind_bin = ind.resample(freq).max()  # bin présent si au moins une valeur
        total_expected = float(len(ind_bin.index))
        if total_expected == 0:
            return pd.DataFrame(columns=base_cols)

        total_present = ind_bin.fillna(0).sum(axis=0)
    else:
        total_expected = float(len(ind))
        if total_expected == 0:
            return pd.DataFrame(columns=base_cols)

        total_present = ind.fillna(0).sum(axis=0)

    rows = []
    for c in ind.columns:
        pres = float(total_present.get(c, 0.0))
        pct = 100.0 * pres / total_expected
        statut = "🟢" if pct >= 80 else ("🟠" if pct > 0 else "🔴")

        rows.append({
            "Capteur": str(c),
            "Présentes": int(round(pres)),
            "% Présentes": round(pct, 2),
            "Manquantes": int(round(total_expected - pres)),
            "% Manquantes": round(100.0 - pct, 2),
            "Statut": statut
        })

    return pd.DataFrame(rows, columns=base_cols)

#----- Bloc 7 -------------#
st.subheader(f"📈 Analyse de complétude des données brutes ({frequence})")

# 1) Calcul complétude
stats_main = analyser_completude_freq(df_main_cleaned, frequence, rule_map)

st.write("DEBUG type stats_main:", type(stats_main))
st.write("DEBUG colonnes stats_main:", list(stats_main.columns) if isinstance(stats_main, pd.DataFrame) else stats_main)

# 2) Sécurités AVANT manip
if stats_main is None or not isinstance(stats_main, pd.DataFrame):
    st.error("⛔ analyser_completude_freq() n'a pas retourné un DataFrame (stats_main est None ou invalide).")
    st.stop()

expected_cols = ["Capteur", "Présentes", "% Présentes", "Manquantes", "% Manquantes", "Statut"]
missing = [c for c in expected_cols if c not in stats_main.columns]
if missing:
    st.error(f"⛔ Colonnes manquantes dans stats_main : {missing} | Colonnes trouvées : {list(stats_main.columns)}")
    st.stop()

# 3) Si vide : on affiche, mais on ne stop pas forcément (tu peux choisir)
if stats_main.empty:
    st.warning("⚠️ Résultat complétude vide (aucun capteur ou aucune donnée exploitable).")
    st.dataframe(stats_main, use_container_width=True)
else:
    # Nettoyage capteur + dédup
    stats_main["Capteur"] = stats_main["Capteur"].astype(str).str.strip()
    stats_main = stats_main.drop_duplicates(subset=["Capteur"], keep="last").reset_index(drop=True)
    st.dataframe(stats_main, use_container_width=True)

# ----------------------------- Légende + Résumé -----------------------------
st.markdown("""
### 🧾 Légende des statuts :
- 🟢 : Capteur exploitable (≥80 % de valeurs présentes)
- 🟠 : Incomplet (entre 1 % et 79 %)
- 🔴 : Données absentes (0 %)
""")

if not stats_main.empty:
    count_vert = int(stats_main["Statut"].value_counts().get("🟢", 0))
    count_orange = int(stats_main["Statut"].value_counts().get("🟠", 0))
    count_rouge = int(stats_main["Statut"].value_counts().get("🔴", 0))

    st.markdown(f"""
**Résumé des capteurs :**
- Capteurs exploitables (🟢) : `{count_vert}`
- Capteurs incomplets (🟠) : `{count_orange}`
- Capteurs vides (🔴) : `{count_rouge}`
""")

# sanity-check
st.caption(f"⏱️ Lignes analysées : {len(df_main)}")
st.caption(
    f"🧮 Colonnes (hors timestamp/notes) : "
    f"{len([c for c in df_main.columns if str(c).lower() not in ('timestamp','notes')])}"
)

# ----------------------------- Statistiques valeurs (min/max/moyenne) -----------------------------
st.subheader("📌 Statistiques des valeurs (min / max / moyenne)")

try:
    df_stats_values = stats_min_max_mean(df_main_cleaned)  # nécessite la fonction ajoutée dans le bloc utilitaires
except Exception as e:
    st.error(f"❌ Erreur stats_min_max_mean : {e}")
    df_stats_values = pd.DataFrame(columns=["Capteur", "Min", "Max", "Moyenne", "Nb valeurs", "Nb manquantes", "% présentes"])

st.dataframe(df_stats_values, use_container_width=True)

# ----------------------------- Graphique complétude (barh) -----------------------------
st.subheader("📊 Graphique — Complétude des capteurs")

if stats_main.empty:
    st.info("Aucun graphique : stats_main est vide.")
else:
    df_plot = stats_main.sort_values(by="% Présentes", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.25)))
    ax.barh(df_plot["Capteur"], df_plot["% Présentes"])
    ax.set_title("Complétude des capteurs")
    ax.set_xlabel("% Données présentes")
    ax.set_ylabel("Capteur")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------- Export Excel -----------------------------
st.subheader("📤 Export des résultats (Excel)")

# ✅ Sécuriser step_info : s’il n’existe pas, on le recalcule ici
if "step_info" not in locals() or not isinstance(step_info, dict):
    step_info = detect_sampling_step(df_main, "timestamp")

# ✅ Sécuriser df_simple : s’il n’existe pas / vide, on crée un DF minimal
if "df_simple" not in locals() or not isinstance(df_simple, pd.DataFrame) or df_simple.empty:
    df_simple = pd.DataFrame(columns=["Capteur", "Présentes", "% Présentes", "Manquantes", "% Manquantes", "Statut", "Nom_nettoye", "Doublon"])

output = BytesIO()

with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    # Feuille 1 : résumé simple
    df_simple.to_excel(writer, index=False, sheet_name="Résumé capteurs")

    # Feuille 2 : complétude brute
    stats_main.to_excel(writer, index=False, sheet_name="Complétude brute")

    # Feuille 3 : stats valeurs
    df_stats_values.to_excel(writer, index=False, sheet_name="Stats valeurs")

    # Feuille 4 : info pas
    df_step = pd.DataFrame([{
        "Pas médian": str(step_info.get("median")),
        "Pas médian (min)": step_info.get("median_min"),
        "Pas le + fréquent": str(step_info.get("mode")),
        "Pas le + fréquent (min)": step_info.get("mode_min"),
        "Top pas (count)": step_info.get("summary"),
        "Fréquence choisie (analyse)": frequence
    }])
    df_step.to_excel(writer, index=False, sheet_name="Info pas")

    # Feuilles optionnelles
    if "df_non_valides" in locals() and isinstance(df_non_valides, pd.DataFrame) and not df_non_valides.empty:
        df_non_valides.to_excel(writer, index=False, sheet_name="Capteurs non reconnus")

    if "df_manquants" in locals() and isinstance(df_manquants, pd.DataFrame) and not df_manquants.empty:
        df_manquants.to_excel(writer, index=False, sheet_name="Capteurs manquants")

    # Mise en forme conditionnelle (si colonne Statut existe)
    workbook = writer.book
    format_vert = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_orange = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})
    format_rouge = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    feuille = writer.sheets["Résumé capteurs"]

    if "Statut" in df_simple.columns:
        statut_col = df_simple.columns.get_loc("Statut")
        last_row = max(1, len(df_simple))  # évite 0

        feuille.conditional_format(1, statut_col, last_row, statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟢', 'format': format_vert
        })
        feuille.conditional_format(1, statut_col, last_row, statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟠', 'format': format_orange
        })
        feuille.conditional_format(1, statut_col, last_row, statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🔴', 'format': format_rouge
        })

output.seek(0)

st.download_button(
    label="📥 Télécharger le rapport Excel",
    data=output,
    file_name="rapport_capteurs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)











