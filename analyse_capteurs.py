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
uploaded_files = st.sidebar.file_uploader(
    "📂 Fichiers principaux (vous pouvez en téléverser plusieurs)",
    type=[".xlsx", ".xls", ".xlsm"],
    accept_multiple_files=True,
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
if not uploaded_files:
    st.warning("⚠️ Veuillez téléverser au moins un fichier principal.")
    st.stop()

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

# 📥 Analyse de chaque fichier principal téléversé
for i, main_file in enumerate(uploaded_files):
    st.markdown(f"## 📁 Fichier {i+1} : `{main_file.name}`")
    
    df_main = charger_et_resampler(main_file, f"Fichier principal {i+1}")
    

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
if capteurs_reference is not None and len(capteurs_reference) > 0:
    import re

    def nettoyer_nom_capteur(nom):
        return re.sub(r"\s*\[[^\]]*\]", "", nom).strip()

    capteurs_reference_cleaned = {nettoyer_nom_capteur(c) for c in capteurs_reference}
    df_simple["Nom_nettoye"] = df_simple["Capteur"].apply(nettoyer_nom_capteur)
    df_simple["Dans la référence"] = df_simple["Nom_nettoye"].apply(
        lambda nom: "✅ Oui" if nom in capteurs_reference_cleaned else "❌ Non"
    )

    df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

    # ✅ Capteurs présents
    st.subheader("✅ Capteurs trouvés dans la référence")
    df_valides = df_simple[df_simple["Dans la référence"] == "✅ Oui"]
    if not df_valides.empty:
        st.dataframe(df_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.markdown("Aucun capteur valide trouvé.")

    # ❌ Capteurs absents
    st.subheader("❌ Capteurs absents de la référence")
    df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]
    if not df_non_valides.empty:
        st.dataframe(df_non_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.markdown("Tous les capteurs sont présents dans la référence.")

    # Liste brute
    if not df_non_valides.empty:
        st.subheader("📋 Liste brute – Capteurs absents de la référence")
        st.write(df_non_valides["Capteur"].tolist())

    # 🔎 Capteurs attendus mais manquants dans le fichier
    capteurs_trouves = set(df_simple["Nom_nettoye"])
    manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
    if manquants:
        st.subheader("📌 Capteurs attendus non trouvés")
        st.markdown("Capteurs attendus dans la référence mais absents du fichier :")
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"])
        st.dataframe(df_manquants, use_container_width=True)
    else:
        st.markdown("✅ Tous les capteurs attendus sont présents dans les données.")


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

# 📈 Analyse de complétude sans rééchantillonnage
st.subheader(f"📈 Complétude – Données brutes (Fichier {i+1})")
stats_main = analyser_completude(df_main)
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
**Résumé des capteurs pour `{main_file.name}` :**
- 🟢 Capteurs exploitables : `{count_vert}`
- 🟠 Capteurs incomplets : `{count_orange}`
- 🔴 Capteurs vides : `{count_rouge}`
""")

# 📉 Graphique horizontal par capteur
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
plt.title(f"Complétude des capteurs – `{main_file.name}`", fontsize=14)
plt.xlabel("% Données présentes")
plt.ylabel("Capteur")
plt.xlim(0, 100)
plt.tight_layout()
st.pyplot(fig)


# ✅ Export Excel final avec couleurs
from io import BytesIO

# === Initialisation ===
export_global = BytesIO()
writer_global = pd.ExcelWriter(export_global, engine='xlsxwriter')
table_globale = []

# === Boucle sur les fichiers téléversés ===
for i, main_file in enumerate(uploaded_files):
    st.markdown(f"## 📁 Fichier {i+1} : `{main_file.name}`")
    df_main = charger_et_resampler(main_file, f"Fichier principal {i+1}")
    df_simple = analyse_simplifiee(df_main, capteurs_reference)
    stats_main = analyser_completude(df_main)

    # Nettoyage doublons
    df_simple["Capteur"] = df_simple["Capteur"].astype(str).str.strip()
    df_simple["Doublon"] = df_simple["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

    # Validation si référence présente
    df_non_valides, df_manquants = None, None
    if capteurs_reference:
        import re

        def nettoyer_nom_capteur(nom):
            return re.sub(r"\s*\[[^\]]*\]", "", nom).strip()

        capteurs_reference_cleaned = {nettoyer_nom_capteur(c) for c in capteurs_reference}
        df_simple["Nom_nettoye"] = df_simple["Capteur"].apply(nettoyer_nom_capteur)
        df_simple["Dans la référence"] = df_simple["Nom_nettoye"].apply(
            lambda nom: "✅ Oui" if nom in capteurs_reference_cleaned else "❌ Non"
        )
        df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)
        df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]
        capteurs_trouves = set(df_simple["Nom_nettoye"])
        manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"]) if manquants else None

    # Nom raccourci
    nom_base = main_file.name.replace(".xlsx", "").replace(".xlsm", "").replace(".xls", "")[:20]

    # === Ajouter à l'Excel ===
    df_simple.to_excel(writer_global, index=False, sheet_name=f"Résumé - {nom_base}")
    stats_main.to_excel(writer_global, index=False, sheet_name=f"Complétude - {nom_base}")
    if df_non_valides is not None and not df_non_valides.empty:
        df_non_valides.to_excel(writer_global, index=False, sheet_name=f"Non reconnus - {nom_base}")
    if df_manquants is not None and not df_manquants.empty:
        df_manquants.to_excel(writer_global, index=False, sheet_name=f"Manquants - {nom_base}")

    # === Mise en forme conditionnelle ===
    workbook = writer_global.book
    feuille = writer_global.sheets[f"Résumé - {nom_base}"]
    format_vert = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_orange = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})
    format_rouge = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    if "Statut" in df_simple.columns:
        statut_col = df_simple.columns.get_loc("Statut")
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟢', 'format': format_vert
        })
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟠', 'format': format_orange
        })
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🔴', 'format': format_rouge
        })

    # === Ajouter à la synthèse globale ===
    for _, row in df_simple.iterrows():
        table_globale.append({
            "Fichier": main_file.name,
            "Capteur": row["Capteur"],
            "% Présentes": row["% Présentes"],
            "Statut": row["Statut"]
        })

# === Ajouter la synthèse globale ===
df_global = pd.DataFrame(table_globale)
df_global.to_excel(writer_global, index=False, sheet_name="Synthèse globale")

# === Finaliser et télécharger ===
writer_global.close()

st.subheader("📤 Export global de tous les fichiers")
st.download_button(
    label="📥 Télécharger le rapport global Excel",
    data=export_global.getvalue(),
    file_name="rapport_global_capteurs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
