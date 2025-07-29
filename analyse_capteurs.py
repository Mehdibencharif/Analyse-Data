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
def analyse_simplifiee(df, capteurs_reference=None, afficher=True, fichier_nom=None):
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

    if afficher:
        titre = f"📊 Données présentes vs manquantes"
        if fichier_nom:
            titre += f" – `{fichier_nom}`"
        st.subheader(titre)
        st.dataframe(df_resume, use_container_width=True)

        # Graphique horizontal
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

    return df_resume

# 📊 Analyse simple avec validation
df_simple = analyse_simplifiee(df_main, capteurs_reference)

# 🔁 Nettoyage des noms de capteurs & détection des doublons
df_simple["Capteur"] = df_simple["Capteur"].astype(str).str.strip()
df_simple["Doublon"] = df_simple["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

# 🔍 Validation par rapport au fichier de référence (si fourni)
if capteurs_reference:
    import re

    # 🔧 Fonction de nettoyage pour comparer proprement les noms
    def nettoyer_nom_capteur(nom):
        return re.sub(r"\s*\[[^\]]*\]", "", nom).strip()

    # Nettoyage de la référence (supprime les unités dans [ ])
    capteurs_reference_cleaned = {nettoyer_nom_capteur(c) for c in capteurs_reference}

    # Ajout d'une colonne nettoyée dans le fichier principal
    df_simple["Nom_nettoye"] = df_simple["Capteur"].apply(nettoyer_nom_capteur)

    # Vérification si chaque capteur est présent dans la référence
    df_simple["Dans la référence"] = df_simple["Nom_nettoye"].apply(
        lambda nom: "✅ Oui" if nom in capteurs_reference_cleaned else "❌ Non"
    )

    # Tri pour voir les capteurs valides en premier
    df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

    # ✅ Capteurs présents dans la référence
    st.subheader("✅ Capteurs trouvés dans la référence")
    df_valides = df_simple[df_simple["Dans la référence"] == "✅ Oui"]
    if not df_valides.empty:
        st.dataframe(df_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.info("Aucun capteur valide trouvé dans la référence.")

    # ❌ Capteurs non présents dans la référence
    st.subheader("❌ Capteurs absents de la référence")
    df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]
    if not df_non_valides.empty:
        st.dataframe(df_non_valides[["Capteur", "Dans la référence", "Doublon"]], use_container_width=True)
    else:
        st.success("Tous les capteurs du fichier sont présents dans la référence.")

    # 📋 Affichage brut des capteurs non reconnus
    if not df_non_valides.empty:
        st.subheader("📋 Liste brute – Capteurs absents de la référence")
        st.write(df_non_valides["Capteur"].tolist())

    # 📌 Capteurs attendus dans la référence mais absents du fichier
    capteurs_trouves = set(df_simple["Nom_nettoye"])
    capteurs_manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)

    if capteurs_manquants:
        st.subheader("📌 Capteurs attendus non trouvés")
        st.markdown("Ces capteurs sont dans la référence mais absents du fichier de données :")
        df_manquants = pd.DataFrame(capteurs_manquants, columns=["Capteur (référence manquant dans les données)"])
        st.dataframe(df_manquants, use_container_width=True)
    else:
        st.success("✅ Tous les capteurs attendus sont présents dans les données.")


# 📈 Analyse de complétude sans rééchantillonnage pour chaque fichier
st.subheader(f"📈 Complétude – Données brutes (Fichier {i+1} : `{main_file.name}`)")
stats_main = analyser_completude(df_main)

if stats_main.empty:
    st.warning("Aucune donnée numérique trouvée dans ce fichier.")
else:
    # 💠 Affichage du tableau
    st.dataframe(stats_main, use_container_width=True)

    # 📘 Légende des statuts
    st.markdown("""
    ### 🧾 Légende des statuts :
    - 🟢 : Capteur exploitable (≥ 80 %)
    - 🟠 : Incomplet (entre 1 % et 79 %)
    - 🔴 : Données absentes (0 %)
    """)

    # 📌 Résumé numérique des capteurs
    count_vert = stats_main["Statut"].value_counts().get("🟢", 0)
    count_orange = stats_main["Statut"].value_counts().get("🟠", 0)
    count_rouge = stats_main["Statut"].value_counts().get("🔴", 0)
    st.markdown(f"""
    **Résumé des capteurs pour `{main_file.name}` :**
    - 🟢 Capteurs exploitables : `{count_vert}`
    - 🟠 Capteurs incomplets : `{count_orange}`
    - 🔴 Capteurs vides : `{count_rouge}`
    """)

    # 📉 Graphique horizontal
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
from pathlib import Path
import re

# === Initialisation ===
export_global = BytesIO()  # Buffer pour l'export Excel
writer_global = pd.ExcelWriter(export_global, engine='xlsxwriter')  # Writer Excel
table_globale = []  # Contiendra toutes les lignes pour la synthèse finale

# ✅ Fonction pour créer un nom de feuille valide pour Excel (max 31 caractères)
def nom_feuille_limite(prefix, base):
    return f"{prefix} - {base}"[:31]

# === Boucle sur les fichiers téléversés ===
for i, main_file in enumerate(uploaded_files):
    st.markdown(f"## 📁 Fichier {i+1} : `{main_file.name}`")

    # 🔄 Chargement et rééchantillonnage des données du fichier
    df_main = charger_et_resampler(main_file, f"Fichier principal {i+1}")

    # 📊 Analyse simplifiée (présentes vs manquantes)
    df_simple = analyse_simplifiee(df_main, capteurs_reference)

    # 📈 Analyse de complétude (sans rééchantillonnage)
    stats_main = analyser_completude(df_main)

 # ✅ Nettoyage du nom du fichier pour créer un nom de feuille Excel propre et court
    nom_fichier_brut = Path(main_file.name).stem  # Retire l'extension du fichier
    nom_base = re.sub(r'[\\/*?:[\]]', '_', nom_fichier_brut)[:20]  # Supprime les caractères invalides et limite à 20 caractères

    # 📤 Export des tableaux individuels vers le fichier Excel global
    df_simple.to_excel(writer_global, index=False, sheet_name=nom_feuille_limite("Résumé", nom_base))
    stats_main.to_excel(writer_global, index=False, sheet_name=nom_feuille_limite("Complétude", nom_base))

    # ⚠️ Export des capteurs non reconnus si variables déjà existantes et non vides
    if 'df_non_valides' in locals() and not df_non_valides.empty:
        df_non_valides.to_excel(writer_global, index=False, sheet_name=nom_feuille_limite("Non reconnus", nom_base))

    # ⚠️ Export des capteurs manquants dans les données
    if 'df_manquants' in locals() and not df_manquants.empty:
        df_manquants.to_excel(writer_global, index=False, sheet_name=nom_feuille_limite("Manquants", nom_base))

    # 🖌️ Mise en forme conditionnelle des statuts dans la feuille "Résumé"
    workbook = writer_global.book
    format_vert = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_orange = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})
    format_rouge = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    nom_feuille_resume = nom_feuille_limite("Résumé", nom_base)
    feuille = writer_global.sheets[nom_feuille_resume]

    # Appliquer la mise en forme conditionnelle si la colonne Statut est présente
    if "Statut" in df_simple.columns:
        statut_col = df_simple.columns.get_loc("Statut")  # Numéro de colonne
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟢', 'format': format_vert
        })
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🟠', 'format': format_orange
        })
        feuille.conditional_format(1, statut_col, len(df_simple), statut_col, {
            'type': 'text', 'criteria': 'containing', 'value': '🔴', 'format': format_rouge
        })

    # 🧹 Nettoyage et détection des doublons dans les noms de capteurs
    df_simple["Capteur"] = df_simple["Capteur"].astype(str).str.strip()
    df_simple["Doublon"] = df_simple["Capteur"].duplicated(keep=False).map({True: "🔁 Oui", False: "✅ Non"})

    # ✅ Validation : comparer avec la référence des capteurs
    df_non_valides, df_manquants = None, None  # Initialisation
    if capteurs_reference:
        def nettoyer_nom_capteur(nom):
            # Nettoie : supprime les unités entre crochets et les espaces
            return re.sub(r"\s*\[[^\]]*\]", "", nom).strip()

        # Nettoyage des noms dans la référence
        capteurs_reference_cleaned = {nettoyer_nom_capteur(c) for c in capteurs_reference}

        # Ajout des colonnes nettoyées au DataFrame principal
        df_simple["Nom_nettoye"] = df_simple["Capteur"].apply(nettoyer_nom_capteur)
        df_simple["Dans la référence"] = df_simple["Nom_nettoye"].apply(
            lambda nom: "✅ Oui" if nom in capteurs_reference_cleaned else "❌ Non"
        )

        # Trier avec les capteurs valides en premier
        df_simple = df_simple.sort_values(by="Dans la référence", ascending=False).reset_index(drop=True)

        # Détection des capteurs non reconnus
        df_non_valides = df_simple[df_simple["Dans la référence"] == "❌ Non"]

        # Détection des capteurs attendus mais absents
        capteurs_trouves = set(df_simple["Nom_nettoye"])
        manquants = sorted(capteurs_reference_cleaned - capteurs_trouves)
        df_manquants = pd.DataFrame(manquants, columns=["Capteur (référence manquant dans les données)"]) if manquants else None

    # === Ajouter les données à l'Excel ===
    df_simple.to_excel(writer_global, index=False, sheet_name=f"Résumé - {nom_base}")
    stats_main.to_excel(writer_global, index=False, sheet_name=f"Complétude - {nom_base}")

    if df_non_valides is not None and not df_non_valides.empty:
        df_non_valides.to_excel(writer_global, index=False, sheet_name=f"Non reconnus - {nom_base}")

    if df_manquants is not None and not df_manquants.empty:
        df_manquants.to_excel(writer_global, index=False, sheet_name=f"Manquants - {nom_base}")

    # === Mise en forme conditionnelle (répétée par sécurité ici pour feuille raccourcie) ===
    feuille = writer_global.sheets[f"Résumé - {nom_base}"]
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

    # ✅ Ajout du fichier à la synthèse globale
    for _, row in df_simple.iterrows():
        table_globale.append({
            "Fichier": main_file.name,
            "Capteur": row["Capteur"],
            "% Présentes": row["% Présentes"],
            "Statut": row["Statut"]
        })

# === Export de la synthèse finale multi-fichiers ===
df_global = pd.DataFrame(table_globale)
df_global.to_excel(writer_global, index=False, sheet_name="Synthèse globale")

# 🔒 Clôture du Writer Excel
writer_global.close()

# 📥 Bouton de téléchargement dans Streamlit
st.subheader("📤 Export global de tous les fichiers")
st.download_button(
    label="📥 Télécharger le rapport global Excel",
    data=export_global.getvalue(),
    file_name="rapport_global_capteurs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
  
