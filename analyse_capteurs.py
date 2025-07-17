import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from io import BytesIO

# ---------------------------
# Config & Title
# ---------------------------
st.set_page_config(page_title="Analyse de données capteurs", layout="wide")
st.title("📊 Analyse de données capteurs")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name):
    """Read a sheet into a DataFrame."""
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df

def clean_columns(df):
    """Strip, normalize, and deduplicate column names."""
    cols = []
    seen = {}
    for c in df.columns:
        c0 = str(c).strip()
        c0 = re.sub(r"\s+", "_", c0)
        c0 = re.sub(r"[^\w\-]", "", c0)
        c0_lower = c0.lower()
        if c0_lower in seen:
            seen[c0_lower] += 1
            c0 = f"{c0}_{seen[c0_lower]}"
        else:
            seen[c0_lower] = 1
        cols.append(c0)
    df.columns = cols
    return df

def detect_time_column(df):
    """Try to pick a timestamp column heuristically."""
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["time","date","timestamp","datetime"])]
    if not candidates:
        # Fallback: first column
        candidates = [df.columns[0]]
    return candidates[0]

def ensure_datetime(df, col, tz=None):
    s = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    if tz:
        # If tz-naive, localize; if tz-aware and different, convert
        if s.dt.tz is None:
            s = s.dt.tz_localize(tz)
        else:
            s = s.dt.tz_convert(tz)
    return s

def summarize_presence(df, time_col="timestamp", exclude=("notes",)):
    """Simple row-based presence (no resampling)."""
    nb_total = len(df)
    rows = []
    for c in df.columns:
        if c == time_col or c.lower() in exclude:
            continue
        nb_pres = df[c].notna().sum()
        pct_pres = 100 * nb_pres / nb_total if nb_total else np.nan
        rows.append({
            "Capteur": c,
            "Présentes": nb_pres,
            "% Présentes": round(pct_pres,2),
            "Manquantes": nb_total - nb_pres,
            "% Manquantes": round(100 - pct_pres,2)
        })
    return pd.DataFrame(rows)

def estimate_freq_minutes(ts):
    """Estimate dominant sampling interval in minutes (median of diffs)."""
    deltas = ts.sort_values().diff().dropna()
    if deltas.empty:
        return np.nan
    median_delta = deltas.median()
    # Round to nearest "nice" interval (min)
    minutes = median_delta.total_seconds() / 60
    if minutes == 0:
        return 0.0
    # Snap to common intervals
    common = np.array([0.25, 0.5, 1, 5, 10, 15, 30, 60, 120, 240, 720, 1440])
    idx = (np.abs(common - minutes)).argmin()
    return common[idx]

def resample_presence(df, time_col="timestamp", freq_minutes=None):
    """Resample to regular grid, compute data presence by expected timestamps."""
    if freq_minutes is None or np.isnan(freq_minutes) or freq_minutes <= 0:
        return None
    freq_str = f"{int(freq_minutes)}T" if freq_minutes >= 1 else f"{int(freq_minutes*60)}S"
    dfr = df.set_index(time_col).sort_index()
    # Cast all non-numeric columns to numeric (for presence only)
    num_df = dfr.apply(pd.to_numeric, errors="coerce")
    # Reindex on full range
    full_index = pd.date_range(start=num_df.index.min(), end=num_df.index.max(), freq=freq_str)
    aligned = num_df.reindex(full_index)
    nb_total = len(aligned)
    rows = []
    for c in aligned.columns:
        nb_pres = aligned[c].notna().sum()
        pct_pres = 100 * nb_pres / nb_total if nb_total else np.nan
        rows.append({
            "Capteur": c,
            "Présentes attendues": nb_total,
            "Présentes réelles": nb_pres,
            "% Présentes (rés échant.)": round(pct_pres,2),
            "% Manquantes (rés échant.)": round(100 - pct_pres,2),
        })
    return pd.DataFrame(rows), aligned

def plot_presence_bar(df_simple, title):
    fig, ax = plt.subplots(figsize=(14,6))
    df_simple.set_index("Capteur")[["% Présentes","% Manquantes"]].plot(
        kind="bar", stacked=True, ax=ax
    )
    ax.set_ylabel("%")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_delta_hist(deltas, seuil):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.hist(deltas, bins=50)
    ax.set_xlabel("Écart de temps (minutes)")
    ax.set_ylabel("Fréquence")
    ax.axvline(seuil, linestyle="--")
    ax.set_title("Distribution des écarts temporels")
    plt.tight_layout()
    return fig

def df_to_csv_download(df, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=label.lower().replace(" ","_") + ".csv",
        mime="text/csv"
    )

# ---------------------------
# User inputs
# ---------------------------
uploaded_files = st.file_uploader(
    "Choisissez un ou plusieurs fichiers Excel à analyser",
    type=["xlsx","xls"],
    accept_multiple_files=True
)

tz_select = st.selectbox(
    "Fuseau horaire des données", 
    options=[None, "America/Toronto", "UTC"],
    index=1,
    format_func=lambda x: "Aucun (laisser tel quel)" if x is None else x
)

if uploaded_files:
    dfs = {}  # store cleaned DF keyed by file name

    # --- Loop over uploaded files ---
    for file in uploaded_files:
        st.divider()
        st.header(f"Fichier : {file.name}")

        try:
            # Load once (cached)
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            sheet = st.selectbox(
                f"Feuille à analyser ({file.name})",
                sheet_names,
                key=f"sheet_{file.name}"
            )
            raw_df = load_excel(file, sheet)

            df = clean_columns(raw_df.copy())
            time_col = detect_time_column(df)

            st.write(f"Colonne temps détectée : **{time_col}**")
            # Allow override
            time_col = st.selectbox(
                f"Sélectionne la colonne temps pour {file.name}",
                options=df.columns.tolist(),
                index=df.columns.get_loc(time_col),
                key=f"timecol_{file.name}"
            )

            # Convert to datetime
            ts = ensure_datetime(df, time_col, tz=tz_select)
            df = df.assign(**{time_col: ts})
            df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

            # Basic period
            st.markdown(f"**Nombre total de lignes :** {len(df)}")
            st.markdown(f"**Période couverte :** {df[time_col].min()} ➡ {df[time_col].max()}")

            # Estimate freq
            est_freq_min = estimate_freq_minutes(df[time_col])
            if not np.isnan(est_freq_min) and est_freq_min > 0:
                st.markdown(f"**Fréquence dominante estimée :** ~{est_freq_min:.2f} min")
            else:
                st.markdown("**Fréquence dominante estimée :** indéterminée (données trop irrégulières).")

            # -------- Méthode simple --------
            st.subheader("📌 Présentes vs Manquantes – Méthode simple (pas de resampling)")
            df_simple = summarize_presence(df, time_col=time_col)
            st.dataframe(df_simple, use_container_width=True)

            fig_simple = plot_presence_bar(df_simple, "Pourcentage de données présentes et manquantes par capteur")
            st.pyplot(fig_simple)
            plt.close(fig_simple)

            df_to_csv_download(df_simple, f"resume_simple_{file.name}")

            # -------- Méthode temporelle optionnelle --------
            with st.expander("🔁 Analyse temporelle basée sur un rééchantillonnage régulier"):
                use_auto = st.checkbox("Utiliser fréquence estimée automatiquement", value=True, key=f"useauto_{file.name}")
                if use_auto and not np.isnan(est_freq_min) and est_freq_min > 0:
                    freq_min = est_freq_min
                else:
                    freq_min = st.number_input(
                        "Fréquence (minutes) pour le rééchantillonnage",
                        min_value=0.1, max_value=1440.0, value=15.0, step=1.0,
                        key=f"freqinp_{file.name}"
                    )
                res = resample_presence(df, time_col=time_col, freq_minutes=freq_min)
                if res is not None:
                    df_res, aligned = res
                    st.dataframe(df_res, use_container_width=True)
                    # Graph
                    fig_res = plot_presence_bar(
                        df_res.rename(columns={"% Présentes (rés échant.)":"% Présentes","% Manquantes (rés échant.)":"% Manquantes"}),
                        f"Présence des données après rééchantillonnage ({freq_min} min)"
                    )
                    st.pyplot(fig_res)
                    plt.close(fig_res)
                    df_to_csv_download(df_res, f"resume_resample_{file.name}")

            # Store cleaned DF
            dfs[file.name] = (df, time_col)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {file.name} : {e}")

    # ---------------------------
    # Global time-gap analysis across *one selected file*
    # ---------------------------
    st.divider()
    st.subheader("⏱️ Analyse des écarts entre les timestamps")

    if len(dfs) == 1:
        selected_name = list(dfs.keys())[0]
    else:
        selected_name = st.selectbox(
            "Choisir le fichier pour l'analyse des écarts temporels",
            options=list(dfs.keys())
        )

    df_sel, time_col_sel = dfs[selected_name]

    # Compute deltas
    deltas = df_sel[time_col_sel].sort_values().diff().dropna()
    deltas_min = deltas.dt.total_seconds() / 60

    st.write("**Statistiques des écarts (en minutes) entre points de données :**")
    st.write(deltas_min.describe())

    seuil = st.slider("Seuil pour considérer un grand écart (minutes)", 10, 240, 60)
    nb_grands_ecarts = (deltas_min > seuil).sum()
    pct_grands_ecarts = 100 * nb_grands_ecarts / len(deltas_min) if len(deltas_min) else np.nan
    st.markdown(f"🔍 **{pct_grands_ecarts:.2f}% des écarts dépassent {seuil} minutes.**")

    fig_delta = plot_delta_hist(deltas_min, seuil)
    st.pyplot(fig_delta)
    plt.close(fig_delta)

else:
    st.info("Téléverse au moins un fichier Excel pour démarrer l’analyse.")
