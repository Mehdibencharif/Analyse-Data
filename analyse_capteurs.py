# === Présentes vs Manquantes (Méthode simple) ===
st.subheader("Présentes vs Manquantes – Méthode simple")

summary_simple = []

nb_total_lignes = len(df)  # total de lignes (attendues pour chaque capteur)

for col in df.columns:
    if col.lower() in ['timestamp', 'notes']:
        continue

    nb_presente = df[col].notna().sum()
    nb_manquante = nb_total_lignes - nb_presente

    pct_presente = 100 * nb_presente / nb_total_lignes
    pct_manquante = 100 - pct_presente

    summary_simple.append({
        "Capteur": col,
        "Présentes": nb_presente,
        "% Présentes": round(pct_presente, 2),
        "Manquantes": nb_manquante,
        "% Manquantes": round(pct_manquante, 2),
    })

df_simple = pd.DataFrame(summary_simple)

# Affichage du tableau
st.dataframe(df_simple)

# Affichage du graphique
fig, ax = plt.subplots(figsize=(14, 6))
df_simple.set_index("Capteur")[["% Présentes", "% Manquantes"]].plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"])
plt.ylabel("%")
plt.title("Pourcentage de données présentes et manquantes par capteur (méthode simple)")
plt.xticks(rotation=45, ha='right')
plt.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig)
