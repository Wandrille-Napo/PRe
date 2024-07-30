import pandas as pd

# Charger le fichier CSV des données quotidiennes
file_path = 'data/DGS10.csv'  # Spécifiez le chemin de votre fichier CSV
df = pd.read_csv(file_path)

# Convertir la colonne 'DATE' en datetime si elle n'est pas déjà convertie
df['DATE'] = pd.to_datetime(df['DATE'])

# Convertir la colonne 'DGS10' en type numérique (float), en traitant les erreurs de conversion comme NaN
df['DGS10'] = pd.to_numeric(df['DGS10'], errors='coerce')

# Convertir la colonne 'DATE' en période mensuelle ('YYYY-MM')
df['DATE'] = df['DATE'].dt.to_period('M')

# Agréger les données quotidiennes en moyenne mensuelle pour chaque mois
monthly_data = df.groupby('DATE', as_index=False)['DGS10'].mean()

# Enregistrer le résultat dans un nouveau fichier CSV
output_file_path = 'data/monthly_DGS10.csv'  # Spécifiez le chemin de sortie souhaité
monthly_data.to_csv(output_file_path, index=False)

print(f"Données mensuelles enregistrées dans {output_file_path}")
