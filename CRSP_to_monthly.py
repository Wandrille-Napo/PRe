import pandas as pd

# Chargement du fichier CSV en spécifiant les types de données
file_path = 'data/CRSP_daily2.csv'  # Remplacez par le chemin de votre fichier
output_file_path = 'data/CRSP_monthly2.csv'  # Remplacez par le chemin de sortie souhaité

df = pd.read_csv(file_path, dtype={
    'NAMEENDT': str, 'SHRCLS': str, 'TSYMBOL': str, 'DCLRDT': str, 'DLPDT': str,
    'NEXTDT': str, 'PAYDT': str, 'RCRDDT': str, 'SHRFLG': str
}, na_values=[''], low_memory=False)

# Convertir la colonne 'date' en datetime
df['date'] = pd.to_datetime(df['date'])

# Convertir les colonnes 'RETX' et 'RET' en numériques, les valeurs non convertibles seront remplacées par NaN
df['RETX'] = pd.to_numeric(df['RETX'], errors='coerce')
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')

# Extraire l'année et le mois
df['year_month'] = df['date'].dt.to_period('M')

# Sélectionner les colonnes numériques pour calculer les moyennes
numeric_cols = df.select_dtypes(include=['number']).columns

# Calculer les moyennes mensuelles pour chaque entreprise (PERMNO) en ignorant les NaN
monthly_avg_numeric = df.groupby(['PERMNO', 'year_month'], as_index=False)[numeric_cols].mean()

# Sélectionner les colonnes non numériques pour la première occurrence du mois
non_numeric_cols = ['NAMEENDT', 'NCUSIP', 'TICKER', 'COMNAM', 'SHRCLS', 'TSYMBOL', 
                    'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT', 'CUSIP', 'DCLRDT', 
                    'DLPDT', 'NEXTDT', 'PAYDT', 'RCRDDT']
monthly_first_non_numeric = df.groupby(['PERMNO', 'year_month'], as_index=False)[non_numeric_cols].first()

# Fusionner les résultats en utilisant 'PERMNO' et 'year_month' comme clés de fusion
monthly_avg = pd.merge(monthly_avg_numeric, monthly_first_non_numeric, on=['PERMNO', 'year_month'])

# Réorganiser les colonnes dans l'ordre spécifié
desired_order = ['PERMNO', 'year_month', 'NAMEENDT', 'SHRCD', 'EXCHCD', 'SICCD', 'NCUSIP', 'TICKER', 'COMNAM', 'SHRCLS',
                 'TSYMBOL', 'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT', 'PERMCO', 'ISSUNO', 'HEXCD', 'HSICCD',
                 'CUSIP', 'DCLRDT', 'DLAMT', 'DLPDT', 'DLSTCD', 'NEXTDT', 'PAYDT', 'RCRDDT', 'SHRFLG', 'HSICMG',
                 'HSICIG', 'DISTCD', 'DIVAMT', 'FACPR', 'FACSHR', 'ACPERM', 'ACCOMP', 'SHRENDDT', 'NWPERM', 'DLRETX',
                 'DLPRC', 'DLRET', 'TRTSCD', 'NMSIND', 'MMCNT', 'NSDINX', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET',
                 'BID', 'ASK', 'SHROUT', 'CFACPR', 'CFACSHR', 'OPENPRC', 'NUMTRD', 'RETX', 'vwretd', 'vwretx', 'ewretd',
                 'ewretx', 'sprtrn']
monthly_avg = monthly_avg.reindex(columns=desired_order)

# Reformater la colonne 'year_month' pour l'affichage
monthly_avg['date'] = pd.to_datetime(monthly_avg['year_month'].dt.to_timestamp())

# Enregistrer le résultat dans un nouveau fichier CSV
monthly_avg.to_csv(output_file_path, index=False)

print(f"Moyennes mensuelles enregistrées dans {output_file_path}")
