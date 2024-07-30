import pandas as pd

file_path = 'data/GDPinput.xlsx'
csv_file_path = 'data/GDPinput.csv'

GDP = pd.read_excel(file_path, sheet_name='Feuil1')
GDP=GDP.drop(columns=['Tcode', 'Ticker']).transpose()
GDP.to_csv(csv_file_path, index=False)

print(f"CSV file saved to {csv_file_path}")