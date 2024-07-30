import pandas as pd

file_path = 'data/GDPTrackingModelDataAndForecasts.xlsx'
monthly_levels = pd.read_excel(file_path, sheet_name='MonthlyLevels')
cons_monthly_levels = pd.read_excel(file_path, sheet_name='ConsMonthlyLevels')
transformed_monthly_series = pd.read_excel(file_path, sheet_name='TransformedMonthlySeries')
cons_transformed_monthly_series = pd.read_excel(file_path, sheet_name='ConsTransformedMonthlySeries')
GDP_now=pd.read_excel(file_path, sheet_name='TrackingArchives')

# Extract relevant data and combine into one DataFrame
# For GDPnow
# GDP_now = GDP_now.drop(columns=deletecolumns)
GDP_now.to_csv('data/GDPNow.csv', index=False)



# For MonthlyLevels
monthly_levels_data = monthly_levels.drop(columns=['Tcode', 'Description', 'Ticker']).transpose()
monthly_levels_data.columns = monthly_levels['Description']

# For ConsMonthlyLevels
cons_monthly_levels_data = cons_monthly_levels.drop(columns=['Tcode', 'Description', 'Ticker']).transpose()
cons_monthly_levels_data.columns = cons_monthly_levels['Description']

# For TransformedMonthlySeries
transformed_monthly_series_data = transformed_monthly_series.drop(columns=['Tcode', 'Description', 'Ticker']).transpose()
transformed_monthly_series_data.columns = transformed_monthly_series['Description']

# For ConsTransformedMonthlySeries
cons_transformed_monthly_series_data = cons_transformed_monthly_series.drop(columns=['Tcode', 'Description', 'Ticker']).transpose()
cons_transformed_monthly_series_data.columns = cons_transformed_monthly_series['Description']

# Combine all data into one DataFrame
combined_data = pd.concat([monthly_levels_data, cons_monthly_levels_data,
                           transformed_monthly_series_data, cons_transformed_monthly_series_data], axis=1)

# Reset index to have dates as a column
combined_data.reset_index(inplace=True)
combined_data.rename(columns={'index': 'Date'}, inplace=True)

# Save the combined data to a CSV file
csv_file_path = 'data/GDPNow_inputs.csv'
combined_data.to_csv(csv_file_path, index=False)

print(f"CSV file saved to {csv_file_path}")