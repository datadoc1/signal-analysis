import pandas as pd
import os

# Read the Excel file
excel_file = 'Signal Analysis.xlsx'
df = pd.read_excel(excel_file, sheet_name=None)

# Create the output folder
output_folder = 'specialties_csv'
os.makedirs(output_folder, exist_ok=True)

# Convert each sheet to CSV
for sheet_name, sheet_data in df.items():
    # Generate the output file path
    output_file = os.path.join(output_folder, f'{sheet_name}.csv')
    
    # Save the sheet data as CSV
    sheet_data.to_csv(output_file, index=False)