import pandas as pd
from src.etl.export_csv_warehouse import export_csv

# Ruta del archivo procesado generado por clean_hr.py
input_path = "Data/processed/clean_hr.csv"
# Ruta de exportación final
output_path = "Data/warwhouse/clean_hr_export.csv"

df = pd.read_csv(input_path)
export_csv(df, output_path)
print(f"Archivo exportado a: {output_path}")
