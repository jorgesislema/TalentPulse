
import pandas as pd

# Cambia la ruta si tu archivo está en otro lugar
csv_path = r"Data/raw/HR_Data_MNC_Data Science Lovers.csv"

# Lee solo las primeras filas para inspección rápida
n_filas = 10

# Mapeo de nombres originales a español
col_map = {
	'Unnamed: 0': 'índice',
	'Employee_ID': 'id_empleado',
	'Full_Name': 'nombre_completo',
	'Department': 'departamento',
	'Job_Title': 'puesto',
	'Hire_Date': 'fecha_contratacion',
	'Location': 'ubicacion',
	'Performance_Rating': 'calificacion_desempeno',
	'Experience_Years': 'anios_experiencia',
	'Status': 'estatus',
	'Work_Mode': 'modalidad_trabajo',
	'Salary_INR': 'salario_inr',
}

df = pd.read_csv(csv_path, nrows=n_filas)
df_es = df.rename(columns=col_map)

print("Columnas en español:")
print(list(df_es.columns))
print("\nPrimeras filas:")
print(df_es.head())
