import pandas as pd
import pytest
from src.etl.clean_hr import clean_dataframe

def test_clean_dataframe_preserva_id_empleado():
	# Simula un pequeño DataFrame similar al real
	data = {
		'Employee_ID': ['EMP0001', 'EMP0002', 'EMP0003'],
		'Full_Name': ['Ana', 'Luis', 'Sofía'],
		'Department': ['IT', 'HR', 'IT'],
		'Performance_Rating': [3, 4, 5],
	}
	df = pd.DataFrame(data)
	df_clean = clean_dataframe(df)
	# Debe renombrar a id_empleado y mantener unicidad
	assert 'id_empleado' in df_clean.columns
	assert df_clean['id_empleado'].nunique() == 3
	# Debe mantener tipo string
	assert df_clean['id_empleado'].dtype == object or str(df_clean['id_empleado'].dtype).startswith('string')
