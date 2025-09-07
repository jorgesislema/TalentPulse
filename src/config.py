import yaml
from types import SimpleNamespace
from pathlib import Path

def load_settings():
	"""
	Carga la configuración ETL desde configs/params_etl.yaml y la retorna como objeto.
	Devuelve: (app_cfg, etl_cfg, _)
	"""
	# Cargar parámetros ETL
	etl_path = Path(__file__).parent.parent / 'configs' / 'params_etl.yaml'
	with open(etl_path, 'r', encoding='utf-8') as f:
		etl_dict = yaml.safe_load(f)
	# Normaliza nombres para compatibilidad con el script
	etl_dict = {k.replace('input_data_path', 'raw_path').replace('output_data_path', 'processed_path'): v for k, v in etl_dict.items()}
	etl_cfg = SimpleNamespace(**etl_dict)
	# app_cfg y _ pueden ser None si no se usan
	return None, etl_cfg, None
