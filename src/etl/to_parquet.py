# src/etl/to_parquet.py
# -*- coding: utf-8 -*-
"""
Conversión de datos a formato Parquet
====================================

Funciones para convertir datasets a Parquet con compresión y validación.
"""

import pandas as pd
from pathlib import Path
import logging

try:
    from src.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("to_parquet")

def save_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    """
    Guarda DataFrame como Parquet con compresión.
    
    Args:
        df: DataFrame a guardar
        path: Ruta del archivo de salida
        compression: Tipo de compresión (snappy, gzip, brotli)
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_parquet(output_path, index=False, compression=compression)
        logger.info(f"✅ Guardado Parquet: {output_path} ({len(df)} filas)")
    except Exception as e:
        logger.error(f"❌ Error al guardar Parquet: {e}")
        raise

def csv_to_parquet(csv_path: str, parquet_path: str, **kwargs) -> None:
    """Convierte CSV a Parquet directamente."""
    df = pd.read_csv(csv_path, **kwargs)
    save_parquet(df, parquet_path)
    
def main():
    """Ejemplo de uso desde línea de comandos."""
    csv_to_parquet(
        "Data/processed/clean_hr.csv", 
        "Data/processed/clean_hr.parquet"
    )

if __name__ == "__main__":
    main()
