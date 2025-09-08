# src/etl/run_quality_checks.py
# -*- coding: utf-8 -*-
"""
Validaciones de calidad de datos
===============================

Funciones para ejecutar chequeos de calidad sobre datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

try:
    from src.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("quality_checks")

def check_nulls(df: pd.DataFrame) -> pd.Series:
    """Retorna conteo de nulos por columna."""
    return df.isnull().sum()

def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
    """Retorna cantidad de filas duplicadas."""
    return df.duplicated(subset=subset).sum()

def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Retorna tipos de datos por columna."""
    return df.dtypes.to_dict()

def check_outliers_iqr(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Detecta outliers usando método IQR."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Columna {column} no es numérica"}
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        "outliers_count": len(outliers),
        "outliers_percentage": len(outliers) / len(df) * 100,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

def check_unique_values(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Analiza unicidad de valores en una columna."""
    unique_count = df[column].nunique()
    total_count = len(df)
    
    return {
        "unique_values": unique_count,
        "total_values": total_count,
        "uniqueness_ratio": unique_count / total_count,
        "is_unique": unique_count == total_count
    }

def run_comprehensive_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Ejecuta un chequeo de calidad completo."""
    logger.info("🔍 Iniciando chequeos de calidad...")
    
    results = {
        "dataset_shape": df.shape,
        "null_counts": check_nulls(df).to_dict(),
        "duplicate_rows": check_duplicates(df),
        "data_types": check_data_types(df),
        "numeric_outliers": {},
        "categorical_analysis": {}
    }
    
    # Análisis de outliers para columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        results["numeric_outliers"][col] = check_outliers_iqr(df, col)
    
    # Análisis categórico
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        results["categorical_analysis"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    
    logger.info("✅ Chequeos de calidad completados")
    return results

def main():
    """Ejemplo de uso."""
    df = pd.read_csv("Data/processed/clean_hr.csv")
    results = run_comprehensive_quality_check(df)
    
    # Mostrar resumen
    print(f"Dataset: {results['dataset_shape']}")
    print(f"Duplicados: {results['duplicate_rows']}")
    print("Nulos por columna:", results['null_counts'])

if __name__ == "__main__":
    main()
