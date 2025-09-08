# src/etl/feature_build.py
# -*- coding: utf-8 -*-
"""
Construcción de características (features)
==========================================

Funciones para crear nuevas features derivadas de datos de RRHH.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

try:
    from src.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("feature_build")

def build_tenure_features(df: pd.DataFrame, hire_date_col: str = "fecha_contratacion") -> pd.DataFrame:
    """
    Crea features relacionadas con antigüedad del empleado.
    
    Args:
        df: DataFrame con datos de empleados
        hire_date_col: Nombre de la columna de fecha de contratación
    
    Returns:
        DataFrame con nuevas features de antigüedad
    """
    df = df.copy()
    
    if hire_date_col not in df.columns:
        logger.warning(f"Columna {hire_date_col} no encontrada")
        return df
    
    # Convertir a datetime si no lo está
    df[hire_date_col] = pd.to_datetime(df[hire_date_col], errors="coerce")
    
    # Fecha de referencia (hoy)
    today = datetime.now()
    
    # Antigüedad en días y años
    df["antiguedad_dias"] = (today - df[hire_date_col]).dt.days
    df["antiguedad_anos"] = df["antiguedad_dias"] / 365.25
    
    # Categorías de antigüedad
    df["categoria_antiguedad"] = pd.cut(
        df["antiguedad_anos"],
        bins=[0, 1, 3, 5, 10, float("inf")],
        labels=["Nuevo", "Junior", "Intermedio", "Senior", "Veterano"]
    )
    
    # Año de contratación
    df["ano_contratacion"] = df[hire_date_col].dt.year
    
    # Mes de contratación
    df["mes_contratacion"] = df[hire_date_col].dt.month
    
    logger.info(f"✅ Features de antigüedad creadas para {len(df)} empleados")
    return df

def build_salary_features(df: pd.DataFrame, salary_col: str = "salario_inr") -> pd.DataFrame:
    """Crea features relacionadas con salario."""
    df = df.copy()
    
    if salary_col not in df.columns:
        logger.warning(f"Columna {salary_col} no encontrada")
        return df
    
    # Percentiles de salario
    df["salario_percentil"] = df[salary_col].rank(pct=True) * 100
    
    # Categorías de salario por cuartiles
    df["categoria_salario"] = pd.qcut(
        df[salary_col],
        q=4,
        labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"],
        duplicates="drop"
    )
    
    # Salario normalizado (z-score)
    df["salario_normalizado"] = (df[salary_col] - df[salary_col].mean()) / df[salary_col].std()
    
    logger.info(f"✅ Features de salario creadas")
    return df

def build_performance_features(df: pd.DataFrame, performance_col: str = "calificacion_desempeno") -> pd.DataFrame:
    """Crea features relacionadas con desempeño."""
    df = df.copy()
    
    if performance_col not in df.columns:
        logger.warning(f"Columna {performance_col} no encontrada")
        return df
    
    # Categorías de desempeño
    df["categoria_desempeno"] = pd.cut(
        df[performance_col],
        bins=[0, 2, 3, 4, 5],
        labels=["Bajo", "Regular", "Bueno", "Excelente"],
        include_lowest=True
    )
    
    # Flag de alto desempeño (top 20%)
    df["alto_desempeno"] = (df[performance_col] >= df[performance_col].quantile(0.8)).astype(int)
    
    logger.info(f"✅ Features de desempeño creadas")
    return df

def build_demographic_features(df: pd.DataFrame, experience_col: str = "anios_experiencia") -> pd.DataFrame:
    """Crea features demográficas y de experiencia."""
    df = df.copy()
    
    if experience_col in df.columns:
        # Categorías de experiencia
        df["categoria_experiencia"] = pd.cut(
            df[experience_col],
            bins=[0, 2, 5, 10, 20, float("inf")],
            labels=["Entry", "Junior", "Mid", "Senior", "Expert"]
        )
    
    logger.info(f"✅ Features demográficas creadas")
    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las features disponibles.
    
    Args:
        df: DataFrame con datos originales
    
    Returns:
        DataFrame con todas las features agregadas
    """
    logger.info("🔧 Iniciando construcción de features...")
    
    # Aplicar todas las transformaciones
    df = build_tenure_features(df)
    df = build_salary_features(df)
    df = build_performance_features(df)
    df = build_demographic_features(df)
    
    logger.info(f"✅ Features construidas. Shape final: {df.shape}")
    return df

def main():
    """Ejemplo de uso."""
    df = pd.read_csv("Data/processed/clean_hr.csv")
    df_with_features = build_all_features(df)
    
    # Guardar resultado
    df_with_features.to_csv("Data/processed/hr_with_features.csv", index=False)
    logger.info("💾 Archivo con features guardado")

if __name__ == "__main__":
    main()
