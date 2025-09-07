# src/etl/clean_hr.py
# -*- coding: utf-8 -*-
"""
Limpieza y normalización del dataset HR_Data_MNC_Data Science Lovers
=====================================================================

QUÉ HACE
--------
1) Estandariza columnas a un esquema canónico en inglés (interno del script).
2) Limpia/normaliza: tipos, dominios (overtime), nulos, duplicados, outliers (winsor 1%-99%).
3) Renombra las columnas canónicas al **español** para una mejor comprención.
4) Valida reglas críticas (unicidad y rompe si fallan .
5) Guarda el resultado en Parquet o CSV .
6) Escribe una auditoría mínima (filas in/out y columnas).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
from src.config import load_settings  # lee .env + configs/*.yaml

# Logger básico si no tienes uno central
try:
    from src.logging import get_logger
except Exception:
    def get_logger(name: str = "clean_hr"):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        return logging.getLogger(name)

logger = get_logger("clean_hr")

# --- Mapeos de nombres (entrada variada → canónico EN → salida ES) -----------------

# 1) Versiones “como vengan” → nombres canónicos en inglés (interno)
TO_CANONICAL_EN: Dict[str, str] = {
    # variantes comunes → canónico
    "employee_id": "employee_id",
    "employeeid": "employee_id",
    "Employee_ID": "employee_id",
    "Full_Name": "full_name",
    "Department": "department",
    "Job_Title": "job_title",
    "Hire_Date": "hire_date",
    "Location": "location",
    "Performance_Rating": "performance_rating",
    "Experience_Years": "experience_years",
    "Status": "status",
    "Work_Mode": "work_mode",
    "Salary_INR": "salary_inr",
}

# 2) Canónico EN → nombre final ES (lo que verá BI/PowerBI/usuarios)
EN_TO_ES: Dict[str, str] = {
    "employee_id": "id_empleado",
    "full_name": "nombre_completo",
    "department": "departamento",
    "job_title": "puesto",
    "hire_date": "fecha_contratacion",
    "location": "ubicacion",
    "performance_rating": "calificacion_desempeno",
    "experience_years": "anios_experiencia",
    "status": "estatus",
    "work_mode": "modalidad_trabajo",
    "salary_inr": "salario_inr",
}

# Inversa para resolver validaciones que vengan en ES
ES_TO_EN = {v: k for k, v in EN_TO_ES.items()}


# ------------------------------- Helpers E/S --------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    """Lee CSV o Parquet con Pandas (requiere pyarrow/fastparquet para Parquet)."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_parquet_or_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Escribe Parquet si es posible; si falla (sin pyarrow), escribe CSV como fallback.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix.lower() != ".parquet":
            path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)  # necesita pyarrow/fastparquet
        logger.info(f"✔ Guardado Parquet: {path}")
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning(f"⚠ Parquet no disponible ({e}). Se guardó CSV: {csv_path}")


def _to_lower_snake(name: str) -> str:
    """Convierte nombres arbitrarios a snake_case (minúsculas + guión bajo)."""
    return (
        name.strip()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("__", "_").lower()
    )


def _winsor_limits(series: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> Tuple[float, float]:
    """Límites inferior/superior para winsorizar (recortar outliers suaves)."""
    lo = float(series.quantile(q_low))
    hi = float(series.quantile(q_high))
    return lo, hi


def _resolve_col(name: str, cols: List[str]) -> str | None:
    """
    Dada una columna mencionada en reglas (puede venir en EN o ES),
    devuelve el nombre real presente en el DataFrame final.
    """
    if name in cols:
        return name
    # Si vino en EN pero el DF ya está en ES
    if name in EN_TO_ES and EN_TO_ES[name] in cols:
        return EN_TO_ES[name]
    # Si vino en ES pero el DF está en EN (antes de renombrar)
    if name in ES_TO_EN and ES_TO_EN[name] in cols:
        return ES_TO_EN[name]
    return None


# ----------------------------- Limpieza principal ---------------------------------

def _standardize_to_canonical_en(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Normaliza nombres a snake_case.
    2) Mapea a nombres canónicos en inglés (interno).
    """
    df = df.copy()
    df.columns = [_to_lower_snake(c) for c in df.columns]

    existing = set(df.columns)
    rename = {c: TO_CANONICAL_EN[c] for c in TO_CANONICAL_EN if c in existing}
    if rename:
        df = df.rename(columns=rename)
    return df


def clean_dataframe(df: pd.DataFrame, rename_to_spanish: bool = True) -> pd.DataFrame:
    """
    Aplica limpieza estable de negocio y (opcional) renombra columnas al español.

    Pasos:
      - Estándar de columnas → canónico EN.
      - Trim/normalización básica de texto.
      - Dominios: overtime Yes/No → 1/0. (attrition idem si viene textual)
      - Casting numérico seguro y fechas tolerantes.
      - Valores negativos defensivos (ingreso_mensual) + winsorización 1%-99%.
      - Deduplicación (employee_id + snapshot_date si existen).
      - Renombrar a español (si rename_to_spanish=True).
    """
    df = _standardize_to_canonical_en(df)

    # 2) Texto: trim y colapsar espacios (heurística simple por dtype)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # 3) Dominios/casting básicos (overtime, attrition)
    if "overtime" in df.columns:
        s = df["overtime"].astype("string").str.lower()
        df["overtime"] = s.isin(["yes", "y", "true", "1", "si", "sí"]).astype("int8")

    if "attrition" in df.columns:
        # Si llega textual ("Yes"/"No"), lo llevamos a 0/1
        s = df["attrition"].astype("string").str.lower()
        mask_known = s.isin(["yes", "y", "true", "1", "no", "n", "false", "0", "si", "sí"])
        if mask_known.any():
            df.loc[mask_known, "attrition"] = s[mask_known].isin(["yes", "y", "true", "1", "si", "sí"]).astype("int8")
        df["attrition"] = pd.to_numeric(df["attrition"], errors="coerce").astype("Float64")

    # 4) Numéricos seguros (NO convertir employee_id, es string)
    for c in ["performance_rating", "age", "overtime_hours", "monthly_income"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Fechas tolerantes
    for dcol in ("hire_date", "snapshot_date"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date

    # 6) Defensivo: ingresos negativos → NaN; winsor 1%-99% si hay suficiente volumen
    if "monthly_income" in df.columns:
        df.loc[df["monthly_income"] < 0, "monthly_income"] = pd.NA
        if df["monthly_income"].notna().sum() >= 1000:
            lo, hi = _winsor_limits(df["monthly_income"].dropna(), 0.01, 0.99)
            df["monthly_income"] = df["monthly_income"].clip(lower=lo, upper=hi)

    # 7) Deduplicación (preferente por clave compuesta)
    if {"employee_id", "snapshot_date"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["employee_id", "snapshot_date"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    # 8) Renombrar al español para el output final (si se pide)
    if rename_to_spanish:
        cols = df.columns.tolist()
        # Renombramos solo las que estén mapeadas EN→ES
        rename_es = {en: EN_TO_ES[en] for en in EN_TO_ES if en in cols}
        df = df.rename(columns=rename_es)

    return df


# ---------------------------- Validaciones críticas --------------------------------

def quick_validate(df: pd.DataFrame, etl_cfg: Any) -> None:
    """
    Reglas críticas desde YAML (pueden venir en ES o EN).
    Rompe el pipeline con ValueError si falla.
    """
    crit: Dict[str, Any] = getattr(etl_cfg, "quality_critical", {}) or {}
    errors: List[str] = []

    # Unicidad
    for name in (crit.get("unique") or []):
        col = _resolve_col(name, df.columns.tolist())
        if col and col in df.columns:
            uniques = int(df[col].nunique(dropna=True))
            total = int(len(df))
            if uniques != total:
                errors.append(f"Unicidad violada en {col}: únicos={uniques}, filas={total}")

    # No-negativos
    for name in (crit.get("non_negative") or []):
        col = _resolve_col(name, df.columns.tolist())
        if col and col in df.columns:
            bad = int((df[col] < 0).sum(skipna=True))
            if bad > 0:
                errors.append(f"Valores negativos en {col}: {bad}")

    # Dominios categóricos/numéricos
    for name, allowed in (crit.get("domain") or {}).items():
        col = _resolve_col(name, df.columns.tolist())
        if col and col in df.columns:
            bad = int((~df[col].isin(allowed)).sum(skipna=True))
            if bad > 0:
                errors.append(f"Valores fuera de dominio en {col}: {bad} (permitidos={allowed})")

    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError(" Validaciones críticas fallaron. Revisa reglas y datos.")


# ------------------------------- Auditoría simple ----------------------------------

def audit_stats(df_before: pd.DataFrame, df_after: pd.DataFrame, out_path: Path) -> None:
    """Guarda filas in/out y columnas (trazabilidad mínima)."""
    stats = {
        "rows_in": int(len(df_before)),
        "rows_out": int(len(df_after)),
        "columns": list(map(str, df_after.columns)),
        "engine": "pandas",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------- Main runner ------------------------------------

def main() -> None:
    """
    Orquesta: lee config → elige entrada → limpia → valida → guarda → audita.
    - Si en tu YAML pusiste `rename_to_spanish: false`, el output se queda en EN canónico.
    """
    app_cfg, etl_cfg, _ = load_settings()

    in_path = Path(getattr(etl_cfg, "interim_path", None) or etl_cfg.raw_path)
    out_path = Path(etl_cfg.processed_path)
    audit_out = Path("artifacts/metrics/clean_stats.json")
    rename_to_spanish = bool(getattr(etl_cfg, "rename_to_spanish", True))

    t0 = time.time()
    logger.info(f"🔹 Leyendo datos de: {in_path}")
    df0 = _read_any(in_path)

    logger.info("🔹 Limpiando/normalizando…")
    df = clean_dataframe(df0, rename_to_spanish=rename_to_spanish)

    logger.info("🔹 Validaciones rápidas…")
    quick_validate(df, etl_cfg)

    logger.info(f"🔹 Guardando procesado en: {out_path}")
    _write_parquet_or_csv(df, out_path)

    audit_stats(df0, df, audit_out)
    logger.info(f" Listo en {time.time() - t0:0.2f}s → {out_path}")


if __name__ == "__main__":
    main()
