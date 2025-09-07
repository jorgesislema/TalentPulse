# src/etl/export_csv_warehouse.py
# -*- coding: utf-8 -*-
"""
Exportación a Warehouse
=======================

Funciones para exportar datos procesados al warehouse.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

try:
    from src.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("export_warehouse")

def export_csv(df: pd.DataFrame, path: str) -> None:
    """Exporta DataFrame a CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"✅ CSV exportado: {output_path} ({len(df)} filas)")

def export_to_warehouse(input_path: str = "Data/processed/hr_with_features.csv", 
                       output_dir: str = "Data/warehouse") -> None:
    """
    Exporta datos al warehouse con timestamp.
    
    Args:
        input_path: Ruta del archivo a exportar
        output_dir: Directorio de warehouse
    """
    try:
        # Cargar datos
        df = pd.read_csv(input_path)
        
        # Crear directorio warehouse
        warehouse_path = Path(output_dir)
        warehouse_path.mkdir(parents=True, exist_ok=True)
        
        # Archivo principal (siempre actualizado)
        main_file = warehouse_path / "hr_final.csv"
        df.to_csv(main_file, index=False)
        
        # Archivo con timestamp (histórico)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_file = warehouse_path / f"hr_final_{timestamp}.csv"
        df.to_csv(timestamped_file, index=False)
        
        logger.info(f"✅ Warehouse actualizado: {main_file}")
        logger.info(f"✅ Backup creado: {timestamped_file}")
        
    except Exception as e:
        logger.error(f"❌ Error exportando a warehouse: {e}")
        raise

def export_summary_report(input_path: str = "Data/processed/hr_with_features.csv",
                         output_path: str = "artifacts/reports/summary_report.txt") -> None:
    """Genera reporte resumen de los datos."""
    try:
        df = pd.read_csv(input_path)
        
        # Crear reporte
        report = []
        report.append("REPORTE RESUMEN - DATOS DE RRHH")
        report.append("=" * 50)
        report.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Estadísticas básicas
        report.append("ESTADÍSTICAS GENERALES")
        report.append("-" * 25)
        report.append(f"Total empleados: {len(df):,}")
        report.append(f"Total columnas: {len(df.columns)}")
        report.append("")
        
        # Por departamento
        if "departamento" in df.columns:
            report.append("DISTRIBUCIÓN POR DEPARTAMENTO")
            report.append("-" * 35)
            dept_counts = df["departamento"].value_counts()
            for dept, count in dept_counts.head(10).items():
                report.append(f"  {dept}: {count:,} empleados")
            report.append("")
        
        # Por estatus
        if "estatus" in df.columns:
            report.append("DISTRIBUCIÓN POR ESTATUS")
            report.append("-" * 30)
            status_counts = df["estatus"].value_counts()
            for status, count in status_counts.items():
                pct = (count / len(df)) * 100
                report.append(f"  {status}: {count:,} ({pct:.1f}%)")
            report.append("")
        
        # Salarios
        if "salario_inr" in df.columns:
            report.append("ESTADÍSTICAS SALARIALES")
            report.append("-" * 25)
            report.append(f"  Salario promedio: ₹{df['salario_inr'].mean():,.0f}")
            report.append(f"  Salario mediano: ₹{df['salario_inr'].median():,.0f}")
            report.append(f"  Salario mínimo: ₹{df['salario_inr'].min():,.0f}")
            report.append(f"  Salario máximo: ₹{df['salario_inr'].max():,.0f}")
            report.append("")
        
        # Guardar reporte
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        
        logger.info(f"✅ Reporte resumen creado: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error generando reporte: {e}")
        raise

def main():
    """Ejemplo de uso."""
    try:
        # Exportar a warehouse
        export_to_warehouse()
        
        # Generar reporte resumen
        export_summary_report()
        
        logger.info("🎉 Exportación completa")
        
    except Exception as e:
        logger.error(f"💥 Error en exportación: {e}")

if __name__ == "__main__":
    main()
