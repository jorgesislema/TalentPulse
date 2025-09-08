# src/etl/pipeline.py
# -*- coding: utf-8 -*-
"""
Pipeline ETL Principal
=====================

Orquesta todo el proceso ETL: limpieza, validación, features y exportación.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

try:
    from src.logging import get_logger
    from src.config import load_settings
    from src.etl.clean_hr import main as clean_main
    from src.etl.run_quality_checks import run_comprehensive_quality_check
    from src.etl.feature_build import build_all_features
    from src.etl.to_parquet import save_parquet
    from src.etl.export_csv_warehouse import export_to_warehouse
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    print(f"⚠️ Import error: {e}")
    print("Ejecuta desde la raíz del proyecto con: python -m src.etl.pipeline")

logger = get_logger("etl_pipeline")

def run_data_cleaning() -> bool:
    """Ejecuta la limpieza de datos."""
    try:
        logger.info("🧹 Iniciando limpieza de datos...")
        clean_main()
        logger.info("✅ Limpieza completada")
        return True
    except Exception as e:
        logger.error(f"❌ Error en limpieza: {e}")
        return False

def run_quality_validation() -> Dict[str, Any]:
    """Ejecuta validaciones de calidad."""
    try:
        import pandas as pd
        logger.info("🔍 Ejecutando validaciones de calidad...")
        
        df = pd.read_csv("Data/processed/clean_hr.csv")
        results = run_comprehensive_quality_check(df)
        
        # Guardar reporte de calidad
        import json
        report_path = Path("artifacts/metrics/quality_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Reporte de calidad guardado: {report_path}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en validación: {e}")
        return {}

def run_feature_engineering() -> bool:
    """Ejecuta construcción de features."""
    try:
        import pandas as pd
        logger.info("🔧 Construyendo features...")
        
        df = pd.read_csv("Data/processed/clean_hr.csv")
        df_with_features = build_all_features(df)
        
        # Guardar dataset con features
        output_path = "Data/processed/hr_with_features.csv"
        df_with_features.to_csv(output_path, index=False)
        
        logger.info(f"✅ Features creadas: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en feature engineering: {e}")
        return False

def run_data_export() -> bool:
    """Ejecuta exportación de datos."""
    try:
        logger.info("📤 Exportando datos...")
        
        # Exportar a warehouse
        export_to_warehouse()
        
        # Crear versión Parquet si es posible
        try:
            import pandas as pd
            df = pd.read_csv("Data/processed/hr_with_features.csv")
            save_parquet(df, "Data/processed/hr_with_features.parquet")
        except Exception as parquet_error:
            logger.warning(f"No se pudo crear Parquet: {parquet_error}")
        
        logger.info("✅ Exportación completada")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en exportación: {e}")
        return False

def run_full_pipeline() -> Dict[str, bool]:
    """
    Ejecuta el pipeline ETL completo.
    
    Returns:
        Diccionario con el estado de cada paso
    """
    start_time = time.time()
    logger.info("🚀 Iniciando pipeline ETL completo...")
    
    results = {
        "cleaning": False,
        "quality_validation": False,
        "feature_engineering": False,
        "data_export": False
    }
    
    # Paso 1: Limpieza
    results["cleaning"] = run_data_cleaning()
    if not results["cleaning"]:
        logger.error("❌ Pipeline abortado: falla en limpieza")
        return results
    
    # Paso 2: Validación de calidad
    quality_results = run_quality_validation()
    results["quality_validation"] = bool(quality_results)
    
    # Paso 3: Feature engineering
    results["feature_engineering"] = run_feature_engineering()
    
    # Paso 4: Exportación
    results["data_export"] = run_data_export()
    
    # Resumen final
    total_time = time.time() - start_time
    success_count = sum(results.values())
    
    logger.info(f"🎯 Pipeline completado en {total_time:.2f}s")
    logger.info(f"📊 Éxito: {success_count}/4 pasos")
    
    if success_count == 4:
        logger.info("🎉 ¡Pipeline ETL exitoso!")
    else:
        logger.warning("⚠️ Pipeline completado con errores")
    
    return results

def main():
    """Punto de entrada principal."""
    try:
        results = run_full_pipeline()
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DEL PIPELINE ETL")
        print("="*50)
        for step, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"{step.replace('_', ' ').title():<20} {status}")
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrumpido por el usuario")
    except Exception as e:
        logger.error(f"💥 Error crítico en pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
