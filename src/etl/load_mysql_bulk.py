# src/etl/load_mysql_bulk.py
# -*- coding: utf-8 -*-
"""
Carga masiva a MySQL
===================

Funciones para cargar grandes volúmenes de datos a MySQL de forma eficiente.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from src.logging import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("mysql_bulk")

def create_mysql_connection(host: str, database: str, user: str, password: str, port: int = 3306):
    """
    Crea conexión a MySQL usando SQLAlchemy.
    
    Requiere: pip install pymysql sqlalchemy
    """
    try:
        from sqlalchemy import create_engine
        
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)
        
        logger.info(f"✅ Conectado a MySQL: {host}:{port}/{database}")
        return engine
        
    except ImportError:
        logger.error("❌ Requiere: pip install pymysql sqlalchemy")
        return None
    except Exception as e:
        logger.error(f"❌ Error de conexión MySQL: {e}")
        return None

def bulk_insert_pandas(df: pd.DataFrame, table_name: str, engine, if_exists: str = "append", 
                      chunksize: int = 10000) -> bool:
    """
    Carga DataFrame a MySQL usando pandas.to_sql().
    
    Args:
        df: DataFrame a cargar
        table_name: Nombre de la tabla destino
        engine: Conexión SQLAlchemy
        if_exists: 'fail', 'replace', 'append'
        chunksize: Tamaño de lote para inserción
    
    Returns:
        True si la carga fue exitosa
    """
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi"
        )
        
        logger.info(f"✅ {len(df)} filas cargadas a tabla '{table_name}'")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en carga masiva: {e}")
        return False

def bulk_insert_csv_direct(csv_path: str, table_name: str, engine, 
                          columns: Optional[list] = None) -> bool:
    """
    Carga CSV directamente usando LOAD DATA INFILE (más rápido).
    
    Nota: Requiere permisos FILE en MySQL y archivo accesible al servidor.
    """
    try:
        # Para uso futuro cuando se tengan permisos FILE
        logger.warning("LOAD DATA INFILE requiere permisos especiales en MySQL")
        
        # Alternativa: leer CSV y usar bulk_insert_pandas
        df = pd.read_csv(csv_path)
        if columns:
            df = df[columns]
        
        return bulk_insert_pandas(df, table_name, engine)
        
    except Exception as e:
        logger.error(f"❌ Error en carga CSV directa: {e}")
        return False

def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine) -> str:
    """
    Genera DDL para crear tabla basada en DataFrame.
    
    Returns:
        SQL DDL como string
    """
    type_mapping = {
        'object': 'VARCHAR(255)',
        'int64': 'BIGINT',
        'float64': 'DOUBLE',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'DATETIME',
        'category': 'VARCHAR(100)'
    }
    
    columns = []
    for col, dtype in df.dtypes.items():
        mysql_type = type_mapping.get(str(dtype), 'TEXT')
        columns.append(f"`{col}` {mysql_type}")
    
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        {', '.join(columns)}
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    logger.info(f"DDL generado para tabla '{table_name}'")
    return ddl

def setup_hr_tables(engine) -> bool:
    """Crea las tablas necesarias para datos de RRHH."""
    try:
        # Tabla principal de empleados
        hr_table_ddl = """
        CREATE TABLE IF NOT EXISTS `empleados` (
            `id_empleado` VARCHAR(20) PRIMARY KEY,
            `nombre_completo` VARCHAR(255),
            `departamento` VARCHAR(100),
            `puesto` VARCHAR(150),
            `fecha_contratacion` DATE,
            `ubicacion` VARCHAR(100),
            `calificacion_desempeno` INT,
            `anios_experiencia` INT,
            `estatus` VARCHAR(50),
            `modalidad_trabajo` VARCHAR(50),
            `salario_inr` BIGINT,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        with engine.connect() as conn:
            conn.execute(hr_table_ddl)
            logger.info("✅ Tabla 'empleados' creada/verificada")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creando tablas: {e}")
        return False

def main():
    """Ejemplo de uso con datos de RRHH."""
    # Configuración de conexión (usar variables de entorno en producción)
    config = {
        "host": "localhost",
        "database": "hr_analytics",
        "user": "hr_user",
        "password": "password123"
    }
    
    # Conectar
    engine = create_mysql_connection(**config)
    if not engine:
        return
    
    # Configurar tablas
    setup_hr_tables(engine)
    
    # Cargar datos procesados
    try:
        df = pd.read_csv("Data/processed/clean_hr.csv")
        success = bulk_insert_pandas(df, "empleados", engine, if_exists="replace")
        
        if success:
            logger.info("🎉 Carga masiva completada exitosamente")
        
    except FileNotFoundError:
        logger.error("❌ Archivo clean_hr.csv no encontrado")
    except Exception as e:
        logger.error(f"❌ Error en proceso principal: {e}")

if __name__ == "__main__":
    main()
