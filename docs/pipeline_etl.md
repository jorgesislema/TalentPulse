# Pipeline ETL - TalentPulse

## Resumen

El pipeline ETL de TalentPulse procesa datos de recursos humanos de forma automatizada, incluyendo limpieza, validación de calidad, construcción de features y exportación.

## 🚀 Ejecución Rápida

```bash
# Activar entorno virtual
.venv\Scripts\activate

# Ejecutar pipeline completo
python -m src.etl.pipeline
```

## 📂 Estructura de Archivos

### Archivos ETL Principales
- `src/etl/clean_hr.py` - Limpieza y normalización de datos
- `src/etl/pipeline.py` - Orquestador principal del pipeline
- `src/etl/run_quality_checks.py` - Validaciones de calidad
- `src/etl/feature_build.py` - Construcción de características
- `src/etl/export_csv_warehouse.py` - Exportación a warehouse
- `src/etl/to_parquet.py` - Conversión a formato Parquet

### Archivos de Configuración
- `configs/params_etl.yaml` - Configuración de limpieza
- `configs/params_pipeline.yaml` - Configuración del pipeline

### Datos Generados
- `Data/processed/clean_hr.csv` - Datos limpios
- `Data/processed/hr_with_features.csv` - Datos con features
- `Data/warehouse/hr_final.csv` - Datos finales para warehouse
- `artifacts/metrics/quality_report.json` - Reporte de calidad

## 🔧 Componentes del Pipeline

### 1. Limpieza de Datos (`clean_hr.py`)
- Estandarización de nombres de columnas
- Conversión de tipos de datos
- Normalización de valores categóricos
- Detección y corrección de outliers
- Eliminación de duplicados
- Validaciones críticas de calidad

### 2. Validación de Calidad (`run_quality_checks.py`)
- Análisis de valores nulos
- Detección de duplicados
- Identificación de outliers (método IQR)
- Análisis de unicidad
- Estadísticas por tipo de variable
- Generación de reporte JSON

### 3. Construcción de Features (`feature_build.py`)

#### Features de Antigüedad
- `antiguedad_dias` - Días desde contratación
- `antiguedad_anos` - Años desde contratación
- `categoria_antiguedad` - Nuevo/Junior/Intermedio/Senior/Veterano
- `ano_contratacion` - Año de contratación
- `mes_contratacion` - Mes de contratación

#### Features de Salario
- `salario_percentil` - Percentil salarial (0-100)
- `categoria_salario` - Bajo/Medio-Bajo/Medio-Alto/Alto
- `salario_normalizado` - Z-score del salario

#### Features de Desempeño
- `categoria_desempeno` - Bajo/Regular/Bueno/Excelente
- `alto_desempeno` - Flag binario (top 20%)

#### Features Demográficas
- `categoria_experiencia` - Entry/Junior/Mid/Senior/Expert

### 4. Exportación (`export_csv_warehouse.py`)
- Exportación a warehouse con timestamp
- Generación de archivo principal (hr_final.csv)
- Backup histórico con timestamp
- Reporte resumen en texto plano

## 📊 Archivos de Salida

### Datos Procesados
```
Data/
├── processed/
│   ├── clean_hr.csv                    # Datos limpios (11 columnas)
│   ├── hr_with_features.csv            # Con features (23 columnas)
│   └── hr_with_features.parquet        # Versión Parquet
└── warehouse/
    ├── hr_final.csv                    # Datos finales
    └── hr_final_20250906_161028.csv    # Backup con timestamp
```

### Reportes y Métricas
```
artifacts/
├── metrics/
│   ├── clean_stats.json               # Estadísticas de limpieza
│   └── quality_report.json            # Reporte de calidad completo
└── reports/
    └── summary_report.txt              # Resumen ejecutivo
```

## 🔍 Validaciones de Calidad

### Reglas Críticas
- **Unicidad**: `id_empleado` debe ser único
- **No nulos**: Columnas críticas no pueden tener valores nulos
- **Dominios**: Valores categóricos dentro de rangos permitidos
- **Outliers**: Detección usando método IQR (rango intercuartil)

### Umbrales de Calidad
- Máximo 5% de valores nulos por columna
- Máximo 1% de filas duplicadas
- Mínimo 95% de unicidad para IDs

## 🛠️ Ejecución Individual

### Ejecutar solo limpieza
```bash
python -m src.etl.clean_hr
```

### Ejecutar solo validaciones
```bash
python -m src.etl.run_quality_checks
```

### Ejecutar solo features
```bash
python -m src.etl.feature_build
```

### Ejecutar solo exportación
```bash
python -m src.etl.export_csv_warehouse
```

## 📋 Logs y Monitoreo

El pipeline genera logs detallados que incluyen:
- Timestamps de cada paso
- Número de filas procesadas
- Errores y advertencias
- Estadísticas de transformaciones
- Tiempos de ejecución

Ejemplo de log típico:
```
2025-09-06 16:08:28,719 INFO etl_pipeline: 🚀 Iniciando pipeline ETL completo...
2025-09-06 16:08:28,720 INFO etl_pipeline: 🧹 Iniciando limpieza de datos...
2025-09-06 16:09:09,402 INFO etl_pipeline: ✅ Limpieza completada
2025-09-06 16:10:57,807 INFO etl_pipeline: 🎉 ¡Pipeline ETL exitoso!
```

## 🔧 Configuración

### Variables de Entorno (Opcional)
Crea un archivo `.env` para configuraciones sensibles:
```
DB_HOST=localhost
DB_USER=hr_user
DB_PASSWORD=secure_password
DB_NAME=hr_analytics
```

### Personalización
Modifica `configs/params_pipeline.yaml` para:
- Cambiar umbrales de calidad
- Habilitar/deshabilitar features específicas
- Configurar formatos de exportación
- Ajustar configuración de logging

## ⚡ Rendimiento

### Tiempos Típicos (2M registros)
- Limpieza: ~40 segundos
- Validación: ~10 segundos
- Features: ~25 segundos
- Exportación: ~60 segundos
- **Total**: ~2.5 minutos

### Optimizaciones
- Usa formato Parquet para mejor rendimiento
- Ajusta `chunksize` para datasets muy grandes
- Considera paralelización para múltiples archivos

## 🧪 Testing

Ejecuta las pruebas del pipeline:
```bash
pytest tests/test_etl.py -v
```

## 📞 Soporte

Para problemas o mejoras:
1. Revisa los logs en `artifacts/logs/`
2. Verifica la configuración en `configs/`
3. Consulta este documento
4. Contacta al equipo de Data Engineering
