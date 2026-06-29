"""
Módulo de Entrenamiento de Modelo ML (MLOps).
Extrae datos limpios, selecciona características, construye un Pipeline 
(Preprocesamiento + Random Forest), entrena y serializa el modelo final.
"""
import sys
import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- 1. EL ARREGLO DEL RADAR DEBE IR AQUÍ ---
DIRECTORIO_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIRECTORIO_SCRIPTS)

from common.database import get_db_engine 

logger = logging.getLogger(__name__)

# Rutas absolutas
BASE_DIR = os.path.dirname(DIRECTORIO_SCRIPTS)
RUTA_MODELS = os.path.join(BASE_DIR, 'models')
RUTA_MODELO_FINAL = os.path.join(RUTA_MODELS, 'pipeline_random_forest.pkl')

QUERY_DATOS_LIMPIOS = """
    SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
    FROM cliente_limpio c
    JOIN prestamo_limpio p ON c.id_cliente = p.id_cliente
"""

# Reglas de negocio y variables protegidas
VARIABLES_PROTEGIDAS = ['es_primer_empleo', 'porcentaje_vida_laboral', 'credit_score']
UMBRAL_CORRELACION = 0.05
TARGET = 'loan_status'

def extraer_datos_limpios() -> pd.DataFrame:
    """1. Extrae los datos limpios desde PostgreSQL."""
    logger.info("1. Extrayendo las tablas limpias desde PostgreSQL...")
    engine = get_db_engine()
    df = pd.read_sql(QUERY_DATOS_LIMPIOS, engine)
    logger.info(f" -> Extracción exitosa. Filas: {df.shape[0]:,}, Columnas: {df.shape[1]}")
    return df

def seleccion_de_variables(df: pd.DataFrame) -> pd.DataFrame:
    """2. Feature Selection basada en correlación y lógica de negocio."""
    logger.info("2. Realizando Selección de Variables (Feature Selection)...")
    df = df.copy()

    if 'id_cliente' in df.columns:
        df = df.drop(columns=['id_cliente'])
        logger.info(" -> Columna 'id_cliente' eliminada.")

    num_cols = df.select_dtypes(include=["number"]).columns
    correlaciones = df[num_cols].corr()[TARGET].abs()

    vars_baja_corr = correlaciones[correlaciones < UMBRAL_CORRELACION].index.tolist()
    vars_a_eliminar = [v for v in vars_baja_corr if v not in VARIABLES_PROTEGIDAS]
    vars_salvadas = [v for v in vars_baja_corr if v in VARIABLES_PROTEGIDAS]

    if vars_a_eliminar:
        df = df.drop(columns=vars_a_eliminar)
        logger.info(f" -> Variables eliminadas por baja correlación (< {UMBRAL_CORRELACION}): {vars_a_eliminar}")
    if vars_salvadas:
        logger.info(f" -> Variables PROTEGIDAS por lógica de negocio: {vars_salvadas}")

    return df

def entrenar_pipeline(df: pd.DataFrame):
    """3. Construye el Pipeline (Transformador + Clasificador) y lo entrena."""
    logger.info("3. Configurando Pipeline y dividiendo datos...")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # División 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_test.to_csv(os.path.join(BASE_DIR, 'data', 'X_test_crudo.csv'), index=False)
    y_test.to_csv(os.path.join(BASE_DIR, 'data', 'y_test.csv'), index=False)

    logger.info(f" -> Datos divididos: Train ({X_train.shape[0]:,} filas), Test ({X_test.shape[0]:,} filas)")

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Bloque A: El Transformador de Columnas
    preprocesador = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

    # Bloque B: El Clasificador
    clasificador = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42
    )

    # Bloque C: LA FUSIÓN (El Pipeline Final)
    pipeline = Pipeline([
        ('preprocesador', preprocesador),
        ('modelo', clasificador)
    ])

    logger.info("4. Entrenando el Pipeline completo (esto puede tomar unos segundos)...")
    # Al hacer fit al pipeline, limpia y entrena internamente
    pipeline.fit(X_train, y_train)
    logger.info(" -> Entrenamiento completado.")

    return pipeline

def guardar_modelo(pipeline):
    """4. Serializa el pipeline completo en el disco."""
    logger.info("5. Exportando el modelo entrenado...")
    os.makedirs(RUTA_MODELS, exist_ok=True)
    joblib.dump(pipeline, RUTA_MODELO_FINAL)
    logger.info(f"¡Modelo guardado exitosamente en {RUTA_MODELO_FINAL}!")

def ejecutar_entrenamiento() -> None:
    """Punto de entrada principal del módulo de entrenamiento."""
    logger.info("=== INICIANDO ENTRENAMIENTO MLOps ===")
    try:
        df = extraer_datos_limpios()
        df = seleccion_de_variables(df)
        pipeline_entrenado = entrenar_pipeline(df)
        guardar_modelo(pipeline_entrenado)
        logger.info("=== PIPELINE FINALIZADO CON ÉXITO ===")
    except Exception as e:
        logger.critical(f"EL PIPELINE FALLÓ: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    ejecutar_entrenamiento()