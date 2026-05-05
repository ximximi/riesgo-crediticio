import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar las variables del archivo .env (útil si se corre localmente)
load_dotenv(os.path.join(BASE_DIR, '.env'))

class QualityCheck:
    def __init__(self, data: pd.DataFrame, exclude_inconsistencies: np.array = None):
        self.data = data
        # Verifica si se excluyen columnas
        if exclude_inconsistencies is None:
          self.exclude_inconsistencies = []
        else:
          self.exclude_inconsistencies = exclude_inconsistencies

    # Valores faltantes
    def has_nulls(self) -> bool:
        return self.data.isnull().values.any()

    # Duplicados
    def has_duplicates(self) -> bool:
        return self.data.duplicated().any()

    # Atípicos (IQR)
    def has_outliers(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])

        for col in numeric_cols.columns:
            Q1 = numeric_cols[col].quantile(0.25)
            Q3 = numeric_cols[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            if ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).any():
                return True

        return False

    # Inconsistencias negativas (valores negativos)
    def has_negative_values(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])
        numeric_cols = numeric_cols.drop(self.exclude_inconsistencies, axis=1, errors='ignore')
        for col in numeric_cols.columns:
            if (numeric_cols[col] < 0).any():
                return True

        return False

    # Inconsistencias categóricas
    def has_categorical_inconsistencies(self) -> bool:
        cat_cols = self.data.select_dtypes(include=["object"])

        for col in cat_cols.columns:
            values = cat_cols[col].dropna().astype(str)

            normalized = values.str.strip().str.lower()

            if len(values.unique()) != len(normalized.unique()):
                return True

        return False

    # Inconsistencias generales
    def has_inconsistencies(self) -> bool:
        return self.has_negative_values() or self.has_categorical_inconsistencies()


    # Reporte completo
    def quality_report(self) -> dict:
        return {
            "nulos/faltantes": bool(self.has_nulls()),
            "duplicados": bool(self.has_duplicates()),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
            "quality_score": self.quality_score_weighted()
        }

    # Calcula el score de calidad
    def quality_score_weighted(self) -> float:

        weights = {
            "nulos/faltantes": 0.3,
            "duplicados": 0.2,
            "outliers": 0.2,
            "inconsistencias": 0.3
        }

        checks = {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies()
        }

        penalty = 0

        for key in checks:
            if checks[key]:  # si hay problema
                penalty += weights[key]

        quality = (1 - penalty) * 100

        return round(quality, 2)


def procesar_datos():
    # 1. Extracción desde la BD
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    
    logging.info("Extrayendo datos desde PostgreSQL...")
    query = """
        SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
        FROM cliente c
        JOIN prestamo p ON c.id_cliente = p.id_cliente
    """
    df = pd.read_sql(query, engine)
    
    # 2. Quality Check Inicial
    logging.info("Ejecutando Quality Check Inicial (Crudo)...")
    qc_inicial = QualityCheck(df)
    reporte_inicial = qc_inicial.quality_report()
    logging.info(f"Reporte Inicial: {reporte_inicial}")
    
    # 2.5 Tratamiento de Inconsistencias y Limpieza
    logging.info("Iniciando tratamiento profundo de inconsistencias...")
    
    # a. Gestión de Duplicados
    # Utilizamos 'id_cliente' como identificador único. Si existe, borramos los clones.
    if 'id_cliente' in df.columns:
        df = df.drop_duplicates(subset=['id_cliente'])
        logging.info("Duplicados eliminados exitosamente basándose en 'id_cliente'.")
    else:
        df = df.drop_duplicates()
        
    # b. Limpieza de Inconsistencias de Texto (Categóricas)
    # Convertimos a minúsculas y quitamos espacios residuales para estandarizar categorías
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        # Convertir a minúsculas y hacer strip
        df[col] = df[col].astype(str).str.lower().str.strip()
        # Restaurar los nulos reales que el astype(str) convirtió a la palabra 'nan' o 'none'
        df[col] = df[col].replace('nan', np.nan).replace('none', np.nan)
        
    # c. Corrección de Valores Negativos
    # Aplicamos valor absoluto asumiendo que los negativos (ej. en edad o ingresos) son errores de tipeo
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col != 'id_cliente' and col != 'loan_status': # Excluir IDs y variable objetivo
            df[col] = df[col].abs()
            
    # d. Eliminación Crítica de Columnas
    # Si alguna variable supera el 50% de datos nulos, la eliminamos por completo
    umbral_nulos = len(df) * 0.5
    for col in list(df.columns):
        if df[col].isnull().sum() > umbral_nulos:
            df = df.drop(columns=[col])
            logging.info(f"Columna '{col}' eliminada por superar el 50% de datos nulos.")
            
    # Actualizar listas de columnas tras posibles eliminaciones
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # e. Eliminación de filas si falta la Variable Objetivo (loan_status)
    if 'loan_status' in df.columns:
        df = df.dropna(subset=['loan_status'])
        logging.info("Filas con 'loan_status' (variable objetivo) nulo fueron eliminadas.")

    # f. Tratamiento de Atípicos (Winsorización) y g. Imputación Numérica (Media)
    for col in num_cols:
        if col != 'id_cliente' and col != 'loan_status':
            # Recortamos los atípicos a los percentiles 1 y 99 para contener los valores extremos
            p01 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p01, upper=p99)
            
            # Como ya tratamos los atípicos, es seguro imputar los vacíos restantes con la media
            if df[col].isnull().any():
                media = df[col].mean()
                df[col] = df[col].fillna(media)
                
    # h. Imputación de Vacíos Categóricos (Moda)
    for col in cat_cols:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            
    logging.info("Tratamiento de inconsistencias y limpieza de datos finalizado.")
    
    # 3. Transformaciones (Solo Categóricas a Numéricas y Sesgos)
    logging.info("Iniciando transformación de variables...")
    
    # a. Eliminar variable para evitar sesgos (según el documento de planificación)
    if 'person_gender' in df.columns:
        df = df.drop('person_gender', axis=1)
        logging.info("Variable 'person_gender' eliminada para mitigar sesgos.")
        
    # b. Variables Binarias
    if 'previous_loan_defaults_on_file' in df.columns:
        # Se utilizan minúsculas ('yes', 'no') porque en el paso 2.5(b) se estandarizó todo el texto
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'yes': 1, 'no': 0})
        
    # c. Variables Dummies (One-Hot Encoding)
    categoricas_a_transformar = ['person_education', 'person_home_ownership', 'loan_intent']
    columnas_existentes = [col for col in categoricas_a_transformar if col in df.columns]
    
    if columnas_existentes:
        df = pd.get_dummies(df, columns=columnas_existentes, drop_first=True)
        # Convertimos booleanos a enteros (1 y 0) ya que pd.get_dummies devuelve booleanos en versiones recientes
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
                
    logging.info("Transformación de variables categóricas a numéricas completada.")
    
    # 4. Quality Check Final
    logging.info("Ejecutando Quality Check Final (Transformado)...")
    qc_final = QualityCheck(df)
    reporte_final = qc_final.quality_report()
    logging.info(f"Reporte Final: {reporte_final}")
    
    # 5. Exportar a CSV
    ruta_exportacion = os.path.join(BASE_DIR, 'data', 'riesgo_crediticio_limpio.csv')
    df.to_csv(ruta_exportacion, index=False)
    logging.info(f"Dataset final exportado exitosamente a: {ruta_exportacion}")

if __name__ == "__main__":
    procesar_datos()
