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
    
    # 3. Transformaciones (Solo Categóricas a Numéricas y Sesgos)
    logging.info("Iniciando transformación de variables...")
    
    # a. Eliminar variable para evitar sesgos (según el documento de planificación)
    if 'person_gender' in df.columns:
        df = df.drop('person_gender', axis=1)
        logging.info("Variable 'person_gender' eliminada para mitigar sesgos.")
        
    # b. Variables Binarias
    if 'previous_loan_defaults_on_file' in df.columns:
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
        
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
