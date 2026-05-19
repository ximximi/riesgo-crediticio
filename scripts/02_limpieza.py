import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar credenciales del .env
load_dotenv('../.env')

# 1. CLASE DE AUDITORÍA 
class QualityCheck:
    def __init__(self, data: pd.DataFrame, exclude_inconsistencies: list = None):
        self.data = data
        #Si se excluyen columnas
        self.exclude_inconsistencies = exclude_inconsistencies if exclude_inconsistencies else []
    #Valores faltantes
    def has_nulls(self) -> bool:
        return self.data.isnull().values.any()
    #Valores duplicados
    def has_duplicates(self) -> bool:
        if 'id_cliente' in self.data.columns:
            return self.data.duplicated(subset=['id_cliente']).any()
        return self.data.duplicated().any()
    #Valores atípicos, se excluyen id_cliente y loan_status y se aplica IQR significa que se consideran outliers aquellos valores que están por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR
    def has_outliers(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])
        for col in numeric_cols.columns:
            if col not in ['id_cliente', 'loan_status']:
                Q1 = numeric_cols[col].quantile(0.25)
                Q3 = numeric_cols[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).any():
                    return True
        return False
    #Inconsistencias numéricas
    def has_negative_values(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])
        numeric_cols = numeric_cols.drop(columns=self.exclude_inconsistencies, errors='ignore')
        for col in numeric_cols.columns:
            if (numeric_cols[col] < 0).any():
                return True
        return False
    #Inconsistencias categóricas, minúsculas y espacios eliminados
    def has_categorical_inconsistencies(self) -> bool:
        cat_cols = self.data.select_dtypes(include=["object"])
        for col in cat_cols.columns:
            values = cat_cols[col].dropna().astype(str)
            normalized = values.str.strip().str.lower()
            if len(values.unique()) != len(normalized.unique()):
                return True
        return False
    #inconsistencias generales, si hay valores negativos o inconsistencias categóricas
    def has_inconsistencies(self) -> bool:
        return self.has_negative_values() or self.has_categorical_inconsistencies()
    #Quality check general, da un diccionario con los resultados de cada def y un score
    def quality_report(self) -> dict:
        return {
            "nulos/faltantes": bool(self.has_nulls()),
            "duplicados": bool(self.has_duplicates()),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
            "quality_score": self.quality_score_weighted()
        }
    #Score ponderado
    def quality_score_weighted(self) -> float:
        weights = {"nulos/faltantes": 0.3, "duplicados": 0.2, "outliers": 0.2, "inconsistencias": 0.3}
        checks = {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies()
        }
        penalty = sum(weights[key] for key in checks if checks[key])
        return round((1 - penalty) * 100, 2)



# 2. FUNCIONES DE LIMPIEZA
def obtener_conexion_db():
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(db_url)

def aplicar_limpieza_base(df):
    logging.info("Aplicando limpieza de datos...")
    df = df.copy()

    # a. Duplicados
    if 'id_cliente' in df.columns:
        df = df.drop_duplicates(subset=['id_cliente'])
        logging.info("Duplicados eliminados basados en 'id_cliente'.")

    # b. Inconsistencias Numéricas (Absolutos)
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col not in ['id_cliente', 'loan_status']:
            df[col] = df[col].abs()

    # c. Inconsistencias Categóricas (Estandarización)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'none': np.nan})

    # d. Imputación Defensiva (Mediana y Moda)
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # e. Tratamiento de Atípicos (Winsorización + Reglas de Negocio)
    for col in num_cols:
        if col not in ['id_cliente', 'loan_status']:
            if col == 'person_age':
                # Regla de Negocio: Edad realista para solicitar un crédito
                df[col] = df[col].clip(lower=18, upper=85)
            else:
                # Winsorización estadística (percentil 1% y 99%) para dinero, historial, etc.
                p01 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=p01, upper=p99)

    return df

def aplicar_feature_eng(df):
    logging.info("Aplicando Feature Engineering...")
    df = df.copy()

    # Eliminar variable para mitigar sesgo
    if 'person_gender' in df.columns:
        df = df.drop(columns=['person_gender'])

    # Corregir correlación alta (person_emp_exp vs person_age)
    if 'person_emp_exp' in df.columns and 'person_age' in df.columns:
        df['es_primer_empleo'] = np.where(df['person_emp_exp'] <= 1, 1, 0)
        divisor = np.maximum(1, df["person_age"] - 18)
        df["porcentaje_vida_laboral"] = df["person_emp_exp"] / divisor
        
    #'yes' y 'no' a 1 y 0
    if 'previous_loan_defaults_on_file' in df.columns:
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'yes': 1, 'no': 0})

    return df

def inyectar_tablas_limpias(df, engine):
    logging.info("Inyectando 2 nuevas tablas limpias a PostgreSQL...")
    
    # 1. Vaciar las tablas respetando las llaves foráneas de tu init.sql
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamo_limpio, cliente_limpio RESTART IDENTITY CASCADE;"))

    # 2. Separar e inyectar en Tabla Cliente Limpio
    cols_cliente = ['id_cliente', 'person_age', 'person_education', 'person_income', 
                    'person_emp_exp', 'person_home_ownership', 'cb_person_cred_hist_length', 
                    'credit_score', 'previous_loan_defaults_on_file', 
                    'es_primer_empleo', 'porcentaje_vida_laboral']
    df_cliente = df[[col for col in cols_cliente if col in df.columns]]
    df_cliente.to_sql('cliente_limpio', engine, if_exists='append', index=False)

    # 3. Separar e inyectar en Tabla Prestamo Limpio
    cols_prestamo = ['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate', 
                    'loan_percent_income', 'loan_status']
    df_prestamo = df[[col for col in cols_prestamo if col in df.columns]]
    df_prestamo.to_sql('prestamo_limpio', engine, if_exists='append', index=False)

# 3. ORQUESTADOR PRINCIPAL
if __name__ == "__main__":
    try:
        motor = obtener_conexion_db()
        
        # 1. Extracción
        query = """
            SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
            FROM cliente c
            JOIN prestamo p ON c.id_cliente = p.id_cliente
        """
        df_crudo = pd.read_sql(query, motor)
        
        # 2. Quality Check Inicial
        qc_inicial = QualityCheck(df_crudo)
        logging.info(f"Reporte Inicial (Crudo): {qc_inicial.quality_report()}")
        
        # 3. Procesamiento
        df_limpio = aplicar_limpieza_base(df_crudo)
        df_final = aplicar_feature_eng(df_limpio)
        
        # 4. Quality Check Final
        qc_final = QualityCheck(df_final)
        logging.info(f"Reporte Final (Limpio): {qc_final.quality_report()}")
        
        # 5. Carga a BD
        inyectar_tablas_limpias(df_final, motor)
        logging.info("--- LIMPIEZA FINALIZADA CON ÉXITO ---")
        
    except Exception as e:
        logging.error(f"Fallo en la limpieza: {e}")