import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar credenciales del .env
load_dotenv('../.env')

# 1. CLASE DE AUDITORÍA (QualityCheck)
# Esta clase actúa como un "Escáner Médico" para los datos. Solo detecta, NO modifica.
class QualityCheck:
    def __init__(self, data: pd.DataFrame, exclude_inconsistencies: list = None):
        # Recibe el DataFrame que se va a auditar.
        self.data = data
        # Lista opcional de columnas que no queremos que arrojen error si tienen inconsistencias.
        self.exclude_inconsistencies = exclude_inconsistencies if exclude_inconsistencies else []

    # 1. VALORES FALTANTES (Nulos)
    def has_nulls(self) -> bool:
        # isnull() convierte vacíos en True. values.any() revisa si existe al menos un True en toda la tabla.
        # Devuelve True si falta al menos un solo dato.
        return self.data.isnull().values.any()

    # 2. VALORES DUPLICADOS
    def has_duplicates(self) -> bool:
        # Si la tabla tiene la columna 'id_cliente', revisa si hay IDs repetidos (dos clientes con el mismo código).
        if 'id_cliente' in self.data.columns:
            return self.data.duplicated(subset=['id_cliente']).any()
        # Si no tiene ID, revisa si hay filas enteras que sean clones exactos de otra fila.
        return self.data.duplicated().any()

    # 3. VALORES ATÍPICOS (Outliers usando IQR)
    def has_outliers(self) -> bool:
        # Selecciona solo las columnas que son números (ignora el texto).
        numeric_cols = self.data.select_dtypes(include=["number"])
        for col in numeric_cols.columns:
            # Ignora el ID y el target (loan_status), ya que no tiene sentido buscar outliers ahí.
            if col not in ['id_cliente', 'loan_status']:
                # Q1: Calcula el percentil 25 (el valor donde cae el 25% de los datos más bajos)
                Q1 = numeric_cols[col].quantile(0.25)
                # Q3: Calcula el percentil 75
                Q3 = numeric_cols[col].quantile(0.75)
                # IQR (Rango Intercuartílico): Es la distancia entre Q3 y Q1.
                IQR = Q3 - Q1
                # Límite Inferior: Todo lo que sea menor a esto es un outlier anormalmente bajo.
                lower = Q1 - 1.5 * IQR
                # Límite Superior: Todo lo que sea mayor a esto es un outlier anormalmente alto.
                upper = Q3 + 1.5 * IQR
                # Revisa si hay ALGÚN valor que rompa esos límites. Si lo hay, devuelve True.
                if ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).any():
                    return True
        return False

    # 4. INCONSISTENCIAS NUMÉRICAS (Valores Negativos)
    def has_negative_values(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])
        numeric_cols = numeric_cols.drop(columns=self.exclude_inconsistencies, errors='ignore')
        for col in numeric_cols.columns:
            # Revisa si hay números menores a 0 (ejemplo: edad -25 o salario -5000).
            if (numeric_cols[col] < 0).any():
                return True
        return False

    # 5. INCONSISTENCIAS CATEGÓRICAS (Texto sucio)
    def has_categorical_inconsistencies(self) -> bool:
        # Filtra solo las columnas de texto (object)
        cat_cols = self.data.select_dtypes(include=["object"])
        for col in cat_cols.columns:
            # Toma los valores, ignora los nulos, y los fuerza a ser texto puro.
            values = cat_cols[col].dropna().astype(str)
            # Simula limpiarlos: strip() borra espacios a los lados, lower() los pasa a minúscula.
            normalized = values.str.strip().str.lower()
            # unique() cuenta cuántas categorías distintas hay.
            # Si al limpiarlos hay MENOS categorías que antes (ej: "Rent" y "rent" se volvieron una sola),
            # significa que la base original estaba sucia. Devuelve True.
            if len(values.unique()) != len(normalized.unique()):
                return True
        return False

    # 6. INCONSISTENCIAS GENERALES
    def has_inconsistencies(self) -> bool:
        # Junta la detección de errores de números negativos y errores de texto sucio.
        return self.has_negative_values() or self.has_categorical_inconsistencies()

    # 7. REPORTE DE CALIDAD (Diccionario Final)
    def quality_report(self) -> dict:
        # Ejecuta todas las funciones anteriores y devuelve un resumen en formato diccionario (JSON).
        return {
            "nulos/faltantes": bool(self.has_nulls()),
            "duplicados": bool(self.has_duplicates()),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
            "quality_score": self.quality_score_weighted()
        }

    # 8. PUNTAJE DE CALIDAD (Score Ponderado)
    def quality_score_weighted(self) -> float:
        # Le asigna un "peso" o gravedad a cada error. Nulos e inconsistencias son los más graves (30% de penalización).
        weights = {"nulos/faltantes": 0.3, "duplicados": 0.2, "outliers": 0.2, "inconsistencias": 0.3}
        checks = {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies()
        }
        # Suma los pesos solo de los errores que dieron True.
        penalty = sum(weights[key] for key in checks if checks[key])
        # Calcula la nota del 0 al 100. Si penalty es 0.2, el score es (1 - 0.2) * 100 = 80%.
        return round((1 - penalty) * 100, 2)

# 2. FUNCIONES DE LIMPIEZA
def obtener_conexion_db():
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(db_url)

def aplicar_limpieza_base(df):
    logging.info("-" * 50)
    logging.info("FASE 1: LIMPIEZA ESTRUCTURAL Y TRATAMIENTO")
    logging.info("-" * 50)
    df = df.copy()

    # a. Duplicados
    if 'id_cliente' in df.columns:
        df = df.drop_duplicates(subset=['id_cliente'])
        logging.info(" -> [OK] Duplicados eliminados basados en 'id_cliente'.")

    # b. Inconsistencias Numéricas (Absolutos)
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col not in ['id_cliente', 'loan_status']:
            df[col] = df[col].abs()
    logging.info(" -> [OK] Valores negativos convertidos a absolutos.")

    # c. Inconsistencias Categóricas (Estandarización)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'none': np.nan})
    logging.info(" -> [OK] Estandarización de texto (minúsculas/espacios).")

    # d. Imputación Defensiva (Mediana y Moda)
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    logging.info(" -> [OK] Imputación defensiva (Mediana/Moda) completada.")

    # e. Tratamiento de Atípicos (Winsorización + Reglas de Negocio)
    for col in num_cols:
        if col not in ['id_cliente', 'loan_status']:
            if col == 'person_age':
                df[col] = df[col].clip(lower=18, upper=85)
            else:
                p01 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=p01, upper=p99)
    logging.info(" -> [OK] Tratamiento de outliers y reglas de negocio aplicadas.")

    return df

def aplicar_feature_eng(df):
    logging.info("-" * 50)
    logging.info("FASE 2: FEATURE ENGINEERING Y ÉTICA")
    logging.info("-" * 50)
    df = df.copy()

    # Eliminar variable para mitigar sesgo
    if 'person_gender' in df.columns:
        df = df.drop(columns=['person_gender'])
        logging.info(" -> [OK] Variable 'person_gender' eliminada (Mitigación de sesgo).")

    # Corregir correlación alta (person_emp_exp vs person_age)
    if 'person_emp_exp' in df.columns and 'person_age' in df.columns:
        df['es_primer_empleo'] = np.where(df['person_emp_exp'] <= 1, 1, 0)
        divisor = np.maximum(1, df["person_age"] - 18)
        df["porcentaje_vida_laboral"] = df["person_emp_exp"] / divisor
        logging.info(" -> [OK] Variables 'es_primer_empleo' y 'porcentaje_vida_laboral' generadas.")
        
    # 'yes' y 'no' a 1 y 0
    if 'previous_loan_defaults_on_file' in df.columns:
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'yes': 1, 'no': 0})
            logging.info(" -> [OK] Mapeo binario en 'previous_loan_defaults_on_file' (1/0).")

    return df

def inyectar_tablas_limpias(df, engine):
    logging.info("-" * 50)
    logging.info("FASE 3: CARGA EN DATA WAREHOUSE (POSTGRESQL)")
    logging.info("-" * 50)
    
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamo_limpio, cliente_limpio RESTART IDENTITY CASCADE;"))
    logging.info(" -> [OK] Tablas limpiadas con TRUNCATE CASCADE.")

    cols_cliente = ['id_cliente', 'person_age', 'person_education', 'person_income', 
                    'person_emp_exp', 'person_home_ownership', 'cb_person_cred_hist_length', 
                    'credit_score', 'previous_loan_defaults_on_file', 
                    'es_primer_empleo', 'porcentaje_vida_laboral']
    df_cliente = df[[col for col in cols_cliente if col in df.columns]]
    df_cliente.to_sql('cliente_limpio', engine, if_exists='append', index=False)
    logging.info(f" -> [OK] {len(df_cliente)} registros inyectados en 'cliente_limpio'.")

    cols_prestamo = ['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate', 
                    'loan_percent_income', 'loan_status']
    df_prestamo = df[[col for col in cols_prestamo if col in df.columns]]
    df_prestamo.to_sql('prestamo_limpio', engine, if_exists='append', index=False)
    logging.info(f" -> [OK] {len(df_prestamo)} registros inyectados en 'prestamo_limpio'.")

# 3. ORQUESTADOR PRINCIPAL
if __name__ == "__main__":
    try:
        motor = obtener_conexion_db()
        
        logging.info("=" * 60)
        logging.info("INICIANDO EJECUCIÓN DEL PIPELINE DE DATOS")
        logging.info("=" * 60)

        query = """
            SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
            FROM cliente c
            JOIN prestamo p ON c.id_cliente = p.id_cliente
        """
        df_crudo = pd.read_sql(query, motor)
        
        qc_inicial = QualityCheck(df_crudo)
        logging.info(f"REPORTE INICIAL (CRUDO): {qc_inicial.quality_report()}")
        
        df_limpio = aplicar_limpieza_base(df_crudo)
        df_final = aplicar_feature_eng(df_limpio)
        
        qc_final = QualityCheck(df_final)
        logging.info(f"REPORTE FINAL (LIMPIO): {qc_final.quality_report()}")
        
        inyectar_tablas_limpias(df_final, motor)
        
    except Exception as e:
        logging.error(f"Fallo en la limpieza: {e}")