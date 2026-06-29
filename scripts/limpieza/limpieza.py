import pandas as pd
import numpy as np
import logging

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
