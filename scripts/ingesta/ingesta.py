"""
Módulo de ingesta de datos.
Descarga el CSV, lo sanea mínimamente para respetar el schema de BD,
e inyecta los datos en las tablas crudas de PostgreSQL.
"""
import os
import sys
import time
import logging
import pandas as pd
import gdown
from sqlalchemy import text

# --- ARREGLO DE RUTAS PARA ENCONTRAR 'common' ---
DIRECTORIO_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIRECTORIO_SCRIPTS not in sys.path:
    sys.path.append(DIRECTORIO_SCRIPTS)

from common.database import get_db_engine

logger = logging.getLogger(__name__)

# Rutas absolutas estables
BASE_DIR = os.path.dirname(DIRECTORIO_SCRIPTS)

def descargar_datos(version='101k'):
    """Descarga el dataset desde Google Drive."""
    file_id = '1eJXQ-rmIxi3zU2YGdyzl3SPUmSWrSiVa' if version == '45k' else '1jkSd9rdI8P5uL70wzas5iXTexbi0nFCI'
    ruta_csv = os.path.join(BASE_DIR, 'data', f'riesgo_crediticio_{version}.csv')
    
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'

    if os.path.exists(ruta_csv):
        logger.info(f"El archivo ya existe en {ruta_csv}. Se omite la descarga.")
    else:
        logger.info(f"Iniciando descarga del archivo CSV ({version}) desde Google Drive...")
        gdown.download(url, ruta_csv, quiet=False)
        logger.info("¡Descarga completada con éxito!")

def _sanear_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las correcciones pre-INSERT para que PostgreSQL no rechace los datos."""
    logger.info("Saneando datos antes del INSERT (respeto al schema crudo)...")
    df = df.copy()

    # 1. Crear id_cliente si no existe
    if 'id_cliente' not in df.columns:
        df.insert(0, 'id_cliente', range(1, 1 + len(df)))

    # Filtro de realidad para person_age
    if 'person_age' in df.columns:
        # Convertimos a numérico por si acaso, forzando errores a NaN
        df['person_age'] = pd.to_numeric(df['person_age'], errors='coerce')
        
        # Identificamos edades absurdas (menores a 18 o mayores a 100)
        mask_absurda = (df['person_age'] < 18) | (df['person_age'] > 100)
        n_absurdas = mask_absurda.sum()
        
        if n_absurdas > 0:
            logger.warning(f"  -> {n_absurdas} edades absurdas detectadas (ej: {df.loc[mask_absurda, 'person_age'].iloc[0]}). Imputando con mediana.")
            mediana_edad = df.loc[~mask_absurda, 'person_age'].median()
            df.loc[mask_absurda, 'person_age'] = mediana_edad

    # NUEVO: Imputación de columnas críticas del préstamo
    cols_prestamo_nulos = ['loan_int_rate', 'loan_amnt']
    for col in cols_prestamo_nulos:
        if col in df.columns:
            nulos = df[col].isnull().sum()
            if nulos > 0:
                # Imputamos con la mediana para no afectar el promedio con valores extremos
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                logger.warning(f"  -> {nulos} nulos en '{col}' (préstamo) imputados con mediana ({mediana}).")

    #Imputación de intención de préstamo (texto) con 'Unknown' o 'Other'
    if 'loan_intent' in df.columns:
        df['loan_intent'] = df['loan_intent'].fillna('Unknown')

    # Filtro para evitar violaciones de CHECK constraint en loan_percent_income
    if 'loan_percent_income' in df.columns:
        # Si alguien dedica más del 100% de su sueldo (ej: 1.50), lo limitamos a 1.0
        mask_exceso = df['loan_percent_income'] > 1.0
        n_excesos = mask_exceso.sum()
        
        if n_excesos > 0:
            logger.warning(f"  -> {n_excesos} registros con 'loan_percent_income' > 1.0 detectados. Ajustando a 1.0.")
            df.loc[mask_exceso, 'loan_percent_income'] = 1.0
    
    # Protección extrema para loan_percent_income
    if 'loan_percent_income' in df.columns:
        # Convertimos a numérico
        df['loan_percent_income'] = pd.to_numeric(df['loan_percent_income'], errors='coerce')
        
        # Si es menor a 0, lo convertimos en positivo (o imputamos con mediana si es muy extremo)
        mask_negativo = df['loan_percent_income'] < 0
        df.loc[mask_negativo, 'loan_percent_income'] = df.loc[mask_negativo, 'loan_percent_income'].abs()
        
        # Si sigue siendo mayor a 1, lo capamos a 1
        mask_exceso = df['loan_percent_income'] > 1.0
        df.loc[mask_exceso, 'loan_percent_income'] = 1.0
        
        # Imputar cualquier nulo restante con la mediana
        mediana = df['loan_percent_income'].median()
        df['loan_percent_income'] = df['loan_percent_income'].fillna(mediana)
        
        logger.info("  -> [OK] 'loan_percent_income' saneado (rango 0.0 - 1.0).")

    # 2. Imputar columnas numéricas obligatorias con mediana
    cols_not_null_num = ['person_age', 'person_income', 'person_emp_exp',
                            'cb_person_cred_hist_length', 'credit_score']
    for col in cols_not_null_num:
        if col in df.columns:
            nulos = df[col].isnull().sum()
            if nulos > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                logger.warning(f"  -> {nulos} nulos en '{col}' imputados con mediana ({mediana}).")

    # 3. Imputar vivienda con moda
    if 'person_home_ownership' in df.columns:
        nulos = df['person_home_ownership'].isnull().sum()
        if nulos > 0:
            moda_own = df['person_home_ownership'].mode()[0]
            df['person_home_ownership'] = df['person_home_ownership'].fillna(moda_own)

    # 4. Forzar CHECK constraint (Yes/No)
    if 'previous_loan_defaults_on_file' in df.columns:
        df['previous_loan_defaults_on_file'] = (
            df['previous_loan_defaults_on_file']
            .astype(str).str.strip().str.title()
            .replace({'Nan': None, 'None': None, 'Na': None, '': None})
        )
        moda = df['previous_loan_defaults_on_file'].dropna().mode()
        moda_val = moda[0] if not moda.empty else 'No'
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].fillna(moda_val)
        
        n_invalidos = ~df['previous_loan_defaults_on_file'].isin(['Yes', 'No'])
        if n_invalidos.any():
            df.loc[n_invalidos, 'previous_loan_defaults_on_file'] = moda_val

    # 5. Cast a Enteros (INT)
    for col in cols_not_null_num:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    logger.info("  -> [OK] Saneamiento pre-INSERT completado.")
    return df

def cargar_base_datos(version='101k'):
    """Lee el CSV, lo sanea y lo inyecta a PostgreSQL."""
    ruta_csv = os.path.join(BASE_DIR, 'data', f'riesgo_crediticio_{version}.csv')
    
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_csv}")

    logger.info(f"Cargando archivo CSV ({version}) desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv, low_memory=False)
    df.columns = df.columns.str.lower()
    
    # Aplicamos el saneamiento de tu compañero
    df = _sanear_datos(df)
    
    engine = get_db_engine()
    time.sleep(2)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamo, cliente RESTART IDENTITY CASCADE;"))

    cols_cliente = ['id_cliente', 'person_age', 'person_gender', 'person_education',
                    'person_income', 'person_emp_exp', 'person_home_ownership',
                    'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']
    df_cliente = df[[col for col in cols_cliente if col in df.columns]]

    cols_prestamo = ['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate',
                        'loan_percent_income', 'loan_status']
    df_prestamo = df[[col for col in cols_prestamo if col in df.columns]]

    logger.info("Inyectando datos a PostgreSQL (Tablas Crudas)...")
    df_cliente.to_sql('cliente', engine, if_exists='append', index=False, chunksize=5000)
    df_prestamo.to_sql('prestamo', engine, if_exists='append', index=False, chunksize=5000)
    logger.info(f"¡Carga cruda completada! {len(df_cliente):,} clientes y {len(df_prestamo):,} préstamos.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Capturar la versión desde los argumentos de la consola (si existe)
    version_arg = sys.argv[1] if len(sys.argv) > 1 else '101k'
    
    if version_arg not in ['101k', '45k']:
        logger.error("Versión no válida. Usa '101k' o '45k'.")
        sys.exit(1)
        
    descargar_datos(version_arg)
    cargar_base_datos(version_arg)