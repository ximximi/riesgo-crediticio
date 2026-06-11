import pandas as pd
from sqlalchemy import create_engine, text
import time
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv('../.env')

def cargar_base_datos():
    ruta_csv = 'data/riesgo_crediticio.csv'
    
    try:
        logging.info("1. Leyendo el archivo CSV crudo (descargado de Drive)...")
        # Leemos el CSV original
        df = pd.read_csv(ruta_csv, low_memory=False)
        df.columns = df.columns.str.lower()
        
        # --- EL TRUCO DE LA LLAVE FORÁNEA ---
        # Si el CSV plano no trae id_cliente, se lo inventamos numerando las filas
        # Esto es vital para que las tablas Cliente y Préstamo se conecten
        if 'id_cliente' not in df.columns:
            df.insert(0, 'id_cliente', range(1, 1 + len(df)))
        
        # --- CONEXIÓN A POSTGRESQL ---
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        time.sleep(2) 
        
        # Limpiamos las tablas crudas por si estamos corriendo el script por segunda vez
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE prestamo, cliente RESTART IDENTITY CASCADE;"))
            
        # --- SEPARACIÓN DE TABLAS ---
        logging.info("2. Separando columnas para la tabla CLIENTE...")
        cols_cliente = ['id_cliente', 'person_age', 'person_gender', 'person_education', 
                        'person_income', 'person_emp_exp', 'person_home_ownership', 
                        'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']
        df_cliente = df[[col for col in cols_cliente if col in df.columns]]
        
        logging.info("3. Separando columnas para la tabla PRESTAMO...")
        cols_prestamo = ['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate', 
                        'loan_percent_income', 'loan_status']
        df_prestamo = df[[col for col in cols_prestamo if col in df.columns]]

        # --- CARGA DE DATOS ---
        logging.info("4. Inyectando datos a PostgreSQL (Tablas Crudas)...")
        df_cliente.to_sql('cliente', engine, if_exists='append', index=False)
        df_prestamo.to_sql('prestamo', engine, if_exists='append', index=False)
        
        logging.info("¡Carga cruda completada exitosamente!")

    except Exception as e:
        logging.error(f"Error durante la carga de datos: {e}")

if __name__ == "__main__":
    cargar_base_datos()