import pandas as pd
from sqlalchemy import create_engine
import time
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar las variables del archivo .env
load_dotenv('../.env')

def cargar_base_datos():
    ruta_csv = '../data/riesgo_crediticio.csv'
    
    try:
        # Usamos low_memory=False como indicaste en tus notas
        logging.info("Leyendo el archivo CSV...")
        df = pd.read_csv(ruta_csv, low_memory=False)
        
        # --- DESCOMPOSICIÓN DEL DATASET ---
        # Creamos una columna id_cliente artificial en Pandas (empezando en 1)
        # Esto asegurará que el cliente y su préstamo queden conectados en Postgres
        df['id_cliente'] = df.index + 1 
        
        # Separamos las columnas para la tabla CLIENTE
        df_cliente = df[['id_cliente', 'person_age', 'person_gender', 'person_education', 
                         'person_income', 'person_emp_exp', 'person_home_ownership', 
                         'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']]
        
        # Separamos las columnas para la tabla PRESTAMO (incluyendo el id_cliente como Llave Foránea)
        df_prestamo = df[['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate', 
                          'loan_percent_income', 'loan_status']]

        # --- CONEXIÓN A POSTGRESQL ---
        # Armamos la URL de conexión con las variables de entorno
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        
        # Intentamos conectar dando un pequeño respiro (patrón Wait-for-it)
        time.sleep(2) 
        
        # --- CARGA DE DATOS ---
        # 1. Cargamos tabla CLIENTE primero (porque no tiene dependencias)
        logging.info("Inyectando datos en la tabla 'cliente'...")
        df_cliente.to_sql('cliente', engine, if_exists='append', index=False)
        
        # 2. Cargamos tabla PRESTAMO después (porque depende del id_cliente)
        logging.info("Inyectando datos en la tabla 'prestamo'...")
        df_prestamo.to_sql('prestamo', engine, if_exists='append', index=False)
        
        logging.info("¡Carga masiva completada exitosamente!")

    except Exception as e:
        logging.error(f"Error durante la carga de datos: {e}")

if __name__ == "__main__":
    cargar_base_datos()