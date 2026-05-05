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
    ruta_csv = '../data/riesgo_crediticio_limpio.csv'
    
    try:
        logging.info("Leyendo el archivo CSV limpio...")
        # Leemos el CSV que generó el Pipeline
        df = pd.read_csv(ruta_csv, low_memory=False)
        df.columns = df.columns.str.lower()
        
        # --- CONEXIÓN A POSTGRESQL ---
        # Armamos la URL de conexión con las variables de entorno
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        
        # Intentamos conectar dando un pequeño respiro
        time.sleep(2) 
        
        # --- CARGA DE DATOS ---
        logging.info(f"Inyectando {len(df)} registros en la tabla 'riesgo_crediticio_limpio'...")
        
        # Usamos to_sql para enviar toda la matriz directamente a la nueva tabla
        # if_exists='append' añade los datos a la estructura que creamos en init.sql
        df.to_sql('riesgo_crediticio_limpio', engine, if_exists='append', index=False)
        
        logging.info("¡Carga masiva completada exitosamente!.")

    except Exception as e:
        logging.error(f"Error durante la carga de datos: {e}")

if __name__ == "__main__":
    cargar_base_datos()