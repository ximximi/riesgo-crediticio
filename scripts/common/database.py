import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

def get_db_engine():
    #Crea y retorna el engine de conexión a PostgreSQL.
    try:
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        logging.error(f"Error al configurar la conexión a la BD: {e}")
        raise