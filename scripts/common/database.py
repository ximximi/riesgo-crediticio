import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ==============================================================================
# ENFOQUE ARQUITECTÓNICO: RUTAS ABSOLUTAS RESPECTO AL ARCHIVO
# ==============================================================================
# __file__ es la ubicación de este script (scripts/common/database.py)
# .parent es la carpeta 'common/'
# .parent.parent es la carpeta 'scripts/'
# .parent.parent.parent es la raíz del proyecto ('riesgo-crediticio/')
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ruta_env = BASE_DIR / '.env'

# Forzamos la carga apuntando directamente al archivo físico absoluto
if ruta_env.exists():
    load_dotenv(dotenv_path=ruta_env)
else:
    logging.error(f"¡CRÍTICO! El archivo .env no existe en la ruta calculada: {ruta_env}")

def get_db_engine():
    """Crea y retorna el engine de conexión a PostgreSQL con validación defensiva."""
    
    # Auditoría preventiva de variables para detectar cuál falta antes de que falle SQLAlchemy
    variables_criticas = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    variables_faltantes = [var for var in variables_criticas if os.getenv(var) is None]
    
    if variables_faltantes:
        logging.error(f"Fallo de configuración. Las siguientes variables son None: {variables_faltantes}")
        logging.error(f"Asegúrate de que existan dentro de tu archivo .env y no tengan espacios extra.")
        raise ValueError("Faltan variables de entorno esenciales para establecer la conexión.")
        
    try:
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        logging.error(f"Error al configurar la conexión a la BD: {e}")
        raise