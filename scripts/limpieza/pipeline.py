import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def ejecutar_pipeline():
    logging.info("--- INICIANDO PIPELINE DE DATOS COMPLETO ---")
    
    # 1. Ejecutar descarga
    logging.info("Paso 1: Ejecutando 01_carga_csv.py...")
    resultado_descarga = subprocess.run(["python", "01_carga_csv.py"])
    if resultado_descarga.returncode != 0:
        logging.error("Falló la descarga del CSV. Deteniendo el pipeline.")
        return

    # 2. Ejecutar carga a base de datos cruda
    logging.info("Paso 2: Ejecutando 01_load_data.py...")
    resultado_carga = subprocess.run(["python", "01_load_data.py"])
    if resultado_carga.returncode != 0:
        logging.error("Falló la inyección a la base de datos.")
        return
        
    # 3. Ejecutar limpieza de datos
    logging.info("Paso 3: Ejecutando 02_limpieza.py...")
    resultado_limpieza = subprocess.run(["python", "02_limpieza.py"])
    if resultado_limpieza.returncode != 0:
        logging.error("Falló el proceso de limpieza y creación de tablas limpias.")
        return

    # 4. Ejecutar transformación y preparación para ML
    logging.info("Paso 4: Ejecutando 03_transformacion.py...")
    resultado_transformacion = subprocess.run(["python", "03_transformacion.py"])
    if resultado_transformacion.returncode != 0:
        logging.error("Falló la transformación de variables y división de datos.")
        return

    logging.info("--- PIPELINE EJECUTADO CON ÉXITO ---")
    logging.info("Las 4 tablas están en Postgres y los CSV del modelo están en la carpeta /data.")

if __name__ == "__main__":
    ejecutar_pipeline()