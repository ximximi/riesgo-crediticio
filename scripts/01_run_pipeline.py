import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def ejecutar_pipeline():
    logging.info("--- INICIANDO PIPELINE DE DATOS ---")
    
    # 1. Ejecutar descarga
    logging.info("Paso 1: Ejecutando carga_csv.py...")
    resultado_descarga = subprocess.run(["python", "carga_csv.py"])
    
    if resultado_descarga.returncode != 0:
        logging.error("Falló la descarga del CSV. Deteniendo el pipeline.")
        return

    # 2. Ejecutar carga a base de datos
    logging.info("Paso 2: Ejecutando load_data.py...")
    resultado_carga = subprocess.run(["python", "load_data.py"])
    
    if resultado_carga.returncode != 0:
        logging.error("Falló la inyección a la base de datos.")
        return
        
    logging.info("--- PIPELINE EJECUTADO CON ÉXITO ---")

if __name__ == "__main__":
    ejecutar_pipeline()