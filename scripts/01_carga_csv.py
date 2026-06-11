import os
import gdown
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def descargar_datos():
    # Ruta donde se guardará el archivo
    ruta_destino = 'data/riesgo_crediticio.csv'

    #Nuevo --> Aqui se crea automaticamente la carpeta data si no existe
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    
    file_id = '1zKA5NZ8kvpI65DAsIS5n3yZCsKCemiCm' 
    url = f'https://drive.google.com/uc?id={file_id}'

    if os.path.exists(ruta_destino):
        logging.info(f"El archivo ya existe en {ruta_destino}. Se omite la descarga.")
    else:
        logging.info("Iniciando descarga del archivo CSV desde Google Drive...")
        # Descarga usando gdown
        gdown.download(url, ruta_destino, quiet=False)
        logging.info("¡Descarga completada con éxito!")

if __name__ == "__main__":
    descargar_datos()