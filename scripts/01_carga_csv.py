import os
import gdown
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def descargar_datos():
    # Ruta donde se guardará el archivo
    ruta_destino = '../data/riesgo_crediticio.csv'
    
    # Reemplaza ESTE_ID por el ID real de tu archivo en Google Drive
    # El ID es la cadena larga de letras y números en el link de compartir
    file_id = 'ESTE_ID_AQUI' 
    url = f'https://drive.google.com/uc?id={file_id}'

    # LÓGICA IF/ELSE: Validar si el archivo ya existe
    if os.path.exists(ruta_destino):
        logging.info(f"El archivo ya existe en {ruta_destino}. Se omite la descarga.")
    else:
        logging.info("Iniciando descarga del archivo CSV desde Google Drive...")
        # Descarga usando gdown
        gdown.download(url, ruta_destino, quiet=False)
        logging.info("¡Descarga completada con éxito!")

if __name__ == "__main__":
    descargar_datos()