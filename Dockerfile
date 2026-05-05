# Usamos una versión ligera de Python 3
FROM python:3.10-slim

# Definimos el directorio de trabajo
WORKDIR /app

# Copiamos el archivo de dependencias
COPY requirements.txt .

# Instalamos las librerías necesarias
RUN pip install --no-cache-dir -r requirements.txt