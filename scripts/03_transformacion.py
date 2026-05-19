import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv('../.env')

def obtener_conexion_db():
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(db_url)

def extraer_datos_limpios(engine):
    logging.info("1. Extrayendo las tablas limpias desde PostgreSQL...")

    query = """
        SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
        FROM cliente_limpio c
        JOIN prestamo_limpio p ON c.id_cliente = p.id_cliente
    """
    return pd.read_sql(query, engine)

def seleccion_de_variables(df):
    logging.info("2. Realizando Selección de Variables (Feature Selection)...")
    df = df.copy()
    
    # a. Eliminar identificadores (evitar que el modelo aprenda IDs de memoria)
    if 'id_cliente' in df.columns:
        df = df.drop(columns=['id_cliente'])

    # b. Filtro por Correlación (Variables Numéricas)
    target = 'loan_status'
    num_cols = df.select_dtypes(include=["number"]).columns
    
    correlaciones = df[num_cols].corr()[target].abs() # abs() porque las negativas también sirven
    
    # AQUÍ SE DEFINE EL UMBRAL DE CORRELACIÓN PARA ELIMINAR VARIABLES IRRELEVANTES
    umbral = 0.05
    vars_baja_corr = correlaciones[correlaciones < umbral].index.tolist()
    
    # --- LISTA BLANCA (REGLAS DE NEGOCIO) ---
    # Protegemos el score crediticio y las variables que creamos en Feature Engineering
    variables_protegidas = ['es_primer_empleo', 'porcentaje_vida_laboral', 'credit_score']
    
    # Separamos las que realmente vamos a borrar de las que vamos a salvar
    vars_a_eliminar = [var for var in vars_baja_corr if var not in variables_protegidas]
    vars_salvadas = [var for var in vars_baja_corr if var in variables_protegidas]
    
    if vars_a_eliminar:
        df = df.drop(columns=vars_a_eliminar)
        logging.info(f"   -> Variables eliminadas por baja correlación lineal (< {umbral}): {vars_a_eliminar}")
        
    if vars_salvadas:
        logging.info(f"   -> Variables PROTEGIDAS por lógica de negocio (sobrevivieron al filtro): {vars_salvadas}")

    return df

def transformar_y_dividir(df):
    logging.info("3. Dividiendo datos (Train/Test) y aplicando Transformaciones...")
    
    target = 'loan_status'
    X = df.drop(columns=[target])
    y = df[target]

    # a. División Estratificada (Train/Test Split)
    # Aquí aplicamos el "stratify=y" de tus notas para proteger a la clase minoritaria
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logging.info(f"   -> Datos divididos: 80% Entrenamiento ({len(X_train)}), 20% Pruebas ({len(X_test)})")

    # b. Identificar qué columnas van a qué transformación
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # c. Construir el motor de transformación (ColumnTransformer)
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    # d. Entrenar el transformador SOLO con los datos de entrenamiento (Evita Data Leakage)
    logging.info("   -> Aplicando StandardScaler a numéricas y OneHotEncoder a categóricas...")
    X_train_transformado = preprocesador.fit_transform(X_train)
    
    # e. Transformar los datos de prueba usando las reglas aprendidas del entrenamiento
    X_test_transformado = preprocesador.transform(X_test)

    # Convertir de vuelta a DataFrames para exportar fácilmente
    nombres_num = num_cols
    # Obtener los nombres de las nuevas columnas binarias generadas por el OneHotEncoder
    nombres_cat = preprocesador.named_transformers_["cat"].get_feature_names_out(cat_cols)
    nombres_finales = list(nombres_num) + list(nombres_cat)

    df_X_train = pd.DataFrame(X_train_transformado, columns=nombres_finales)
    df_X_test = pd.DataFrame(X_test_transformado, columns=nombres_finales)

    return df_X_train, df_X_test, y_train, y_test

def exportar_datos_modelo(X_train, X_test, y_train, y_test):
    logging.info("4. Exportando matrices finales para el modelo...")
    ruta_base = '../data/'
    
    # Guardamos los CSVs que alimentarán al modelo la próxima semana
    X_train.to_csv(os.path.join(ruta_base, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(ruta_base, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(ruta_base, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(ruta_base, 'y_test.csv'), index=False)
    
    logging.info("¡Archivos generados exitosamente en la carpeta /data!")

if __name__ == "__main__":
    try:
        motor = obtener_conexion_db()
        df_limpio = extraer_datos_limpios(motor)
        df_seleccionado = seleccion_de_variables(df_limpio)
        X_train, X_test, y_train, y_test = transformar_y_dividir(df_seleccionado)
        exportar_datos_modelo(X_train, X_test, y_train, y_test)
        logging.info("--- TRANSFORMACIÓN FINALIZADA CON ÉXITO ---")
    except Exception as e:
        logging.error(f"Fallo en la transformación: {e}")