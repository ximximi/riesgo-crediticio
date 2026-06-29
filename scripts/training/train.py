import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from sklearn.ensemble import RandomForestClassifier

import joblib # libreria para la serializacion del archivo pkl


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
    ruta_base = '../results/'
    
    X_train.to_csv(os.path.join(ruta_base, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(ruta_base, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(ruta_base, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(ruta_base, 'y_test.csv'), index=False)
    
    logging.info("¡Archivos generados exitosamente en la carpeta /results!")

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

def entrenar_modelo():
# PASO 1: se cargan los datos que ya fueron limpiados y divididos de la fase anterior
    print("\n--- INICIANDO ENTRENAMIENTO DEL MODELO ---")
    print("1. Cargando datos de entrenamiento...")

    # X_train contiene las características (ingresos, edad, etc.)
    ruta_x_train = '../results/X_train.csv'
    # y_train contiene la respuesta (si fue fraude/moroso o no)
    ruta_y_train = '../results/y_train.csv'
    
    
    X_train = pd.read_csv(ruta_x_train)

    # y_train usualmente se lee como un DataFrame, pero el modelo lo prefiere como una Serie (una sola columna),
    # por eso usamos .squeeze() para aplanarlo.
    y_train = pd.read_csv(ruta_y_train).squeeze() 

# PASO 2: Configurar el cerebro de la IA (Random Forest)
    # n_estimators=200: Le decimos que cree 200 árboles de decisión distintos.
    # class_weight='balanced': IMPORTANTE!, Como hay muy pocos fraudes/morosos (desbalance), 
    # esto le avisa al modelo que preste más atención a la clase minoritaria para que no la ignore.
    # random_state=42: Asegura que si ejecutas esto mil veces, los árboles se generen siempre igual.
    print("2. Configurando el algoritmo Random Forest (200 árboles, balanceado)...")
    modelo_rf = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42
    )
    # PASO 3: El Entrenamiento real
    # La función .fit() es la importante donde, el modelo analiza X_train e intenta descubrir 
    # las reglas ocultas o patrones que llevan a los resultados en y_train.
    print("3. Entrenando el modelo (esto puede tomar unos segundos)...")
    modelo_rf.fit(X_train, y_train)
    # PASO 4: Serialización (Guardar el modelo)
    # Usamos joblib.dump para tomar el modelo matemático que está en la memoria RAM 
    # y guardarlo físicamente como un archivo .pkl en tu disco duro.
    print("4. Serializando y guardando el modelo entrenado...")
    ruta_modelo = '../models/modelo_random_forest.pkl'
    joblib.dump(modelo_rf, ruta_modelo)
    
    print(f"--- ÉXITO: Modelo guardado en {ruta_modelo} ---")
# Esta línea asegura que la función solo corra si ejecutamos este script directamente
if __name__ == "__main__":
    entrenar_modelo()
