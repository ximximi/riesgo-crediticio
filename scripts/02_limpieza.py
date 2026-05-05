import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from correlation_filter import CorrelationFilter
from winsorizer import Winsorizer
from feature_engineering import FeatureEngineering
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def tratar_duplicados(X, drop=True):
    return X.drop_duplicates() if drop else X

# FASE 0: Carga de Datos

# Leemos el archivo crudo que descargó del script 01_carga_csv.py
logging.info("Cargando datos crudos...")
data = pd.read_csv('../data/riesgo_crediticio.csv')


# FASE 1: Selección y Enrutamiento
logging.info("Iniciando Fase 1: Selección y Enrutamiento...")

# Eliminar variables por sesgo 
# Eliminamos el género por razones éticas y legales en riesgo crediticio
if 'person_gender' in data.columns:
    data = data.drop(columns=['person_gender'])
    logging.info(" - Columna 'person_gender' eliminada por sesgo.")

# Separar la Variable Objetivo (Target)
target = "loan_status"

# X = Todas las características (Variables independientes)
X = data.drop(columns=[target], errors="ignore")

# y = Lo que queremos predecir (Variable dependiente)
y = data[target]

logging.info(f" - Variable objetivo separada: {target}")

# Se define qué columnas van a la cinta de matemáticas (numéricas)
num_cols = [
    'person_age', 
    'person_income', 
    'person_emp_exp', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_cred_hist_length', 
    'credit_score', 
    'porcentaje_vida_laboral' # Nueva variable obtenida de feature_engineering

]

# Se define qué columnas van a la cinta de texto (categóricas)
cat_cols = [
    'person_education', 
    'person_home_ownership', 
    'loan_intent', 
    'previous_loan_defaults_on_file'
]

logging.info("Fase 1 completada con éxito. Datos listos para el Pipeline.")





# FASE 2: Preprocesamiento Paralelo
logging.info("\nIniciando Fase 2: Construyendo las cintas transportadoras...")

# Cinta A: Para las variables numéricas
pipeline_numerico = Pipeline([
    ("winsorizer", Winsorizer()),                       # 1. Aplasta atípicos extremos
    ("imputer", SimpleImputer(strategy="mean")),        # 2. Rellena vacíos con el promedio
    ("scaler", StandardScaler())                        # 3. Estandariza la escala de los números
])

# Cinta B: Para las variables categóricas (texto)
pipeline_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # 1. Rellena vacíos con la moda
    ("onehot", OneHotEncoder(handle_unknown="ignore"))    # 2. Convierte texto a números binarios
])

# Unimos ambas cintas en la máquina principal: ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", pipeline_numerico, num_cols), # Pasa las num_cols por la Cinta A
        ("cat", pipeline_categorico, cat_cols) # Pasa las cat_cols por la Cinta B
    ]
)

logging.info(" - Máquina preprocesadora construida con éxito.")


# FASE 4: Ensamblaje del Gran Pipeline
logging.info("\nIniciando Fase 4: Ensamblaje y Transformación...")

# Instanciamos nuestra clase de creación de variables
fe = FeatureEngineering()


# Construimos la cinta transportadora principal (Pipeline Maestro)
pipeline_preparacion = Pipeline(steps=[
    ("duplicados", FunctionTransformer(tratar_duplicados, kw_args={"drop": False})),
    ("feature_engineering", fe),
    ("preprocesador", preprocessor),
    ("colinealidad", CorrelationFilter(threshold=0.9))
])

logging.info(" - Pipeline ensamblado. Encendiendo la fábrica de datos (esto puede tomar unos segundos)...")

# fit: calcula los promedios, los límites y las modas.
pipeline_preparacion.fit(X)

# Obtenemos los nombres de las columnas que salen del preprocesador
feature_names = pipeline_preparacion.named_steps["preprocesador"].get_feature_names_out()

# Le pasamos esos nombres al filtro de correlación
pipeline_preparacion.named_steps["colinealidad"].set_feature_names(feature_names)

# transform: aplica todas las matemáticas y conversiones a los datos
X_transformada = pipeline_preparacion.transform(X)

# Rescatamos los nombres finales (después de que el filtro eliminó redundancias)
cols_finales = pipeline_preparacion.named_steps["colinealidad"].get_feature_names_out()

# Volvemos a armar el DataFrame de Pandas con los datos puros
data_transformada = pd.DataFrame(X_transformada, columns=cols_finales)

# Reconectamos la variable objetivo (loan_status) que habíamos guardado al principio
data_transformada[target] = y.values



# FASE 5: Exportación a PostgreSQL
logging.info("\nIniciando Fase 5: Exportación...")
# Reemplazamos espacios por guiones bajos en los nombres de las columnas
data_transformada.columns = data_transformada.columns.str.replace(' ', '_')
ruta_salida = '../data/riesgo_crediticio_limpio.csv'
data_transformada.to_csv(ruta_salida, index=False)

logging.info(f"¡ÉXITO TOTAL! Dataset limpio y estandarizado guardado en: {ruta_salida}")
logging.info(f"Dimensiones de los datos finales: {data_transformada.shape}")






