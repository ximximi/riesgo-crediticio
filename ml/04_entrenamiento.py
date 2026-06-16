import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import joblib # libreria para la serializacion del archivo pkl

def entrenar_modelo():
# PASO 1: se cargan los datos que ya fueron limpiados y divididos de la fase anterior
    print("\n--- INICIANDO ENTRENAMIENTO DEL MODELO ---")
    print("1. Cargando datos de entrenamiento...")

    # X_train contiene las características (ingresos, edad, etc.)
    ruta_x_train = '../data/X_train.csv'
    # y_train contiene la respuesta (si fue fraude/moroso o no)
    ruta_y_train = '../data/y_train.csv'
    
    
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
   