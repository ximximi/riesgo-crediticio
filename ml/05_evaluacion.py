import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def evaluar_modelo():
    print("--- INICIANDO EVALUACIÓN Y GENERACIÓN DE GRÁFICAS ---")

    # 1. Cargar el cerebro de la IA (Deserialización) y los datos de prueba
    ruta_modelo = '../models/modelo_random_forest.pkl'
    modelo_rf = joblib.load(ruta_modelo)
    
    # Usamos los datos de "test", que la IA jamás ha visto, para ver si realmente aprendió
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv').squeeze()

    # 2. Generar Predicciones
    # y_pred = Qué cree la IA que va a pasar (Paga o no paga / Fraude o no)
    # y_prob = Qué porcentaje de seguridad tiene la IA en su decisión
    print("Generando predicciones sobre los datos de prueba...")
    y_pred = modelo_rf.predict(X_test)
    y_prob = modelo_rf.predict_proba(X_test)[:, 1]

    # 3. Calcular las Métricas Numéricas y guardarlas en JSON
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred), # % de aciertos totales
        "precision": precision_score(y_test, y_pred), # De los que dijo que eran fraude, ¿cuántos lo eran?
        "recall": recall_score(y_test, y_pred), # De todos los fraudes reales, ¿cuántos logró atrapar?
        "f1_score": f1_score(y_test, y_pred) # El balance entre precision y recall
    }
    
    with open('../results/metricas.json', 'w') as f:
        json.dump(metricas, f, indent=4)
    print("-> [OK] Métricas exportadas a results/metricas.json")

    # === 4. GENERACIÓN DE GRÁFICAS PARA EL DASHBOARD (METABASE) ===
    
    # A. Matriz de Confusión Visual
    # Nos muestra visualmente aciertos vs errores (Falsos Positivos, etc.)
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción de la IA')
    plt.savefig('../results/matriz_confusion.png')
    plt.close()
    
    # B. Curva ROC
    # Mide la capacidad del modelo para distinguir entre el bien y el mal. 
    # Mientras más cerca esté la línea a la esquina superior izquierda, mejor.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('../results/curva_roc.png')
    plt.close()

    # C. Importancia de las Variables (Feature Importance)
    # Le preguntamos al Random Forest: "¿En qué te fijaste más para tomar tu decisión?"
    importancias = modelo_rf.feature_importances_
    nombres_columnas = X_test.columns
    # Tomamos las 10 más importantes para no saturar el gráfico
    importancias_df = pd.DataFrame({'Variable': nombres_columnas, 'Importancia': importancias}).sort_values(by='Importancia', ascending=False).head(10)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x='Importancia', y='Variable', data=importancias_df, palette='viridis')
    plt.title('Top 10 Variables más Importantes')
    plt.savefig('../results/importancia_variables.png')
    plt.close()

    # D. Distribución de Probabilidades
    # Vemos qué tan segura estaba la IA al clasificar los casos legítimos vs fraudulentos
    plt.figure(figsize=(6,4))
    sns.histplot(y_prob[y_test == 0], color='green', label='Legítimo (0)', kde=True, stat="density")
    sns.histplot(y_prob[y_test == 1], color='red', label='Problema (1)', kde=True, stat="density")
    plt.title('Distribución de Probabilidades')
    plt.legend()
    plt.savefig('../results/distribucion_probabilidades.png')
    plt.close()

    print("-> [OK] Todas las gráficas (.png) han sido generadas en la carpeta /results/")
    print("--- EVALUACIÓN FINALIZADA CON ÉXITO ---")

if __name__ == "__main__":
    evaluar_modelo()
