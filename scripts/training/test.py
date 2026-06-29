import os
import json
import logging
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

matplotlib.use('Agg')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CAMBIO 1: Apuntamos al modelo Pipeline y al CSV crudo
RUTA_MODELO  = os.path.join(BASE_DIR, 'models', 'pipeline_random_forest.pkl')
RUTA_X_TEST  = os.path.join(BASE_DIR, 'data', 'X_test_crudo.csv')
RUTA_Y_TEST  = os.path.join(BASE_DIR, 'data', 'y_test.csv')
RUTA_RESULTS = os.path.join(BASE_DIR, 'results')

def evaluar_modelo() -> dict:
    """
    Carga el modelo entrenado, genera predicciones y exporta métricas + gráficas.
    Crea la carpeta /results automáticamente si no existe.
    Retorna el diccionario de métricas.
    """
    logger.info("--- INICIANDO EVALUACIÓN Y GENERACIÓN DE GRÁFICAS ---")
    os.makedirs(RUTA_RESULTS, exist_ok=True)

    # PASO 1: Cargar el modelo serializado y los datos de prueba que el modelo nunca vio
    logger.info("1. Cargando modelo entrenado y datos de prueba...")
    pipeline_rf = joblib.load(RUTA_MODELO)
    X_test = pd.read_csv(RUTA_X_TEST)
    y_test = pd.read_csv(RUTA_Y_TEST).squeeze()

    # PASO 2: Generar predicciones
    # y_pred → decisión binaria (0 = pagará bien, 1 = riesgo de mora)
    # y_prob → probabilidad de riesgo (número entre 0 y 1)
    logger.info("2. Generando predicciones sobre los datos de prueba...")
    y_pred = pipeline_rf.predict(X_test)
    y_prob = pipeline_rf.predict_proba(X_test)[:, 1]

    # PASO 3: Calcular métricas numéricas y exportarlas como JSON
    metricas = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(auc(*roc_curve(y_test, y_prob)[:2]), 4)
    }
    ruta_metricas = os.path.join(RUTA_RESULTS, 'metricas.json')
    with open(ruta_metricas, 'w') as f:
        json.dump(metricas, f, indent=4)


    # === PASO 4: GRÁFICAS DE DIAGNÓSTICO ===

    # A. MATRIZ DE CONFUSIÓN: muestra aciertos y errores del modelo
    # Verdaderos Positivos (riesgo real detectado), Falsos Positivos (alarmas falsas), etc.
    ruta_mc = os.path.join(RUTA_RESULTS, 'matriz_confusion.png')
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin riesgo (0)', 'Con riesgo (1)'],
                yticklabels=['Sin riesgo (0)', 'Con riesgo (1)'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(ruta_mc, dpi=150)
    plt.close()
    logger.info(f"   Gráfica generada → {ruta_mc}")

    # B. CURVA ROC: mide la capacidad del modelo para distinguir riesgo del no-riesgo
    # Mientras más a la esquina superior-izquierda, mejor. AUC=1 es perfecto, AUC=0.5 es azar.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    ruta_roc = os.path.join(RUTA_RESULTS, 'curva_roc.png')
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'Curva ROC (AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
    plt.title('Curva ROC — Capacidad Discriminatoria del Modelo')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(ruta_roc, dpi=150)
    plt.close()
    logger.info(f"   Gráfica generada → {ruta_roc}")

    # C. IMPORTANCIA DE VARIABLES: qué columnas usó más el Random Forest para decidir
    # Extraemos las piezas del pipeline
    modelo = pipeline_rf.named_steps['modelo']
    preprocesador = pipeline_rf.named_steps['preprocesador']
    
    # Reconstruimos los nombres de las columnas
    num_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_test.select_dtypes(include=["object"]).columns.tolist()
    nombres_cat = preprocesador.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    nombres_finales = num_cols + nombres_cat

    importancias = modelo.feature_importances_
    importancias_df = (
        pd.DataFrame({'Variable': nombres_finales, 'Importancia': importancias})
        .sort_values(by='Importancia', ascending=False)
        .head(10)
    )
    
    ruta_imp = os.path.join(RUTA_RESULTS, 'importancia_variables.png')
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importancia', y='Variable', data=importancias_df, palette='viridis')
    plt.title('Top 10 Variables más Importantes para el Modelo')
    plt.tight_layout()
    plt.savefig(ruta_imp, dpi=150)
    plt.close()
    logger.info(f"   Gráfica generada → {ruta_imp}")


    # D. DISTRIBUCIÓN DE PROBABILIDADES: cómo de seguro está el modelo
    # Queremos dos curvas bien separadas: la verde (pagará) a la izquierda, la roja (mora) a la derecha
    ruta_dist = os.path.join(RUTA_RESULTS, 'distribucion_probabilidades.png')
    plt.figure(figsize=(6, 4))
    sns.histplot(y_prob[y_test == 0], color='green', label='Sin riesgo (0)', kde=True, stat="density")
    sns.histplot(y_prob[y_test == 1], color='red',   label='Con riesgo (1)', kde=True, stat="density")
    plt.title('Distribución de Probabilidades Predichas')
    plt.xlabel('Probabilidad de Riesgo de Mora')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_dist, dpi=150)
    plt.close()
    logger.info(f"   Gráfica generada → {ruta_dist}")

    logger.info("--- EVALUACIÓN FINALIZADA CON ÉXITO ---")
    return metricas

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    evaluar_modelo()