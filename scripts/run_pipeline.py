import pandas as pd
from sqlalchemy import text
import logging

from common.database import get_db_engine
from ingesta.ingesta import descargar_datos, cargar_base_datos
from limpieza.quality import QualityCheck
from limpieza.limpieza import aplicar_limpieza_base, aplicar_feature_eng

# Configuración del log para todo el proyecto
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inyectar_tablas_limpias(df, engine):
    #Toma el DataFrame procesado y lo inyecta en las tablas limpias de PostgreSQL.
    logging.info("-" * 50)
    logging.info("FASE 3: CARGA EN DATA WAREHOUSE (POSTGRESQL)")
    logging.info("-" * 50)
    
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE prestamo_limpio, cliente_limpio RESTART IDENTITY CASCADE;"))
    logging.info(" -> [OK] Tablas limpiadas con TRUNCATE CASCADE.")

    cols_cliente = ['id_cliente', 'person_age', 'person_education', 'person_income', 
                    'person_emp_exp', 'person_home_ownership', 'cb_person_cred_hist_length', 
                    'credit_score', 'previous_loan_defaults_on_file', 
                    'es_primer_empleo', 'porcentaje_vida_laboral']
    df_cliente = df[[col for col in cols_cliente if col in df.columns]]
    df_cliente.to_sql('cliente_limpio', engine, if_exists='append', index=False)
    logging.info(f" -> [OK] {len(df_cliente)} registros inyectados en 'cliente_limpio'.")

    cols_prestamo = ['id_cliente', 'loan_amnt', 'loan_intent', 'loan_int_rate', 
                    'loan_percent_income', 'loan_status']
    df_prestamo = df[[col for col in cols_prestamo if col in df.columns]]
    df_prestamo.to_sql('prestamo_limpio', engine, if_exists='append', index=False)
    logging.info(f" -> [OK] {len(df_prestamo)} registros inyectados en 'prestamo_limpio'.")

def ejecutar_pipeline():
    #Orquesta el flujo: Ingesta -> Auditoría -> Limpieza -> Carga
    try:
        logging.info("=" * 60)
        logging.info("INICIANDO EJECUCIÓN DEL PIPELINE DE DATOS")
        logging.info("=" * 60)

        # 1. Ingesta
        logging.info("--- Ejecutando módulo de Ingesta ---")
        descargar_datos()
        cargar_base_datos()

        # 2. Extracción para limpieza
        motor = get_db_engine()
        query = """
            SELECT c.*, p.loan_amnt, p.loan_intent, p.loan_int_rate, p.loan_percent_income, p.loan_status
            FROM cliente c
            JOIN prestamo p ON c.id_cliente = p.id_cliente
        """
        df_crudo = pd.read_sql(query, motor)
        
        # 3. Auditoría Inicial
        qc_inicial = QualityCheck(df_crudo)
        logging.info(f"REPORTE INICIAL (CRUDO): {qc_inicial.quality_report()}")
        
        # 4. Limpieza y Feature Engineering
        df_limpio = aplicar_limpieza_base(df_crudo)
        df_final = aplicar_feature_eng(df_limpio)
        
        # 5. Auditoría Final
        qc_final = QualityCheck(df_final)
        logging.info(f"REPORTE FINAL (LIMPIO): {qc_final.quality_report()}")
        
        # 6. Inyección a BD
        inyectar_tablas_limpias(df_final, motor)

        # 7. Exportación para ML
        ruta_csv_limpio = 'data/datos_limpios.csv'
        df_final.to_csv(ruta_csv_limpio, index=False)
        logging.info(f" -> [OK] Dataset limpio exportado a {ruta_csv_limpio}")
        
        logging.info("=" * 60)
        logging.info("PIPELINE EJECUTADO CON ÉXITO")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Fallo crítico en el pipeline: {e}")

if __name__ == "__main__":
    ejecutar_pipeline()