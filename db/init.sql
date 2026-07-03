
-- PREREQUISITO: Base de datos interna de Metabase
-- Se crea aquí para garantizar su existencia desde el primer arranque.
-- Sin esta BD, el contenedor de Metabase falla al intentar conectarse.
SELECT 'CREATE DATABASE metabase_db OWNER ' || current_user
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metabase_db')\gexec

-- FASE 1: TABLAS CRUDAS (Datos originales con restricciones)
CREATE TABLE IF NOT EXISTS cliente (
    id_cliente SERIAL PRIMARY KEY,
    person_age INT NOT NULL CHECK (person_age >= 0),
    person_gender VARCHAR(50),
    person_education VARCHAR(100),
    person_income INT NOT NULL CHECK (person_income >= 0),
    person_emp_exp INT NOT NULL CHECK (person_emp_exp >= 0),
    person_home_ownership VARCHAR(50) NOT NULL,
    cb_person_cred_hist_length INT NOT NULL CHECK (cb_person_cred_hist_length >= 0),
    credit_score INT NOT NULL CHECK (credit_score >= 0),
    previous_loan_defaults_on_file VARCHAR(10) NOT NULL CHECK (previous_loan_defaults_on_file IN ('Yes', 'No'))
);

CREATE TABLE IF NOT EXISTS prestamo (
    id_prestamo SERIAL PRIMARY KEY,
    id_cliente INT NOT NULL,
    loan_amnt INT NOT NULL CHECK (loan_amnt >= 0),
    loan_intent VARCHAR(100),
    loan_int_rate NUMERIC(6,2) NOT NULL CHECK (loan_int_rate >= 0),
    loan_percent_income NUMERIC(3,2) NOT NULL CHECK (loan_percent_income >= 0 AND loan_percent_income <= 1),
    loan_status INT NOT NULL CHECK (loan_status IN (0, 1)),
    
    CONSTRAINT fk_cliente
        FOREIGN KEY (id_cliente) 
        REFERENCES cliente(id_cliente)
        ON DELETE CASCADE
);


-- FASE 2: TABLAS LIMPIAS (Datos procesados y sin sesgos)

CREATE TABLE IF NOT EXISTS cliente_limpio (
    id_cliente INT PRIMARY KEY,  -- No es SERIAL, recibe el ID exacto de la tabla original
    person_age INT,
    -- person_gender ELIMINADO 
    person_education VARCHAR(100),
    person_income INT,
    person_emp_exp INT,
    person_home_ownership VARCHAR(50),
    cb_person_cred_hist_length INT,
    credit_score INT,
    previous_loan_defaults_on_file INT,  -- Limpio: ahora es 0 o 1
    
    -- Nuevas variables (Feature Engineering)
    es_primer_empleo INT,
    porcentaje_vida_laboral NUMERIC(5,2) 
);

CREATE TABLE IF NOT EXISTS prestamo_limpio (
    id_prestamo SERIAL PRIMARY KEY,
    id_cliente INT NOT NULL,
    loan_amnt INT,
    loan_intent VARCHAR(100),
    loan_int_rate NUMERIC(6,2),
    loan_percent_income NUMERIC(3,2),
    loan_status INT,
    
    CONSTRAINT fk_cliente_limpio
        FOREIGN KEY (id_cliente) 
        REFERENCES cliente_limpio(id_cliente)
        ON DELETE CASCADE
);

-- ============================================================
-- FASE 3: MÉTRICAS DEL MODELO (Para dashboards dinámicos)
-- ============================================================
CREATE TABLE IF NOT EXISTS metricas_modelo (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    accuracy NUMERIC(5,2) NOT NULL,
    precision NUMERIC(5,2) NOT NULL,
    recall NUMERIC(5,2) NOT NULL,
    f1_score NUMERIC(5,2) NOT NULL,
    roc_auc NUMERIC(5,2) NOT NULL,
    fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- BASE DE DATOS INTERNA DE METABASE
-- Metabase necesita su propia base de datos para guardar
-- la configuración del dashboard, usuarios, preguntas y métricas.
-- Se crea en el mismo servidor PostgreSQL pero completamente
-- separada de los datos del proyecto (riesgo_db).
-- ============================================================

-- Crear la base de datos de Metabase solo si no existe
-- (Usamos DO $$ porque \gexec solo funciona en psql interactivo, no en init scripts de Docker)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metabase_db') THEN
        PERFORM dblink_exec('dbname=postgres', 'CREATE DATABASE metabase_db');
    END IF;
END
$$;