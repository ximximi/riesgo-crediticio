
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