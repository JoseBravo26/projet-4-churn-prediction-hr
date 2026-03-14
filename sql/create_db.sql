-- 1) Crear la base de datos (solo si no existe)
DO $$
BEGIN
   IF NOT EXISTS (
       SELECT FROM pg_database WHERE datname = 'churn_predictor_db'
   ) THEN
      CREATE DATABASE churn_predictor_db
          WITH OWNER = postgres
          ENCODING = 'UTF8'
          TEMPLATE template1;
   END IF;
END$$;

-- 3) Crear el rol de la aplicación si no existe
DO $$
BEGIN
   IF NOT EXISTS (
       SELECT FROM pg_catalog.pg_roles WHERE rolname = 'churn_app'
   ) THEN
      CREATE ROLE churn_app LOGIN PASSWORD 'secure_password_123';
   END IF;
END$$;

-- 4) Crear tablas

DROP TABLE IF EXISTS employees CASCADE;

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL,
    genre VARCHAR(20) NOT NULL,
    etat_civil VARCHAR(20) NOT NULL,
    salaire NUMERIC(10,2) NOT NULL,
    distance NUMERIC(6,2) NOT NULL,
    departement VARCHAR(50) NOT NULL,
    domaine_etude VARCHAR(50) NOT NULL,
    niveau_hierarchique INTEGER NOT NULL,
    poste_freq_deplacement VARCHAR(20) NOT NULL,
    emplois_precedents INTEGER NOT NULL,
    experience_totale NUMERIC(5,2) NOT NULL,
    annees_entreprise NUMERIC(5,2) NOT NULL,
    annees_poste NUMERIC(5,2) NOT NULL,
    annees_derniere_promotion NUMERIC(5,2) NOT NULL,
    annees_responsable_actuel NUMERIC(5,2) NOT NULL,
    heures_semaine NUMERIC(5,2) NOT NULL,
    heures_supplementaires BOOLEAN NOT NULL,
    employes_supervision INTEGER NOT NULL,
    evaluation_precedente INTEGER NOT NULL,
    evaluation_actuelle INTEGER NOT NULL,
    satisfaction_environnement INTEGER NOT NULL,
    satisfaction_travail INTEGER NOT NULL,
    satisfaction_equipe INTEGER NOT NULL,
    satisfaction_balance INTEGER NOT NULL,
    augmentation_salaire NUMERIC(5,2) NOT NULL,
    participation_pee INTEGER NOT NULL,
    formations_completees INTEGER NOT NULL,
    turnover BOOLEAN NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    emp_id INTEGER NOT NULL REFERENCES employees(id),
    prediction VARCHAR(20) NOT NULL,
    probability NUMERIC(5,4) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    action VARCHAR(50) NOT NULL
);

-- 5) Índices
CREATE INDEX IF NOT EXISTS idx_predictions_emp_id
    ON predictions(emp_id);

CREATE INDEX IF NOT EXISTS idx_audit_log_prediction_id
    ON audit_log(prediction_id);

-- 6) Permisos para churn_app
GRANT CONNECT ON DATABASE churn_predictor_db TO churn_app;
GRANT USAGE ON SCHEMA public TO churn_app;

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA public
TO churn_app;

GRANT USAGE, SELECT
ON ALL SEQUENCES IN SCHEMA public
TO churn_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO churn_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO churn_app;
