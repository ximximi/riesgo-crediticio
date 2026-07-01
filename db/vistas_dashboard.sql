-- ============================================================
-- VISTAS PARA DASHBOARD EN METABASE
-- Proyecto: Predicción de Riesgo Crediticio
-- Asignatura: Gestión de Datos para IA
-- 
-- Estas vistas pre-calculan las métricas más importantes
-- del proyecto para ser usadas directamente en Metabase
-- sin necesidad de escribir SQL en el dashboard.
--
-- CÓMO APLICAR: Ejecutar este script en la base de datos riesgo_db
-- Desde Docker: 
--   docker exec -i db_riesgo_crediticio psql -U admin -d riesgo_db < db/vistas_dashboard.sql
-- ============================================================


-- ============================================================
-- VISTA 1: Tasa de mora por tipo de préstamo (loan_intent)
-- Muestra qué finalidad de préstamo tiene más riesgo de mora.
-- Útil para identificar categorías de alto riesgo.
-- ============================================================
CREATE OR REPLACE VIEW vista_mora_por_intencion AS
SELECT
    -- Tipo de préstamo (educación, médico, personal, etc.)
    loan_intent                                         AS intencion_prestamo,

    -- Total de préstamos para ese tipo
    COUNT(*)                                            AS total_prestamos,

    -- Cantidad de préstamos en mora (loan_status = 1)
    SUM(loan_status)                                    AS total_en_mora,

    -- Tasa de mora como porcentaje, redondeado a 2 decimales
    ROUND(
        AVG(loan_status) * 100, 2
    )                                                   AS tasa_mora_porcentaje

FROM prestamo_limpio
GROUP BY loan_intent
ORDER BY tasa_mora_porcentaje DESC;

-- Comentario: Ordena de mayor a menor tasa de mora para identificar
-- rápidamente cuáles tipos de préstamo son más riesgosos


-- ============================================================
-- VISTA 2: Distribución de clientes por rango de credit_score
-- Agrupa los clientes en rangos de 50 puntos para ver
-- cómo se distribuye la salud crediticia de la cartera.
-- ============================================================
CREATE OR REPLACE VIEW vista_distribucion_credito AS
SELECT
    -- Rango inferior del segmento de puntaje (múltiplos de 50)
    FLOOR(credit_score / 50) * 50                       AS rango_desde,

    -- Rango superior del segmento (rango_desde + 49)
    FLOOR(credit_score / 50) * 50 + 49                  AS rango_hasta,

    -- Etiqueta legible del rango (ej: "600 - 649")
    CONCAT(
        FLOOR(credit_score / 50) * 50,
        ' - ',
        FLOOR(credit_score / 50) * 50 + 49
    )                                                   AS rango_credito,

    -- Cantidad de clientes en ese rango
    COUNT(*)                                            AS cantidad_clientes,

    -- Porcentaje del total de clientes
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2
    )                                                   AS porcentaje_del_total

FROM cliente_limpio
GROUP BY FLOOR(credit_score / 50)
ORDER BY rango_desde ASC;

-- Comentario: La función OVER() calcula el total general para
-- obtener el porcentaje relativo de cada segmento


-- ============================================================
-- VISTA 3: Perfil comparativo entre clientes morosos y no morosos
-- Compara promedios clave entre quienes cayeron en mora (1)
-- y quienes cumplieron sus pagos (0).
-- Útil para entender qué características diferencian a cada grupo.
-- ============================================================
CREATE OR REPLACE VIEW vista_perfil_cliente_moroso AS
SELECT
    -- Etiqueta legible del estado del préstamo
    CASE
        WHEN pl.loan_status = 1 THEN 'Con Mora'
        WHEN pl.loan_status = 0 THEN 'Sin Mora'
    END                                                 AS estado_prestamo,

    -- Cantidad de registros en cada grupo
    COUNT(*)                                            AS total_clientes,

    -- Ingreso promedio del cliente (en USD)
    ROUND(AVG(cl.person_income), 2)                     AS ingreso_promedio,

    -- Edad promedio del cliente
    ROUND(AVG(cl.person_age), 1)                        AS edad_promedio,

    -- Monto promedio del préstamo solicitado
    ROUND(AVG(pl.loan_amnt), 2)                         AS monto_prestamo_promedio,

    -- Tasa de interés promedio del préstamo (%)
    ROUND(AVG(pl.loan_int_rate), 2)                     AS tasa_interes_promedio,

    -- Puntaje crediticio promedio
    ROUND(AVG(cl.credit_score), 1)                      AS credit_score_promedio,

    -- Porcentaje del ingreso destinado al préstamo (promedio)
    ROUND(AVG(pl.loan_percent_income) * 100, 2)         AS porcentaje_ingreso_promedio

FROM prestamo_limpio pl
-- JOIN para cruzar datos del préstamo con el perfil del cliente
JOIN cliente_limpio cl ON pl.id_cliente = cl.id_cliente

GROUP BY pl.loan_status
ORDER BY pl.loan_status ASC;

-- Comentario: Este perfil comparativo permite al modelo de ML
-- identificar los patrones que distinguen clientes de riesgo


-- ============================================================
-- VISTA 4: KPIs globales del proyecto (fila única de resumen)
-- Concentra los indicadores más importantes en una sola fila
-- para mostrar en tarjetas de métricas en el dashboard.
-- ============================================================
CREATE OR REPLACE VIEW vista_kpis_generales AS
SELECT
    -- Total de clientes únicos en el sistema (datos limpios)
    (SELECT COUNT(*) FROM cliente_limpio)               AS total_clientes,

    -- Total de préstamos registrados (datos limpios)
    (SELECT COUNT(*) FROM prestamo_limpio)              AS total_prestamos,

    -- Tasa global de mora: % de préstamos con loan_status = 1
    ROUND(
        (SELECT AVG(loan_status) * 100 FROM prestamo_limpio), 2
    )                                                   AS tasa_mora_global_porcentaje,

    -- Puntaje crediticio promedio de todos los clientes
    ROUND(
        (SELECT AVG(credit_score) FROM cliente_limpio), 1
    )                                                   AS credit_score_promedio,

    -- Monto promedio de préstamo en toda la cartera (en USD)
    ROUND(
        (SELECT AVG(loan_amnt) FROM prestamo_limpio), 2
    )                                                   AS monto_prestamo_promedio,

    -- Ingreso promedio de los clientes (en USD)
    ROUND(
        (SELECT AVG(person_income) FROM cliente_limpio), 2
    )                                                   AS ingreso_cliente_promedio,

    -- --------------------------------------------------------
    -- MÉTRICAS DEL MODELO DE ML (Random Forest Classifier)
    -- Ahora estas métricas son dinámicas: extraen el último
    -- resultado insertado en la tabla 'metricas_modelo' por Python.
    -- --------------------------------------------------------
    COALESCE(
        (SELECT accuracy FROM metricas_modelo ORDER BY fecha_entrenamiento DESC LIMIT 1), 
        0
    )                                                   AS modelo_accuracy_porcentaje,
    
    COALESCE(
        (SELECT recall FROM metricas_modelo ORDER BY fecha_entrenamiento DESC LIMIT 1), 
        0
    )                                                   AS modelo_recall_porcentaje,
    
    COALESCE(
        (SELECT roc_auc FROM metricas_modelo ORDER BY fecha_entrenamiento DESC LIMIT 1), 
        0
    )                                                   AS modelo_roc_auc_porcentaje;

-- Comentario: Al ser una fila única, esta vista es ideal para
-- las tarjetas de métricas (KPI cards) en el dashboard de Metabase
