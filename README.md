# Predicción de Riesgo Crediticio — Loan Default

**Integrantes:** Felipe Rubio, Martina López, Ximena Soliz, Yassmin Bazán — Grupo N°8  
**Asignatura:** Gestión de datos para IA 001D  
**Docente:** Jazna Meza Hidalgo  
**Fecha:** Abril-Julio 2026

---

## ¿De qué trata este proyecto?

El sistema predice si un solicitante de crédito caerá en **default** (incumplimiento de pago) o si pagará exitosamente su préstamo. La predicción se basa en variables como edad, ingresos, experiencia laboral, historial crediticio y características del préstamo solicitado.

El objetivo práctico es ayudar a las instituciones financieras a optimizar la matriz de aprobaciones y reducir el riesgo de no pago mediante un modelo de clasificación entrenado sobre datos reales.

---

## Stack tecnológico

| Herramienta | Rol en el proyecto |
|---|---|
| **Python** | Ingesta, limpieza, feature engineering y entrenamiento del modelo |
| **PostgreSQL** | Data Warehouse: almacena los datos crudos y los datos limpios en tablas separadas |
| **Docker** | Contenerización completa para garantizar reproducibilidad en cualquier entorno |
| **Docker Compose** | Orquesta los servicios de la aplicación y la base de datos juntos |
| **Git & GitHub** | Control de versiones y colaboración del equipo |
| **GitHub Actions** | CI/CD: construye y despliega la imagen automáticamente al hacer push a `main` |
| **Render (PaaS)** | Plataforma cloud donde se despliega el contenedor en producción |
| **VS Code** | IDE principal, con asistencia de IA para el diseño de la arquitectura |
| Metabase | Herramienta de Business Intelligence (BI) para la visualización interactiva del Dashboard |

---

## Estructura del proyecto

```text
riesgo-crediticio/
│
├── .env                          # Variables de entorno
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt              # Dependencias Python del proyecto
├── README.md
│
├── data/                         # Datasets — no versionados en git (archivos grandes)
│   ├── 03_Base_Loan_Nueva_Ingesta.csv     # CSV de entrada (~8.2 MB)
│   ├── datos_limpios.csv                  # Salida del pipeline de limpieza
│   ├── X_train.csv                        # Features de entrenamiento (80%)
│   ├── X_test.csv                         # Features de evaluación (20%)
│   ├── y_train.csv                        # Variable objetivo — entrenamiento
│   └── y_test.csv                         # Variable objetivo — evaluación
│
├── db/
│   └── init.sql                  # Schema de PostgreSQL: tablas crudas y limpias
│
├── docs/
│   ├── DisenoTecnico_FelipeRubio_XimenaSoliz_YassminBazan.docx.pdf       # Documento de diseño técnico del proyecto
│
├── models/                       # Modelo serializado — generado al entrenar
│   └── modelo_random_forest.pkl
│
├── results/                      # Gráficas y métricas — generadas al evaluar
│   ├── metricas.json
│   ├── curva_roc.png
│   ├── matriz_confusion.png
│   ├── importancia_variables.png
│   └── distribucion_probabilidades.png
│
└── scripts/                          # Paquetes Python del pipeline de datos
    ├── __init__.py
    │── run_pipeline.py
    │
    ├── common/
    │   ├── __init__.py
    │   └── database.py           # Conexión centralizada a PostgreSQL (reutilizable)
    │
    ├── ingesta/
    │   ├── __init__.py
    │   └── ingesta.py            # Carga el CSV, sanea datos y los inserta en la BD
    │
    ├── limpieza/
    │   ├── __init__.py
    │   ├── limpieza.py           # Limpieza estructural y feature engineering
    │   ├── pipeline.py           # Orquestador del pipeline de limpieza
    │   └── quality.py            # Auditor de calidad de datos (QualityCheck)
    │
    └── training/
        ├── __init__.py
        ├── train.py              # Feature selection, split train/test, preprocesamiento y export, entrenamiento del Random Forest y serialización del modelo
        └── test.py         # Evaluación, métricas y generación de gráficas de diagnóstico
```

---

## Pipeline de datos

El sistema procesa los datos en 5 pasos secuenciales. Cada paso está implementado como un módulo Python independiente bajo `scripts/`.

### Paso 1 — Ingesta (`scripts/ingesta/ingesta.py`)

Carga el CSV de entrada y aplica un **saneamiento mínimo** antes de insertar en PostgreSQL:
- Genera `id_cliente` si el CSV no lo trae (clave para el JOIN entre tablas)
- Imputa nulos en columnas numéricas con la mediana
- Normaliza `previous_loan_defaults_on_file` a los únicos valores aceptados por la BD: `'Yes'` o `'No'`
- Inserta los datos en las tablas crudas `cliente` y `prestamo`

> Este saneamiento previo es clave. Sin él, el `CHECK constraint` de PostgreSQL rechaza el INSERT y las tablas quedan vacías. 

### Paso 2 — Auditoría de calidad (`scripts/limpieza/quality.py`)

La clase `QualityCheck` actúa como un escáner: **solo detecta problemas, no los modifica**. Genera un reporte con:
- Presencia de nulos
- Duplicados por `id_cliente`
- Outliers (método IQR)
- Inconsistencias categóricas (texto con mayúsculas o espacios sucios)
- Score de calidad ponderado del 0 al 100

### Paso 3 — Limpieza y Feature Engineering (`scripts/limpieza/limpieza.py` + `pipeline.py`)

Sobre los datos crudos extraídos de la BD, se aplica:
- Eliminación de duplicados
- Valores negativos → convertidos a absolutos
- Estandarización de texto (minúsculas, sin espacios extra)
- Imputación defensiva con mediana/moda
- Winsorización de outliers (percentiles 1–99), con regla de negocio para edad (18–85)
- Eliminación de `person_gender` para mitigar sesgo algorítmico
- Nuevas variables: `es_primer_empleo` y `porcentaje_vida_laboral`
- Mapeo binario: `previous_loan_defaults_on_file` → `1/0`

Los resultados se cargan en `cliente_limpio` y `prestamo_limpio`, y se exportan a `data/datos_limpios.csv`.

### Paso 4 — Transformación para ML (`scripts/training/train.py`)

Prepara los datos limpios para el modelo:
- Feature selection por correlación con el target (umbral 0.05), preservando variables protegidas por lógica de negocio
- Split estratificado 80/20 (semilla fija `random_state=42`)
- `StandardScaler` para variables numéricas
- `OneHotEncoder` (drop first) para variables categóricas
- Exporta `X_test_crudo.csv` y `y_test.csv`

### Paso 5 — Entrenamiento y Evaluación (`scripts/training/train.py` y `scripts/training/test.py`)

Entrena el clasificador Random Forest sobre los datasets preprocesados y evalúa su desempeño:
- `entrenamiento.py` — lee `X_train`/`y_train`, ajusta el modelo y serializa el resultado en `models/modelo_random_forest.pkl`
- `evaluacion.py` — carga el `.pkl` y `X_test`/`y_test`, calcula métricas numéricas y genera cuatro gráficas de diagnóstico en `results/`

Las carpetas `models/` y `results/` se crean automáticamente en la primera ejecución si no existen.

---

## Cómo ejecutar el proyecto

### Requisitos previos

- Docker y Docker Compose instalados
- Git instalado
- Python 3.10+ (si se corre fuera de Docker)

### 1. Clonar el repositorio

```bash
git clone https://github.com/ximximi/riesgo-crediticio.git
cd riesgo-crediticio
```

### 2. Configurar las variables de entorno

Crear el archivo `.env` en la raíz con las credenciales de la base de datos:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=riesgo_crediticio
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña
```
Aquí tienes la versión actualizada y maquetada directamente en formato Markdown para que solo tengas que copiar y pegarla en el archivo `README.md`.

Se mantiene la estructura que ya tenían, respetando el uso del parámetro `45k` para el entorno local, y se integran de forma nativa los pasos de restauración de Metabase con la advertencia de credenciales.

---

### 3. Levantar los servicios

```bash
docker-compose up -d --build

```

Esto inicializa PostgreSQL con el schema definido en `db/init.sql` y levanta el entorno de la aplicación. **Nota:** Se recomienda esperar aproximadamente 60 segundos o verificar con `docker ps` que el contenedor de PostgreSQL marque el estado `(healthy)` antes de continuar.

### 4. Ejecutar el pipeline completo

**Importante:** Todos los scripts deben ejecutarse DENTRO del contenedor de Docker (`entorno_scripts`) para garantizar que las variables de entorno y las dependencias funcionen correctamente.

Utilizamos el parámetro `45k` para ejecutar una versión optimizada del dataset y evitar desbordamientos de memoria (OOM) en entornos locales (se puede reemplazar por `101k` si el equipo tiene los recursos necesarios).

Desde la raíz del proyecto, ejecuta en orden:

```bash
# Paso 1: Orquestador (Ejecuta Ingesta cruda + Limpieza + Feature Engineering)
docker exec -it entorno_scripts python scripts/run_pipeline.py 45k

# Paso 2: Entrenamiento del modelo Random Forest (Genera el archivo .pkl)
docker exec -it entorno_scripts python scripts/training/train.py 45k

# Paso 3: Evaluación y generación de gráficas (Inyecta métricas a la BD)
docker exec -it entorno_scripts python scripts/training/test.py 45k

```

### 5. Restaurar Dashboards e Integración BI (Metabase)

Para garantizar que los dashboards y reportes configurados previamente estén disponibles en cualquier entorno local sin necesidad de recrearlos manualmente, restauraremos la base de datos interna de Metabase utilizando el archivo de respaldo versionado.

Dependiendo de la terminal que estés utilizando, ejecuta **uno** de los siguientes comandos:

**Opción A (Símbolo del sistema / CMD):**

```cmd
docker exec -i db_riesgo_crediticio psql -U admin -d metabase_db < db\metabase_backup.sql

```

**Opción B (PowerShell):**

```powershell
Get-Content db\metabase_backup.sql | docker exec -i db_riesgo_crediticio psql -U admin -d metabase_db

```

Una vez inyectado el archivo `.sql`, es obligatorio reiniciar el contenedor para que la máquina virtual de Java vuelva a cargar los dashboards en memoria:

```bash
docker restart metabase_dashboard

```

### 6. Acceso al Dashboard

Espera entre 30 y 40 segundos después del reinicio para asegurar que el servicio esté completamente arriba, y luego abre tu navegador web en:

**http://localhost:3030**

> **IMPORTANTE - CREDENCIALES DE ACCESO:**
> Al ejecutar el comando de restauración, la configuración local de usuarios se sobreescribe. Para iniciar sesión, es necesario utilizar **exclusivamente el correo y la contraseña** del integrante del equipo que generó el archivo `metabase_backup.sql` original.

---

## Arquitectura de la base de datos

PostgreSQL almacena los datos en dos capas separadas:

**Tablas crudas** (datos originales, con mínimo saneamiento):
- `cliente` — datos del solicitante
- `prestamo` — características del préstamo

**Tablas limpias** (Data Warehouse, listas para el modelo):
- `cliente_limpio` — sin `person_gender`, con variables derivadas
- `prestamo_limpio` — importes y tasas normalizadas

El schema completo está en `db/init.sql`.

---

## Documentación técnica

El documento de diseño técnico formal del proyecto está disponible en:

```
docs/DisenoTecnico_FelipeRubio_XimenaSoliz_YassminBazan.docx.pdf
```

Incluye el análisis de requerimientos, la arquitectura de datos, el diseño del modelo y el plan de despliegue.

---

## Estado del proyecto

| Fase | Estado |
|---|---|
| Planificación y WBS | ✅ Completado |
| Diseño arquitectónico y selección de stack | ✅ Completado |
| Ingesta de datos (CSV → PostgreSQL) | ✅ Completado |
| Limpieza y feature engineering | ✅ Completado |
| Entrenamiento y evaluación del modelo | ✅ Completado |
| Despliegue en Render (CI/CD via GitHub Actions) | ⏳ Pendiente |
