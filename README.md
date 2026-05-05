# Proyecto: Predicción de Riesgo Crediticio (Loan Default)

**Integrantes del equipo:** Felipe Rubio, Ximena Soliz, Yassmin Bazán (Grupo N°8)  
**Asignatura:** Gestión de datos para IA 001D  
**Docente:** Jazna Meza Hidalgo  
**Fecha:** Abril 2026

---

## Objetivo general del sistema

Este repositorio contiene la arquitectura de datos, el marco técnico orientado a la nube y el análisis de variables para un modelo de **Machine Learning** enfocado en la evaluación de riesgo crediticio.

El objetivo central del sistema es **predecir si un cliente caerá en default** o si pagará exitosamente el préstamo, basándose en su solvencia, historia crediticia y otras características. Esto permite optimizar la matriz de aprobaciones y reducir el riesgo de no pago.

---

## Componentes y herramientas utilizadas

Para construir nuestro **Pipeline Híbrido** y asegurar su inmutabilidad y despliegue continuo, utilizamos el siguiente stack tecnológico:

- **Visual Studio Code:** IDE principal asistido por IA para agilizar el diseño arquitectónico.
* **Python:** Lenguaje estándar para la ingesta, limpieza y el entrenamiento del modelo de Machine Learning.
- **PostgreSQL:** Base de datos relacional (Data Warehouse) para manejar grandes volúmenes de datos estructurados.
- **Docker:** Contenerización del proyecto para garantizar que se ejecute exactamente igual en desarrollo y en producción.
- **Git & GitHub:** Control de versiones para el trabajo colaborativo e historial del proyecto.
- **GitHub Actions:** Orquestador CI/CD para automatizar la construcción de la imagen Docker y su despliegue.
- **Render (PaaS):** Plataforma en la nube para el despliegue final del contenedor y los servicios web.

---

##  Estructura de carpetas y archivos principales
El repositorio está organizado siguiendo la secuencia lógica del pipeline de datos:

```text
📁 riesgo-crediticio
├── 📁 data/                  
│   ├── 01_Metadata.txt       
│   └── 02_loan_data.csv      
├── 📁 scripts/               
│   ├── 01_ingesta.py         
│   ├── 02_limpieza.py        
│   └── 03_entrenamiento.py   
├── 📁 docs/                  
│   └── DisenoTecnico_FelipeRubio_XimenaSoliz_YassminBazanpdf 
├── .env.example              
├── .gitignore                
├── Dockerfile                
├── docker-compose.yml        
├── requirements.txt          
└── README.md                 

```
---

##  Arquitectura de Datos: Pipeline Híbrido

Para soportar este ecosistema en producción, se ha implementado un **Pipeline Híbrido** que combina la robustez del almacenamiento tradicional con la agilidad del procesamiento en la nube:

1. **Ingesta (ELT inicial):** Los archivos de solicitudes en bruto (`CSV`) se extraen y se ingieren crudos.
2. **Limpieza y Transformación (ETL):** Mediante Python, se purgan variables sesgadas, se manejan valores nulos y se transforman variables cualitativas a cuantitativas.
3. **Carga en Data Warehouse:** Los registros impecables se almacenan en **PostgreSQL** para su fácil consulta.
4. **Entrenamiento (IA):** El modelo de clasificación aprende de los datos curados en ciclos iterativos.
5. **Despliegue Continuo:** Todo el ecosistema se empaqueta y se despliega automáticamente en la nube.


---

## Documentación técnica
El documento de diseño técnico está disponible en:
docs/DisenoTecnico_FelipeRubio_XimenaSoliz_YassminBazanpdf

##  Instrucciones de Ejecución y Despliegue

*(Esta sección se irá completando conforme se avance en la fase de Desarrollo Técnico)*

1. Clonar este repositorio: `git clone https://github.com/ximximi/riesgo-crediticio.git`
2. Construir la imagen de Docker localmente: `docker build -t riesgo-crediticio-app .`
3. Levantar los contenedores (Base de datos y App): `docker-compose up`
4. Los despliegues a producción se realizan automáticamente en **Render** mediante **GitHub Actions** al integrar cambios en la rama `main`.

---
##  Estado del Proyecto (Roadmap V1)
Actualmente, el proyecto se encuentra finalizando la fase de **Diseño Técnico**. 
Próximos pasos en desarrollo:
- [x] Planificación y WBS (Completado)
- [x] Diseño Arquitectónico y Selección de Stack (Completado)
- [ ] Desarrollo de script de ingesta (En progreso)
- [ ] Limpieza y transformación de datos (Pendiente)
- [ ] Entrenamiento del modelo IA (Pendiente)
