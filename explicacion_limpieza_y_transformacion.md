# Documentación de Limpieza y Transformación de Datos

Este documento explica de forma detallada los procesos técnicos ejecutados durante la fase de limpieza y transformación de los datos para el modelo predictivo de Riesgo Crediticio.

---

## 1. Control de Calidad (Quality Check)

Para asegurar que los datos estén en óptimas condiciones antes de entrenar la Inteligencia Artificial, utilizamos un script automatizado que actúa como un "examen médico" para el dataset.

### ¿Cómo funciona la evaluación?
El sistema realiza 4 pruebas fundamentales a los datos:
1.  **Nulos/Faltantes (Penalización: 30%):** Verifica si existen casillas en blanco (`NaN` o nulos) en cualquier parte del archivo.
2.  **Duplicados (Penalización: 20%):** Busca si existen filas completas que estén exactamente repetidas.
3.  **Inconsistencias (Penalización: 30%):** Revisa errores lógicos como números negativos (ej. edades o ingresos de -10) o errores tipográficos y de formato en las variables de texto.
4.  **Outliers / Atípicos (Penalización: 20%):** Utiliza una fórmula matemática llamada Rango Intercuartílico (IQR) para encontrar datos extremos que se alejan de la normalidad (ej. un préstamo desproporcionadamente alto).

### ¿Qué significa el Score de Calidad?
El reporte arranca con un **100% (nota perfecta)** y va restando los porcentajes de penalización explicados arriba si detecta problemas.

**Nuestro Resultado: 80.0 / 100**
En nuestra corrida, los datos superaron con éxito las pruebas de Nulos, Duplicados e Inconsistencias (ninguna restó puntos). Sin embargo, se detectaron **Outliers (valores atípicos)**, lo cual aplicó una penalidad del 20%. Esto es muy común en el ámbito financiero, donde algunas personas ganan excepcionalmente más dinero o piden préstamos enormes. Es un excelente puntaje para un dataset crudo.

---

## 2. Transformación de Variables (Categóricas a Numéricas)

Los modelos matemáticos de Inteligencia Artificial no entienden texto (como "Bachelor", "RENT" o "Yes"); únicamente pueden procesar números. Por ello, tuvimos que transformar el texto.

### El problema de asignar números simples
Al principio, parece lógico cambiar las categorías de texto por una secuencia de números. Por ejemplo, para la educación:
*   High School = 1
*   Bachelor = 2
*   Master = 3

**El Error:** Si hacemos esto, el algoritmo matemático se confundirá e interpretará que un "Master" (3) vale el triple que un "High School" (1), o que la suma de "High School" y "Bachelor" da como resultado un "Master". Esto arruinaría las predicciones.

### La Solución: La técnica "One-Hot Encoding" (Variables Dummy)
Para evitar la jerarquía matemática falsa, aplicamos la técnica de *One-Hot Encoding*.

**¿Cómo funciona?**
En lugar de crear una sola columna con números del 1 al 5, el sistema elimina la columna original y **crea una columna nueva y exclusiva para cada una de las categorías existentes**. 

Luego, rellena estas nuevas columnas solo con **1 (Verdadero)** o **0 (Falso)**.

**Ejemplo aplicado a nuestros datos:**
La columna original `person_education` tenía múltiples categorías. El script la eliminó y la transformó en:
*   `person_education_High School`: Tiene un `1` si el cliente tiene esa educación, `0` si no.
*   `person_education_Bachelor`: Tiene un `1` si el cliente tiene esa educación, `0` si no.
*   `person_education_Master`: `1` o `0`.
*   `person_education_Doctorate`: `1` o `0`.

Lo mismo se hizo para `person_home_ownership` (RENT, OWN, etc.) y para `loan_intent` (MEDICAL, EDUCATION, PERSONAL, etc.). 

### Eliminación de Sesgos
Finalmente, en línea con el documento de planificación, la variable `person_gender` no fue transformada a números, sino que fue **eliminada por completo** del dataset. Esta decisión técnica se tomó para evitar sesgos discriminatorios en las predicciones del modelo.

---

## 3. Generación del Dataset Limpio (CSV)

Una vez aplicadas las transformaciones y evaluado el Quality Check, el proceso finaliza con la creación de una nueva "instancia" de los datos, dejándolos listos para la fase de modelado con Inteligencia Artificial.

### ¿Cómo se exporta el archivo?
Utilizando la librería **Pandas** en Python, tomamos la tabla virtual (DataFrame) que ya contiene todas nuestras variables transformadas (con las nuevas columnas de 1s y 0s) y la exportamos a un archivo físico. 

Esto se realiza mediante la instrucción técnica `df.to_csv()`. Durante esta exportación, tomamos una precaución metodológica importante: utilizar el parámetro `index=False`. Esto le indica al sistema que **no** debe guardar los números correlativos de las filas (0, 1, 2, 3...) como si fueran una columna adicional, manteniendo el archivo enfocado estrictamente en las variables del préstamo.

### Destino del archivo
El resultado es el archivo **`riesgo_crediticio_limpio.csv`**, el cual queda guardado automáticamente en tu carpeta compartida `data/`. Este archivo es la versión final e impecable de tu universo de solicitantes de crédito, y servirá como el alimento principal para entrenar y evaluar el modelo predictivo.

---

## 4. Tratamiento Propuesto para Inconsistencias (Próximos Pasos)

Dado que actualmente estamos en la etapa de ingesta de datos crudos, el sistema actúa como un **auditor** (detecta problemas y calcula un puntaje de calidad). Para automatizar la limpieza profunda, se deben implementar las siguientes reglas de tratamiento directamente en el script `02_limpieza.py`, basándonos en metodologías estandarizadas:

### 1. Tratamiento de Datos Ausentes (Valores Faltantes o Nulos)
El manejo de valores ausentes es un aspecto crucial en la preparación de datos. Existen diversas estrategias que aplicaremos dependiendo del contexto:

*   **Variables Cuantitativas (Numéricas):**
    *   **Uso de la Mediana:** Si la variable contiene valores atípicos (outliers) o datos asimétricos, imputaremos los datos faltantes con la mediana, ya que es menos sensible a los extremos.
    *   **Uso de la Media:** Si la variable tiene una distribución normal y no presenta atípicos, sustituiremos con la media. *(Nota: Si los valores atípicos de esta columna son tratados previamente para acercar los extremos a la normalidad, sí es válido imputar luego con la media).*
*   **Variables Cualitativas (Categóricas o Texto):**
    *   **Uso de la Moda:** Se reemplazarán los valores ausentes con la categoría más frecuente (la moda).
*   **Estrategias alternativas según el escenario:**
    *   **Eliminación de Filas:** Si los datos faltantes se concentran en un número muy pequeño de filas dentro de un dataset grande, se pueden eliminar directamente esas filas para evitar los sesgos de la imputación.
    *   **Eliminación de Variables:** Si una columna posee más del 50% de sus datos ausentes y aporta poco valor analítico, la acción correcta es eliminar la variable completa.
    *   *(Nota: Se desaconseja totalmente rellenar los datos numéricos ausentes con ceros, ya que introduce un sesgo significativo).*

### 2. Registros Duplicados
*   **La acción:** Su eliminación está estrictamente condicionada.
*   **Cómo funcionaría:** Solo se pueden eliminar los registros duplicados si existe una columna que funcione como "Identificador Único" (como un ID de transacción o ID de cliente). Si no contamos con una columna que identifique unívocamente la información, no podemos eliminarlos, ya que podríamos estar borrando datos de transacciones legítimamente iguales que pertenecen a eventos distintos.

### 3. Valores Atípicos (Outliers)
*   **La acción:** Tratamiento de acercamiento o "recorte" (Winsorización).
*   **Cómo funcionaría:** En lugar de eliminarlos indiscriminadamente, el objetivo es alterar la columna haciendo que el cambio sea lo menos dramático posible. Se "acortan" (limitan) los valores extremos para acercarlos a la distribución general. Como mencionamos antes, una vez tratados, se habilita la posibilidad de usar la media para la imputación de vacíos.

### 4. Valores Numéricos Negativos e Inconsistencias Categóricas
*   **Valores negativos:** Edades o ingresos negativos no tienen sentido y deben ser corregidos (ej. usando valor absoluto) o transformados a valores nulos para ser imputados mediante las reglas de la Sección 1.
*   **Inconsistencias de texto:** Estandarización automática convirtiendo todas las categorías a letras minúsculas y eliminando espacios adicionales (ej. transformar `"Medico "` y `"MEDICO"` en `"medico"`).

### ¿Qué tendríamos que modificar en el código?

Para transformar nuestro script actual en un motor de limpieza y tratamiento que cumpla con este diseño riguroso (asegurando la trazabilidad del proceso), añadiríamos bloques de código utilizando Pandas:

*   **Para los vacíos (Numéricos):** Evaluaciones condicionales que usen `df['col'].fillna(df['col'].median())` o `df['col'].fillna(df['col'].mean())` según la presencia de atípicos.
*   **Para los vacíos (Categóricos):** `df['col'].fillna(df['col'].mode()[0])`.
*   **Para los vacíos críticos:** `df.dropna(subset=['col_clave'])` para eliminar filas, o `df.drop(columns=['col_inutil'])` si los vacíos superan el 50%.
*   **Para los duplicados:** `df.drop_duplicates(subset=['id_cliente'])` (solo si poseemos dicha columna identificadora).

Implementar estas decisiones de forma automatizada y bien documentada garantizará que el archivo final `riesgo_crediticio_limpio.csv` contenga datos de alta calidad para las etapas posteriores de análisis e inteligencia artificial.

---

## 5. Diccionario de Tratamiento por Variable

Para ser rigurosos con el Análisis Exploratorio de Datos (AED) y mantener la trazabilidad de los procesos, a continuación se documenta de forma explícita cómo se tratará y transformará cada columna específica de nuestra base de datos:

### Identificadores y Claves
*   **`id_cliente`**: Funciona como identificador único (llave primaria).
    *   **Tratamiento:** Se utilizará exclusivamente para detectar y **eliminar registros duplicados**. Si una fila carece de este valor crítico, será eliminada (no se imputa). No sufre transformaciones matemáticas.

### Variables Cuantitativas (Numéricas)
*(Incluye: `person_age`, `person_income`, `person_emp_length`, `cb_person_cred_hist_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`)*.
*   **Tratamiento de Inconsistencias:** 
    1. Se aplica la función de "valor absoluto" para corregir inconsistencias de números negativos (ej. edad -25 pasa a 25).
    2. Se someten a **Winsorización** (recorte a los percentiles 1 y 99) para mitigar el impacto de los valores atípicos (outliers) sin perder la información de la fila.
    3. Una vez acotados los atípicos, cualquier celda vacía (nula) será **imputada utilizando la media** de la columna.
*   **Transformación Posterior:** Ninguna. Conservan su naturaleza continua.

### Variables Cualitativas (Categóricas o de Texto)
*(Incluye: `person_education`, `person_home_ownership`, `loan_intent`)*.
*   **Tratamiento de Inconsistencias:** 
    1. Se estandariza todo el texto a minúsculas y se eliminan espacios residuales (`" RENT "` -> `"rent"`) para evitar categorías falsamente separadas.
    2. Los valores nulos se **imputarán con la moda** (la categoría más frecuente).
*   **Transformación Posterior:** Una vez limpias, se aplica la técnica **One-Hot Encoding**. Cada una de estas columnas desaparecerá para dar paso a múltiples columnas nuevas (ej. `person_education_Master`) rellenas exclusivamente con `1` o `0`.

### Variables Categóricas Binarias
*(Incluye: `previous_loan_defaults_on_file`)*.
*   **Tratamiento de Inconsistencias:** Mismo tratamiento categórico inicial (imputación por la moda si falta).
*   **Transformación Posterior:** En lugar de separarse en nuevas columnas, se mapea directamente a valores binarios matemáticos: `"Yes"` se transforma en `1` y `"No"` se transforma en `0`.

### Variables Descartadas (Mitigación de Sesgos)
*   **`person_gender`** (Género): 
    *   **Acción y Justificación:** Es **eliminada por completo** del dataset. Esta decisión técnica se tomó desde la etapa de planificación para evitar que la red neuronal o el modelo de machine learning aprenda patrones discriminatorios basados en el género al evaluar el riesgo crediticio.

### Variable Objetivo (Target)
*   **`loan_status`** (Estado histórico del préstamo: `0` o `1`).
    *   **Acción:** Es la etiqueta que nuestro modelo intentará predecir. 
    *   **Tratamiento:** Si encontramos filas donde este valor sea nulo (es decir, no sabemos si el cliente pagó o no), **la fila será eliminada**. Nunca se debe imputar la variable objetivo, ya que esto corrompería el entrenamiento de la Inteligencia Artificial al enseñarle sobre ejemplos inventados.
