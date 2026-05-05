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
