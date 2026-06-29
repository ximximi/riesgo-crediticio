import pandas as pd
import numpy as np

# 1. CLASE DE AUDITORÍA (QualityCheck)
# Esta clase actúa como un "Escáner Médico" para los datos. Solo detecta, NO modifica.
class QualityCheck:
    def __init__(self, data: pd.DataFrame, exclude_inconsistencies: list = None):
        # Recibe el DataFrame que se va a auditar.
        self.data = data
        # Lista opcional de columnas que no queremos que arrojen error si tienen inconsistencias.
        self.exclude_inconsistencies = exclude_inconsistencies if exclude_inconsistencies else []

    # 1. VALORES FALTANTES (Nulos)
    def has_nulls(self) -> bool:
        # isnull() convierte vacíos en True. values.any() revisa si existe al menos un True en toda la tabla.
        # Devuelve True si falta al menos un solo dato.
        return self.data.isnull().values.any()

    # 2. VALORES DUPLICADOS
    def has_duplicates(self) -> bool:
        # Si la tabla tiene la columna 'id_cliente', revisa si hay IDs repetidos (dos clientes con el mismo código).
        if 'id_cliente' in self.data.columns:
            return self.data.duplicated(subset=['id_cliente']).any()
        # Si no tiene ID, revisa si hay filas enteras que sean clones exactos de otra fila.
        return self.data.duplicated().any()

    # 3. VALORES ATÍPICOS (Outliers usando IQR)
    def has_outliers(self) -> bool:
        # Selecciona solo las columnas que son números (ignora el texto).
        numeric_cols = self.data.select_dtypes(include=["number"])
        for col in numeric_cols.columns:
            # Ignora el ID y el target (loan_status), ya que no tiene sentido buscar outliers ahí.
            if col not in ['id_cliente', 'loan_status']:
                # Q1: Calcula el percentil 25 (el valor donde cae el 25% de los datos más bajos)
                Q1 = numeric_cols[col].quantile(0.25)
                # Q3: Calcula el percentil 75
                Q3 = numeric_cols[col].quantile(0.75)
                # IQR (Rango Intercuartílico): Es la distancia entre Q3 y Q1.
                IQR = Q3 - Q1
                # Límite Inferior: Todo lo que sea menor a esto es un outlier anormalmente bajo.
                lower = Q1 - 1.5 * IQR
                # Límite Superior: Todo lo que sea mayor a esto es un outlier anormalmente alto.
                upper = Q3 + 1.5 * IQR
                # Revisa si hay ALGÚN valor que rompa esos límites. Si lo hay, devuelve True.
                if ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).any():
                    return True
        return False

    # 4. INCONSISTENCIAS NUMÉRICAS (Valores Negativos)
    def has_negative_values(self) -> bool:
        numeric_cols = self.data.select_dtypes(include=["number"])
        numeric_cols = numeric_cols.drop(columns=self.exclude_inconsistencies, errors='ignore')
        for col in numeric_cols.columns:
            # Revisa si hay números menores a 0 (ejemplo: edad -25 o salario -5000).
            if (numeric_cols[col] < 0).any():
                return True
        return False

    # 5. INCONSISTENCIAS CATEGÓRICAS (Texto sucio)
    def has_categorical_inconsistencies(self) -> bool:
        # Filtra solo las columnas de texto (object)
        cat_cols = self.data.select_dtypes(include=["object"])
        for col in cat_cols.columns:
            # Toma los valores, ignora los nulos, y los fuerza a ser texto puro.
            values = cat_cols[col].dropna().astype(str)
            # Simula limpiarlos: strip() borra espacios a los lados, lower() los pasa a minúscula.
            normalized = values.str.strip().str.lower()
            # unique() cuenta cuántas categorías distintas hay.
            # Si al limpiarlos hay MENOS categorías que antes (ej: "Rent" y "rent" se volvieron una sola),
            # significa que la base original estaba sucia. Devuelve True.
            if len(values.unique()) != len(normalized.unique()):
                return True
        return False

    # 6. INCONSISTENCIAS GENERALES
    def has_inconsistencies(self) -> bool:
        # Junta la detección de errores de números negativos y errores de texto sucio.
        return self.has_negative_values() or self.has_categorical_inconsistencies()

    # 7. REPORTE DE CALIDAD (Diccionario Final)
    def quality_report(self) -> dict:
        # Ejecuta todas las funciones anteriores y devuelve un resumen en formato diccionario (JSON).
        return {
            "nulos/faltantes": bool(self.has_nulls()),
            "duplicados": bool(self.has_duplicates()),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies(),
            "quality_score": self.quality_score_weighted()
        }

    # 8. PUNTAJE DE CALIDAD (Score Ponderado)
    def quality_score_weighted(self) -> float:
        # Le asigna un "peso" o gravedad a cada error. Nulos e inconsistencias son los más graves (30% de penalización).
        weights = {"nulos/faltantes": 0.3, "duplicados": 0.2, "outliers": 0.2, "inconsistencias": 0.3}
        checks = {
            "nulos/faltantes": self.has_nulls(),
            "duplicados": self.has_duplicates(),
            "outliers": self.has_outliers(),
            "inconsistencias": self.has_inconsistencies()
        }
        # Suma los pesos solo de los errores que dieron True.
        penalty = sum(weights[key] for key in checks if checks[key])
        # Calcula la nota del 0 al 100. Si penalty es 0.2, el score es (1 - 0.2) * 100 = 80%.
        return round((1 - penalty) * 100, 2)