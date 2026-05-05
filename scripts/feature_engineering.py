import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #Por buenas practicas se trabaja sobre una copia
        X = X.copy()
        


        # CREACIÓN DE NUEVAS VARIABLES        
        
        # Porcentaje de vida laboral (Estabilidad del cliente)
        if 'person_emp_exp' in X.columns and 'person_age' in X.columns:
            
            X["porcentaje_vida_laboral"] = np.where( # np.where para evitar el famoso error de "división por cero"
                X["person_age"] > 0,
                X["person_emp_exp"] / X["person_age"],
                0
            )

        # Más variables :]

        return X