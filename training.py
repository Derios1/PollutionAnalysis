import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

def train_models(estimator, X_train, Y_train, horizons_n: int, params: dict):
    models = []
    for i in range(horizons_n):
        X_train = X_train[:-1]
        Y_train = Y_train.shift(-1).dropna()
        
        model = MultiOutputRegressor(estimator(**params))
        model.fit(X_train, Y_train)
        
        models.append(model)
        
        print(f'{(i + 1) / (horizons_n) * 100:.2f}%', end='\r')
    
    return models


def train_and_predict(estimator, X_train, Y_train, horizons_n, params):
    result = np.zeros(shape=(horizons_n, 3))
    X_train = X_train[:-1]
    for i in range(horizons_n):
        Y_train = Y_train.shift(-1).dropna()
        
        model = MultiOutputRegressor(estimator(**params))
        model.fit(X_train, Y_train)
        
        result[i] = model.predict(np.array([X_train[-1]]))
        Y_train = Y_train.append(pd.DataFrame(np.array([result[-1]]), columns=Y_train.columns), ignore_index=True)
        
        print(f'{(i + 1) / (horizons_n) * 100:.2f}%', end='\r')
    
    return result


def predict(models, X_point):
    result = np.zeros(shape=(len(models), 3))
    
    for i, m in enumerate(models):
        result[i] = m.predict(np.array([X_point]))
    
    return result