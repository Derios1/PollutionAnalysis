from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import numpy as np
from sklearn.base import clone

class TimeSeriesWindowCV:
    
    def __init__(self, estimator, train_size, test_size, param_grid=None):
        self.estimator_cls = estimator
        self.train_size = train_size
        self.test_size = test_size
        self.param_grid = param_grid
        
        self.estimator = None
        self.errors = []
        self.best_estimators = [None]*3
        self.best_errors = [np.inf]*3
    
    def fit(self, X, Y):
        self.errors.clear()
        
        window_size = self.train_size + self.test_size
        
        start = 0
        
        if self.param_grid:
            param_grid = list(map(lambda x: dict(zip(self.param_grid.keys(), x)), list(product(*self.param_grid.values()))))
        else:
            param_grid = [None]
        
        for params in param_grid:
            while start + window_size <= len(X):
                X_train = X[start : start + self.train_size]
                Y_train = Y[start : start + self.train_size]

                X_test = X[start + self.train_size : start + window_size]
                Y_test = Y[start + self.train_size : start + window_size]    

                if params:
                    self.estimator = self.estimator_cls(**params)
                else:
                    self.estimator = self.estimator_cls()
                    
                self.estimator.fit(X_train, Y_train)
                Y_pred = self.estimator.predict(X_test)

                self.errors.append(((Y_pred - Y_test)**2).mean(axis=0))

                start += self.test_size
                
            mean = self.mean_error()
            for i in range(3): 
                if mean[i] < self.best_errors[i]:
                    self.best_errors[i] = mean[i]
                    self.best_estimators[i] = clone(self.estimator)
            
            self.errors.clear()
            start = 0
             
    def mean_error(self):
        return np.mean(self.errors, axis=0)

class TimeSeriesWalkingForwardCV:
    def __init__(self, estimator, test_size, n_splits, gap=0, param_grid=None):
        self.estimator_cls = estimator
        self.test_size = test_size
        self.n_splits = n_splits
        self.gap = gap
        self.param_grid = param_grid
        
        self.estimator = None
        self.errors = []
        self.best_estimators = [None]*3
        self.best_errors = [np.inf]*3
    
    def fit(self, X, Y):
        self.errors.clear()
        
        ts_split = TimeSeriesSplit(test_size=self.test_size, n_splits=self.n_splits, gap=self.gap)
        
        if self.param_grid:
            param_grid = list(map(lambda x: dict(zip(self.param_grid.keys(), x)), list(product(*self.param_grid.values()))))
        else:
            param_grid = [None]
            
        for params in param_grid:    
            for train_index, test_index in ts_split.split(X):
                X_train, Y_train = X[train_index], Y[train_index]
                X_test, Y_test = X[test_index], Y[test_index]

                if params:
                    self.estimator = self.estimator_cls(**params)
                else:
                    self.estimator = self.estimator_cls()
    
                self.estimator.fit(X_train, Y_train)
                Y_pred = self.estimator.predict(X_test)

                self.errors.append(((Y_pred - Y_test)**2).mean(axis=0))
               
            mean = self.mean_error()
            for i in range(3): 
                if mean[i] < self.best_errors[i]:
                    self.best_errors[i] = mean[i]
                    self.best_estimators[i] = clone(self.estimator)
            
            self.errors.clear()
    
    def mean_error(self):
        return np.mean(self.errors, axis=0)
    