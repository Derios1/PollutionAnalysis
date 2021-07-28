from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class TimeSeriesWindowCV:
    
    def __init__(self, estimator, train_size, test_size, param_grid=None):
        self.estimator = estimator()
        self.train_size = train_size
        self.test_size = test_size
        self.param_grid = param_grid
        
        self.errors = []
    
    def fit(self, X, Y):
        self.errors.clear()
        
        window_size = self.train_size + self.test_size
        n = len(X) // window_size
        
        start = 0
        
        while start + window_size <= len(X):
            X_train = X[start : start + self.train_size]
            Y_train = Y[start : start + self.train_size]
            
            X_test = X[start + self.train_size : start + window_size]
            Y_test = Y[start + self.train_size : start + window_size]    
            
            if self.param_grid:
                self.estimator.fit(X_train, Y_train, **self.param_grid)
            else:
                self.estimator.fit(X_train, Y_train)
                
            Y_pred = self.estimator.predict(X_test)
            
            self.errors.append(((Y_pred - Y_test)**2).mean(axis=0))
            
            start += self.test_size
    
    def mean_error(self):
        return np.mean(self.errors, axis=0)

class TimeSeriesWalkingForwardCV:
    def __init__(self, estimator, test_size, n_splits, gap=0, param_grid=None):
        self.estimator = estimator()
        self.test_size = test_size
        self.n_splits = n_splits
        self.gap = gap
        self.param_grid = param_grid
        
        self.errors = []
    
    def fit(self, X, Y):
        ts_split = TimeSeriesSplit(test_size=self.test_size, n_splits=self.n_splits, gap=self.gap)
        for train_index, test_index in ts_split.split(X):
            X_train, Y_train = X[train_index], Y[train_index]
            X_test, Y_test = X[test_index], Y[test_index]
            
            if self.param_grid:
                self.estimator.fit(X_train, Y_train, **self.param_grid)
            else:
                self.estimator.fit(X_train, Y_train)
            
            Y_pred = self.estimator.predict(X_test)
            
            self.errors.append(((Y_pred - Y_test)**2).mean(axis=0))
    
    def mean_error(self):
        return np.mean(self.errors, axis=0)
    