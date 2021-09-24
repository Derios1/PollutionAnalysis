import training
import numpy as np
from sklearn.ensemble import StackingRegressor


class StackingModel:
    def __init__(self, estimators: list, final_estimator, horizons_n: int):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.horizons_n = horizons_n
        self.models = None

    def fit(self, X_train, Y_train):
        self.models = training.train_models(StackingRegressor, X_train, Y_train, self.horizons_n,
                                            params={'estimators': self.estimators,
                                                    'final_estimator': self.final_estimator,
                                                    'n_jobs': 6}
                                            )

    def predict(self, X_point, output_n=3):
        return training.predict(self.models, X_point, output_n)
