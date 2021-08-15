import abc
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from itertools import product

class TimeSeriesCV(abc.ABC):
    def __init__(self, estimator, test_size, fig, axs, dates, param_grid):
        self.estimator_cls = estimator
        self.param_grid = param_grid
        self.test_size = test_size
        self.dates = dates.copy()

        self.estimator = None
        self.best_errors_sequence = [[None] * 3 for _ in range(test_size)]
        self.best_params = [[None] * 3 for _ in range(test_size)]

        self.fig, self.axs = fig, axs

        self.axs[0].set_title('target_carbon_monoxide')
        self.axs[1].set_title('target_benzene')
        self.axs[2].set_title('target_nitrogen_oxides')

    @abc.abstractmethod
    def make_forecasts(self, X, Y, metric, params):
        pass

    def make_plot(self, errors):
        n = errors.shape[2]
        colors = plt.cm.plasma(np.linspace(0, 1, self.test_size))
        for h_i in range(self.test_size):
            for ax, error in zip(self.axs, errors[h_i]):
                ax.plot(self.dates, error, color=colors[h_i])
        self.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=1, vmax=self.test_size), cmap='plasma'), ax=self.axs)

    def create_param_grid(self):
        if self.param_grid:
            return list(map(lambda x: dict(zip(self.param_grid.keys(), x)), list(product(*self.param_grid.values()))))
        else:
            return [None]

    def test(self, X_train, Y_train, params):
        if params:
            self.estimator = MultiOutputRegressor(self.estimator_cls(**params))
        else:
            self.estimator = MultiOutputRegressor(self.estimator_cls())

        self.estimator.fit(X_train, Y_train)

        return self.estimator.predict(X_train[-1].reshape(1, -1))

    def fit(self, X, Y, metric):
        param_grid = self.create_param_grid()

        for params in param_grid:
            self.make_forecasts(X, Y, metric, params)

        self.make_plot(np.array(self.best_errors_sequence))

    def update_errors(self, h_i, errors, best_errors, params):
        means = errors[h_i].mean(axis=0)
        for i in range(3):
            if means[i] < best_errors[i]:
                best_errors[i] = means[i]
                self.best_params[h_i][i] = params
                self.best_errors_sequence[h_i][i] = errors[h_i, :, i]


class TimeSeriesWindowCV(TimeSeriesCV):

    def __init__(self, estimator, train_size, test_size, shift, fig, axs, dates, param_grid=None):
        super().__init__(estimator, test_size, fig, axs, dates, param_grid)

        self.train_size = train_size
        self.shift = shift
        self.dates = self.dates[self.train_size:-2*self.test_size:self.shift]

    def make_forecasts(self, X, Y, metric, params):
        window_size = self.train_size + self.test_size
        n = (len(X) - window_size - self.test_size) // self.shift + 1
        errors, i = np.zeros(shape=(self.test_size, n, 3)), 0

        for i in range(self.test_size):
            best_errors_mean = [np.inf] * 3
            start = 0
            Y = np.roll(Y, -1, axis=0)
            j = 0
            while start + window_size <= len(X) - self.test_size:
                X_train, Y_train = X[start: start + self.train_size], Y[start: start + self.train_size]
                Y_test = Y[start + self.train_size]

                Y_pred = self.test(X_train, Y_train, params)

                errors[i][j] = metric(Y_pred, Y_test)

                start += self.shift
                j += 1
            print(f'{(i + 1) / (self.test_size) * 100:.2f}%', end='\r')    
            self.update_errors(i, errors, best_errors_mean, params)


class TimeSeriesWalkingForwardCV(TimeSeriesCV):
    def __init__(self, estimator, test_size, n_splits, fig, axs, dates, gap=0, param_grid=None):
        super().__init__(estimator, test_size, fig, axs, dates, param_grid)

        self.test_size = test_size
        self.n_splits = n_splits
        self.gap = gap
        self.ts_split = TimeSeriesSplit(test_size=self.test_size, n_splits=self.n_splits, gap=self.gap)
        self.dates = self.dates.iloc[[a[-1] for a, b in list(self.ts_split.split(self.dates))]]

    def make_forecasts(self, X, Y, metric, params):
        errors = np.zeros(shape=(self.test_size, self.n_splits, 3))

        for i in range(self.test_size):
            Y = np.roll(Y, -1, axis=0)
            j = 0
            best_errors_mean = [np.inf] * 3
            for train_index, test_index in self.ts_split.split(X):
                X_train, Y_train = X[train_index], Y[train_index]
                Y_test = Y[test_index]

                Y_pred = self.test(X_train, Y_train, params)

                errors[i][j] = metric(Y_pred, Y_test)
                j += 1
            print(f'{(i + 1) / (self.test_size) * 100:.2f}%', end='\r')    
            self.update_errors(i, errors, best_errors_mean, params)
