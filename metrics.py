import numpy as np

def MAE(x, y):
    return (np.abs(x - y)).mean(axis=0)


def RelativeAE(x, y):
    return (np.abs(x - y)).sum(axis=0) / np.abs(y).sum(axis=0)