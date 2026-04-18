import numpy as np


def mean_squared_error(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)


def mean_absolute_error(y_real, y_pred):
    return np.mean(y_real - y_pred)
