import numpy as np


def mse(yPrediction, yTest):
    return np.mean(np.power(yTest - yPrediction, 2))


def d_mse(yPrediction, yTest):
    return 2 * (yPrediction - yTest) / yTest.size
