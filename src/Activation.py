import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid(x):
    return x * (1. - x)


def reLu(x):
    '''
    ReLU activation function
        https://cs231n.github.io/neural-networks-1/#actfun
        f(x) = max(0, x)
    '''
    return np.maximum(0, x)


def d_reLu(x):
    '''
    ReLU derivative function
    Set all x <= 0 to 0 and keep other numbers
    '''
    return 1. * (x > 0.)


def softMax(x):
    '''
    Numerically stable softMax
    '''
    x = x - np.max(x, axis=1).reshape(x.shape[0], 1)
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1)


def d_softMax(x):
    '''
    General softMax derivative
    '''
    x = x.reshape(-1, 1)
    return np.diagflat(x) - np.dot(x, x.T)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x)**2
