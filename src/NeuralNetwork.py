from time import time
from typing import List
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, size: List = [30, 16, 8, 2], epochs=70, learningRate=0.001, seed=False) -> None:
        '''
        TODO: Activation function (+ output layer) as a parameter
        '''
        if len(size) < 4:
            print("The neural network must have at least 2 hidden layers")
            exit(1)
        if seed != False:
            np.random.seed(seed)
        self.size = size
        self.epochs = epochs
        self.learningRate = learningRate
        # w: weights for a given layer
        # b: biases for a given layer
        self.params = self.initialize()

    def initialize(self) -> None:
        '''
        Generate the matrix of weights for each layers.
        Weights:
            https://cs231n.github.io/neural-networks-2/#weight-initialization
            Random number with variance sqrt(2 / n)
            w = np.random.randn(n) / sqrt(2 / n)
        Bias is initialized to 0
        '''
        params = {}
        length = len(self.size)
        for i in range(1, length):
            params['w{}'.format(i)] = np.random.randn(
                self.size[i], self.size[i - 1]) * np.sqrt(2. / self.size[i])
            params['b{}'.format(i)] = np.zeros((1, self.size[i - 1]))
        return params

    def reLu(self, x, weights, bias):
        '''
        ReLU activation function
            https://cs231n.github.io/neural-networks-1/#actfun
            f(x) = max(0, x)
        '''
        return np.maximum(0, np.dot(x, weights) + bias)

    def d_reLu(self, x):
        pass

    def softMax(self, x, y):
        return -np.log(x, y)

    def d_softMax(self, x):
        pass

    def forward(self, trainingData: pd.DataFrame):
        pass

    def backward(self, correctResult, trainedResult):
        pass

    def train(self, xTrain, yTrain, xTest, yTest):
        startTime = time()
        for iteration in range(self.epochs):
            pass
        print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
            iteration+1, time() - startTime, accuracy
        ))
