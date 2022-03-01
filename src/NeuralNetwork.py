from time import time
from typing import List
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, size: List = [30, 16, 8, 1], epochs=70, learningRate=0.001, seed=False) -> None:
        if len(size) < 4:
            print("The neural network must have at least 2 hidden layers")
            exit(1)
        if seed != False:
            np.random.seed(seed)
        self.size = size
        self.epochs = epochs
        self.learningRate = learningRate
        self.params = self.initialize()

    def initialize(self) -> None:
        '''
        Generate the matrix of weights for each layers.
        Weights:
            https://cs231n.github.io/neural-networks-2/#weight-initialization
            Random number with variance sqrt(2 / n)
            w = np.random.randn(n) / sqrt(2 / n)
        '''
        params = {}
        length = len(self.size)
        for i in range(1, length):
            params['w{}'.format(i)] = np.random.randn(self.size[i], self.size[i - 1]
                                                      ) * np.sqrt(2. / self.size[i])
        return params

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softMax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward(self, trainingData: pd.DataFrame):
        self.params['a0'] = trainingData

        # Feedforward on each layersto the next one
        length = len(self.size)
        for i in range(1, length - 1):
            self.params['z{}'.format(i)] = np.dot(
                self.params['w{}'.format(i)], self.params['a{}'.format(i - 1)])
            self.params['a{}'.format(i)] = self.sigmoid(
                self.params['z{}'.format(i)])

        # Output layer has a softMax activation function instead of sigmoid
        self.params['z{}'.format(length - 1)] = np.dot(
            self.params['w{}'.format(length - 1)], self.params['a{}'.format(length - 2)])
        self.params['a{}'.format(
            length - 1)] = self.softMax(self.params['z{}'.format(length - 1)])

        return self.params['a{}'.format(length - 1)]

    def backward(self, correctResult, trainedResult):
        change = {}

        # Reverse of the forward pass, start with the output layer
        length = len(self.size)
        print('trainedResult shape', trainedResult.shape)
        print('correctResult shape', np.array(
            [correctResult], dtype=float).shape)
        error = 2 * (trainedResult - np.array(
            [correctResult], dtype=float)) / trainedResult.shape[0] * self.softMax(
            self.params['z{}'.format(length-1)], derivative=True)
        change['w{}'.format(length - 1)] = np.outer(error,
                                                    self.params['a{}'.format(length-1)])
        print('error shape', error.shape)

        for i in range(length - 2, 0, -1):
            error = np.dot(self.params['w{}'.format(
                i + 1)].T, error) * self.sigmoid(self.params['z{}'.format(i)], derivative=True)
            change['w{}'.format(i)] = np.outer(
                error, self.params['a{}'.format(i-1)])
        return change

    def train(self, xTrain, yTrain, xTest, yTest):
        startTime = time()
        for iteration in range(self.epochs):
            for x, y in zip(xTrain, yTrain):
                output = self.forward(x)
                print('output is', output)
                weightChange = self.backward(y, output)
                self.updateNetwork(weightChange)
            accuracy = self.compute_accuracy(xTest, yTest)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                iteration+1, time() - startTime, accuracy
            ))

    def updateNetwork(self, changes):
        for key, value in changes.items():
            self.params[key] -= self.learningRate * value

    def compute_accuracy(self, xTest, yTest):
        predictions = []

        for x, y in zip(xTest, yTest):
            output = self.forward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)
