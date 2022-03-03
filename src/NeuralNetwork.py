from random import random
from time import time
from typing import List
import numpy as np


class NeuralNetwork:
    def __init__(self, size: List, epochs=50, learningRate=0.001, seed=False) -> None:
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
        self.initialize()
        print("Created neural network of size", size)

    def initialize(self) -> None:
        '''
        Generate the matrix of weights for each layers.
        Weights:
            https://cs231n.github.io/neural-networks-2/#weight-initialization
            Random number with variance sqrt(2 / n)
            w = np.random.randn(n) / sqrt(2 / n)
        Bias is initialized to 0
        '''
        self.weights = []
        self.biases = []
        for i in range(1, len(self.size)):
            # weights for a given layer
            # [N, K] where K is the number of neurons in the layer and N the number of neurons in the previous layer
            self.weights.append(np.random.randn(self.size[i - 1],
                                                self.size[i]) * np.sqrt(2. / self.size[i]))
            # biases for a given layer
            # [K, 1] where K is the number of neurons in the layer
            self.biases.append(np.zeros((1, self.size[i])))
            # print("weights and bias", i, "shape",
            #       self.weights[i - 1].shape, self.biases[i - 1].shape)

    def reLu(self, x):
        '''
        ReLU activation function
            https://cs231n.github.io/neural-networks-1/#actfun
            f(x) = max(0, x)
        '''
        return np.maximum(0, x)

    def d_reLu(self, x):
        pass

    def mse(self, yTrained, yTest):
        return np.mean(np.power(yTest - yTrained, 2))

    def d_mse(self, yTrained, yTest):
        return 2 * (yTrained - yTest) / yTest.size

    def softMax(self, x, y):
        return -np.log(x, y)

    def d_softMax(self, x):
        pass

    def forward(self, trainingData: np.ndarray):
        '''
        trainingData [1, D] where D is the number of features
        returns the list values for each neurons in each layers
        '''
        inputs = []
        result = trainingData
        for i in range(len(self.size) - 1):
            result = self.reLu(
                np.dot(result, self.weights[i]) + self.biases[i])
            inputs.append(result)
        return result, inputs

    def backward(self, inputs, error):
        '''
        Backpropagation, in reverse order
        '''
        for i in range(len(inputs) - 1, -1, -1):
            inputError = np.dot(error, self.weights[i].T)
            weightsError = np.dot(inputs[i - 1].T, error)
            self.weights[i] -= self.learningRate * weightsError
            self.biases[i] -= self.learningRate * error
            error = inputError
        return error

    def train(self, xTrain, yTrain, xTest, yTest):
        # print(xTrain)
        allTime = time()
        for epoch in range(self.epochs):
            startTime = time()
            err = 0
            for trainingData, correctData in zip(xTrain, yTrain):
                # print("training data shape", trainingData.shape)
                result, inputs = self.forward(trainingData)
                err += self.mse(result, correctData)
                error = self.d_mse(result, correctData)
                # Update weights and biasses depending on the error value
                self.backward(inputs, error)
            err = err / len(xTrain)
            print('Epoch: {}, Time Spent: {:.2f}s, error: {:.2f}'.format(
                epoch + 1, time() - startTime, err))
        print("Trained {} epochs in {:.2f}s".format(
            self.epochs, time() - allTime))

    def predict(self, xPredict):
        result, _ = self.forward(xPredict)
        return result
