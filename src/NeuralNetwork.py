from time import time
from typing import Callable, List
import numpy as np


class NeuralNetwork:
    def __init__(self, size: List, epochs=50, learningRate=0.001, seed=False, verbose=True) -> None:
        if len(size) < 4:
            print("The neural network must have at least 2 hidden layers")
            exit(1)
        if seed != False:
            np.random.seed(seed)
        self.size = size
        self.epochs = epochs
        self.learningRate = learningRate
        self.verbose = verbose
        self.loss = False
        self.d_loss = False
        self.activation = False
        self.d_activation = False
        self.output_activation = False
        self.d_output_activation = False
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

    def setLoss(self, fct: Callable[[np.ndarray, np.ndarray], np.ndarray], fctDerivative: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.loss = fct
        self.d_loss = fctDerivative

    def setActivation(self, fct: Callable[[np.ndarray], np.ndarray], fctDerivative: Callable[[np.ndarray], np.ndarray]):
        self.activation = fct
        self.d_activation = fctDerivative
        if self.output_activation == False:
            self.output_activation = fct
            self.d_output_activation = fctDerivative

    def setOutputActivation(self, fct: Callable[[np.ndarray], np.ndarray], fctDerivative: Callable[[np.ndarray], np.ndarray]):
        self.output_activation = fct
        self.d_output_activation = fctDerivative

    def forward(self, trainingData: np.ndarray):
        '''
        trainingData [1, D] where D is the number of features
        returns the list values for each neurons in each layers
        '''
        layers = []
        activeLayers = []
        result = trainingData
        # All layers except last
        length = len(self.weights)
        for i in range(length):
            result = np.dot(result, self.weights[i]) + self.biases[i]
            layers.append(result)
            if i == length - 1:
                result = self.output_activation(result)
            else:
                result = self.activation(result)
            activeLayers.append(result)
        return result, layers, activeLayers

    def backward(self, layers: np.ndarray, activeLayers: np.ndarray, error: np.ndarray):
        '''
        Backpropagation, in reverse order
        Error is of the size of the last layer neurons (amount of classes)
        '''
        length = len(layers) - 1
        for i in range(length, -1, -1):
            inputError = np.dot(error, self.weights[i].T)
            # ! i - 1 < 0 ???
            if i == length:
                weightsError = np.dot(layers[i - 1].T, error)
            else:
                weightsError = self.d_activation(
                    np.dot(layers[i - 1].T, error))
            self.weights[i] -= self.learningRate * weightsError
            self.biases[i] -= self.learningRate * error
            error = inputError
        return error

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray):
        assert self.loss != False
        assert self.d_loss != False
        assert self.activation != False
        assert self.d_activation != False
        assert self.output_activation != False
        assert self.d_output_activation != False

        # print(xTrain)
        allTime = time()
        for epoch in range(self.epochs):
            startTime = time()
            for trainingData, correctData in zip(xTrain, yTrain):
                # print("training data shape", trainingData.shape)
                result, layers, activeLayers = self.forward(trainingData)
                visualError = self.loss(result, correctData)
                error = self.d_loss(result, correctData)
                # Update weights and biasses depending on the error value
                self.backward(layers, activeLayers, error)
            if self.verbose:
                print('Epoch: {}, Time Spent: {:.2f}s, error: {:.2f}'.format(
                    epoch + 1, time() - startTime, visualError))
        print("Calculated weights", self.weights)
        print("Trained {} epochs in {:.2f}s".format(
            self.epochs, time() - allTime))

    def predict(self, xPredict: np.ndarray):
        result, _, __ = self.forward(xPredict)
        return result
