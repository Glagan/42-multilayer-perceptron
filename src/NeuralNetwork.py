from time import time
from typing import List
from matplotlib import pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self, size: List, epochs=50, learningRate=0.001, seed=False, verbose=True) -> None:
        self.seed = seed
        self.size = size
        self.epochs = epochs
        self.learningRate = learningRate
        self.verbose = verbose
        self.initialize()
        print("Created neural network of size", size)
        self.loss_over_epoch = []

    def initialize(self) -> None:
        '''
        Generate the matrix of weights for each layers.
        Weights:
            https://cs231n.github.io/neural-networks-2/#weight-initialization
            Random number with variance sqrt(2 / n)
            w = np.random.randn(n) / sqrt(2 / n)
        Bias is initialized to 0
        '''
        if self.seed != False:
            np.random.seed(self.seed)
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
        self.loss_over_epoch = []

    def forward(self, trainingData: np.ndarray):
        '''
        trainingData [1, D] where D is the number of features
        returns the list values for each neurons in each layers
        '''
        layers = []
        result = trainingData
        # All layers except last
        length = len(self.weights) - 1
        for i in range(length):
            result = np.maximum(
                0, np.dot(result, self.weights[i]) + self.biases[i])
            layers.append(result)
        result = np.dot(result, self.weights[length]) + self.biases[length]
        return result, layers

    def backward(self, xTrain: np.ndarray, layers: np.ndarray, error: np.ndarray):
        '''
        Backpropagation, in reverse order
        Error is of the size of the last layer neurons (amount of classes)
        '''
        # Reverse used list for easier access
        r_layers = layers[::-1]
        r_weights = self.weights[::-1]
        r_biases = self.biases[::-1]
        d_weights = []
        length = len(layers)
        # Backpropagate each layers
        for i in range(length):
            d_weights = np.dot(r_layers[i].T, error)
            d_biases = np.sum(error, axis=0, keepdims=True)
            d_layer = np.dot(error, r_weights[i].T)
            d_layer[r_layers[i] <= 0] = 0
            error = d_layer
            # Update weights and biases
            r_weights[i] += -self.learningRate * d_weights
            r_biases[i] += -self.learningRate * d_biases
        # Output layer
        d_weights = np.dot(xTrain.T, error)
        d_biases = np.sum(error, axis=0, keepdims=True)
        r_weights[length] += -self.learningRate * d_weights
        r_biases[length] += -self.learningRate * d_biases
        # Save updated weights
        self.weights = r_weights[::-1]
        self.biases = r_biases[::-1]
        return error

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray):
        allTime = time()
        num_examples = xTrain.shape[0]
        for epoch in range(self.epochs):
            # print("training data shape", trainingData.shape)
            result, layers = self.forward(xTrain)
            # Get probabilities (softMax) for predicted results
            exp_result = np.exp(result)
            probs = exp_result / np.sum(exp_result, axis=1, keepdims=True)
            # Compute the loss: average cross-entropy loss
            correct_logprobs = -np.log(probs[range(num_examples), yTrain])
            data_loss = np.sum(correct_logprobs) / num_examples
            loss = data_loss
            self.loss_over_epoch.append(loss)
            # How far the probability is from the expected result (0-1)
            value_loss = np.sum(
                yTrain - np.argmax(probs, axis=1)) / num_examples
            if self.verbose and epoch % 1000 == 0:
                print('epoch: {}/{}, loss: {:.2f}, value loss: {:.2f}'.format(epoch,
                      self.epochs, loss, value_loss))
            # compute the gradient on scores
            d_result = probs
            d_result[range(num_examples), yTrain] -= 1
            d_result /= num_examples
            self.backward(xTrain, layers, d_result)
        print("Trained {} epochs in {:.2f}s".format(
            self.epochs, time() - allTime))

    def showHistory(self):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.loss_over_epoch)
        plt.show()

    def accuracy(self, xPredict: np.ndarray, yPredict: np.ndarray):
        result = xPredict
        # All layers except last
        length = len(self.weights) - 1
        for i in range(length):
            result = np.maximum(
                0, np.dot(result, self.weights[i]) + self.biases[i])
        # Output layer
        result = np.dot(result, self.weights[length]) + self.biases[length]
        # Get probabilities (softMax) for predicted results
        exp_result = np.exp(result)
        probs = exp_result / np.sum(exp_result, axis=1, keepdims=True)
        # Select the highest probability for each classes in the results
        predicted_class = np.argmax(probs, axis=1)
        print("Prediction accuracy: %.2f" %
              (np.mean(predicted_class == yPredict)))
