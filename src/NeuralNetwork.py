from time import time
from typing import List
import numpy as np


class NeuralNetwork:
    def __init__(self, size: List, epochs=50, learning_rate=0.001, regularization_strength=0.001, seed=False, verbose=True) -> None:
        self.seed = seed
        self.size = size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.verbose = verbose
        self.initialize()
        print("Created neural network of size", size)
        self.loss_over_epoch = []
        self.test_loss_over_epoch = []

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
        self.test_loss_over_epoch = []

    def softMax(self, y: np.ndarray) -> np.ndarray:
        exp_result = np.exp(y)
        return exp_result / np.sum(exp_result, axis=1, keepdims=True)

    def loss(self, yPred: np.ndarray, yTrue: np.ndarray, size: int) -> float:
        correct_logprobs = -np.log(yPred[range(size), yTrue])
        data_loss = np.sum(correct_logprobs) / size
        reg_loss = 0
        for weights in self.weights:
            reg_loss += (0.5 * self.regularization_strength *
                         np.sum(weights * weights))
        return data_loss, reg_loss

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
            d_weights += self.regularization_strength * r_weights[i]
            d_biases = np.sum(error, axis=0, keepdims=True)
            d_layer = np.dot(error, r_weights[i].T)
            d_layer[r_layers[i] <= 0] = 0
            error = d_layer
            # Update weights and biases
            r_weights[i] += -self.learning_rate * d_weights
            r_biases[i] += -self.learning_rate * d_biases
        # Output layer
        d_weights = np.dot(xTrain.T, error)
        d_weights += self.regularization_strength * r_weights[length]
        d_biases = np.sum(error, axis=0, keepdims=True)
        r_weights[length] += -self.learning_rate * d_weights
        r_biases[length] += -self.learning_rate * d_biases
        # Save updated weights
        self.weights = r_weights[::-1]
        self.biases = r_biases[::-1]
        return error

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, xTest=np.zeros((1)), yTest=np.zeros((1))):
        allTime = time()
        has_test = xTest.any() and yTest.any()
        num_examples = xTrain.shape[0]
        num_test = xTest.shape[0] if has_test else 0
        for epoch in range(self.epochs):
            # print("training data shape", trainingData.shape)
            result, layers = self.forward(xTrain)
            # Get probabilities (softMax) for predicted results
            probs = self.softMax(result)
            # (average) cross-entropy loss and L2 regularization
            data_loss, reg_loss = self.loss(probs, yTrain, num_examples)
            self.loss_over_epoch.append(data_loss + reg_loss)
            # Calculate loss with test set if there is one
            if has_test:
                test_result = self.predict(xTest, yTest)
                test_data_loss, _ = self.loss(test_result, yTest, num_test)
                self.test_loss_over_epoch.append(test_data_loss)
            if self.verbose and epoch % 1000 == 0:
                if has_test:
                    print('epoch: {:5}/{} | [Train] loss: {:.4f}, data loss: {:.4f}, regularization loss: {:.4f} | [Test] loss: {:.4f}' .format(epoch,
                                                                                                                                                self.epochs, data_loss + reg_loss, data_loss, reg_loss, test_data_loss))
                else:
                    print('epoch: {:5}/{} | [Train] loss: {:.4f}, data loss: {:.4f}, regularization loss: {:.4f}'.format(epoch,
                                                                                                                         self.epochs, data_loss + reg_loss, data_loss, reg_loss))
            # compute the gradient on scores
            d_result = probs
            d_result[range(num_examples), yTrain] -= 1
            d_result /= num_examples
            self.backward(xTrain, layers, d_result)
        print("Trained {} epochs in {:.2f}s".format(
            self.epochs, time() - allTime))

    def predict(self, xPredict: np.ndarray, yPredict: np.ndarray):
        result = xPredict
        # All layers except last
        length = len(self.weights) - 1
        for i in range(length):
            result = np.maximum(
                0, np.dot(result, self.weights[i]) + self.biases[i])
        # Output layer
        result = np.dot(result, self.weights[length]) + self.biases[length]
        probs = self.softMax(result)
        return probs

    def accuracy(self, xPredict: np.ndarray, yPredict: np.ndarray):
        num_examples = xPredict.shape[0]
        probs = self.predict(xPredict, yPredict)
        # (average) cross-entropy loss
        data_loss, _ = self.loss(probs, yPredict, num_examples)
        # Select the highest probability for each classes in the results
        predicted_class = np.argmax(probs, axis=1)
        if self.verbose:
            print('Prediction\n', np.where(predicted_class > 0, 'M', 'B'))
            print('Real values\n', np.where(yPredict > 0, 'M', 'B'))
        errors = np.sum(predicted_class != yPredict)
        print("Prediction accuracy: {:.2f}%, loss: {:.2f} ({} errors out of {})".format(
            np.mean(predicted_class == yPredict) * 100, data_loss, errors, num_examples))
