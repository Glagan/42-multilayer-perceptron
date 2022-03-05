import random
import sys
from matplotlib import pyplot as plt
import pandas as pd
from src.Normalizer import minMaxNormalize
from src.NeuralNetwork import NeuralNetwork
from src.Dataset import openDataset, selectDataset, splitDataset

if __name__ == "__main__":
    # Manually set seed if needed
    # Good seeds: [4172509286, 2862616662, 3380935500, 283079681, 1657489538]
    seed = random.randrange(2**32 - 1)
    # seed = 2825771122
    print("Using seed [{}]".format(seed))

    # * Open and read dataset
    dataset_path = selectDataset()
    df = openDataset(dataset_path)
    # * Normalize data
    normalized = minMaxNormalize(df.loc[:, df.columns > 0])
    # * Split dataset in one training and one result set
    xTrain, yTrain, xTest, yTest = splitDataset(
        normalized, quantity=0.75, seed=seed)
    # * Initialize (simplified) neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=[30, 64, 32, 2],
                            learningRate=0.001, epochs=5000, seed=seed, verbose=False)
    # * Test multiple learning rate
    loss_per_learning_rate = []
    # * Each new training has the same dataset split and seed
    learningRates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for learningRate in learningRates:
        network.learningRate = learningRate
        network.initialize()
        print("Training neural network with learning rate {}".format(learningRate))
        network.train(xTrain, yTrain)
        loss_per_learning_rate.append(network.loss_over_epoch)
    # * Show results, one line per learning rate
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for index, loss in enumerate(loss_per_learning_rate):
        plt.plot(loss, label=learningRates[index])
    plt.legend(loc="upper right")
    plt.show()
