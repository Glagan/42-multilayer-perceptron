import random
import sys
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
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=[30, 128, 32, 2],
                            learningRate=0.001, epochs=10000, seed=seed, verbose=True)
    print("Training neural network...")
    network.train(xTrain, yTrain)
    network.accuracy(xTest, yTest)
    network.showHistory()
    # TODO Save weights to weights.csv (with network topology)
