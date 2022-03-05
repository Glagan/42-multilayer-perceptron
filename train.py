import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.Normalizer import minMaxNormalize
from src.NeuralNetwork import NeuralNetwork

"""
Dataset used as 32 columns and 569 rows.
1 column is the output, and another column might be the ID, which means we have 30 features we will be using.
Here are some layout that could be used and tested "(Input) -> Layers -> Output (binary)":
(Input) 30 -> 16 -> 8 -> 1 (Output) (Nearly half every layer)
(Input) 30 -> 30 -> 16 -> 1 (Output) (Half after first)
(Input) 30 -> 20 -> 10 -> 1 (Output) (2/3)
Data is normalized for performances and to be numerically stable, since some features have a much bigger range than others.
"""


def splitDataset(df: pd.DataFrame, quantity=0.75, seed=False):
    cpy = df.copy()
    train = cpy.sample(frac=quantity, random_state=seed if seed else None)
    test = cpy.drop(train.index)
    # Convert y to an array with 2 columns, one for each class
    # yTrain = np.ndarray((train.shape[0], 2), dtype="uint8")
    # index = 0
    # for value in train.loc[:, 1]:
    #     if value:
    #         yTrain[index] = [0, 1]
    #     else:
    #         yTrain[index] = [1, 0]
    #     index += 1
    # yTest = np.ndarray((test.shape[0], 2), dtype="uint8")
    # index = 0
    # for value in test.loc[:, 1]:
    #     if value:
    #         yTest[index] = [0, 1]
    #     else:
    #         yTest[index] = [1, 0]
    #     index += 1
    return [
        train.loc[:, cpy.columns > 1].to_numpy(),
        train[1].to_numpy(dtype="uint8"),
        # yTrain,
        test.loc[:, cpy.columns > 1].to_numpy(),
        test[1].to_numpy(dtype="uint8"),
        # yTest,
    ]


if __name__ == "__main__":
    # Manually set seed if needed
    # Good seeds: [4172509286, 2862616662, 3380935500, 283079681, 1657489538]
    seed = random.randrange(2**32 - 1)
    # seed = 2825771122
    print("Using seed [{}]".format(seed))

    # * Read dataset
    argc = len(sys.argv)
    if argc > 2:
        print("Usage: train.py [datasets/data.csv]")
        exit()
    dataset = "datasets/data.csv"
    if argc == 2:
        dataset = sys.argv[1]
    try:
        df = pd.read_csv(dataset, header=None)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
        exit(1)
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
        exit(1)
    # * Set result column as a value
    df[1] = df[1].apply(lambda value: 1 if value == 'M' else 0)
    # * Normalize data
    try:
        print("Normalizing data...")
        normalized = minMaxNormalize(df.loc[:, df.columns > 0])
        # print(normalized)
    except ValueError as err:
        print(err)
        exit(1)
    # * Split dataset in one training and one result set
    xTrain, yTrain, xTest, yTest = splitDataset(
        normalized, quantity=0.75, seed=seed)
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=[30, 256, 64, 2],
                            learningRate=0.001, epochs=10000, seed=seed, verbose=True)
    print("Training neural network...")
    network.train(xTrain, yTrain)
    network.accuracy(xTest, yTest)
    # * Save weights to weights.csv (with network topology)
