import sys
import pandas as pd
import matplotlib.pyplot as plt
from src.Normalizer import Normalizer
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


def splitDataset(df: pd.DataFrame, quantity=0.7, seed=False):
    train = df.sample(frac=0.8, random_state=seed if seed else None)
    test = df.drop(train.index)
    return [train.loc[:, df.columns > 1], train[1], test.loc[:, df.columns > 1], test[1]]


if __name__ == "__main__":
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
        normalizer = Normalizer()
        normalized = normalizer.normalize(df.loc[:, df.columns > 0])
        # print(normalized)
    except ValueError as err:
        print(err)
        exit(1)
    # * Split dataset in one training and one result set
    xTrain, yTrain, xTest, yTest = splitDataset(normalized, seed=42)
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(seed=42)
    print("Training neural network...")
    network.train(xTrain, yTrain, xTest, yTest)
    # * Per loop stats (epoch, cost, accuracy ?) + Graph
    # *   epoch 39/70 - loss: 0.0750 - val_loss: 0.0406
    # * Save weights to weights.csv (with network topology)
