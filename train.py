import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.Activation import d_reLu, d_softMax, reLu, softMax
from src.Loss import d_mse, mse
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


def splitDataset(df: pd.DataFrame, quantity=0.75, seed=False):
    cpy = df.copy()
    train = cpy.sample(frac=quantity, random_state=seed if seed else None)
    test = cpy.drop(train.index)
    return [train.loc[:, cpy.columns > 1].to_numpy(), train[1].to_numpy(), test.loc[:, cpy.columns > 1].to_numpy(), test[1].to_numpy()]


if __name__ == "__main__":
    # Manually set seed if needed
    # Good seeds: [4172509286, 2862616662, 3380935500, 283079681]
    # seed = 42
    seed = random.randrange(2**32 - 1)
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
        normalizer = Normalizer()
        normalized = normalizer.normalize(df.loc[:, df.columns > 0])
        # print(normalized)
    except ValueError as err:
        print(err)
        exit(1)
    # * Split dataset in one training and one result set
    xTrain, yTrain, xTest, yTest = splitDataset(
        normalized, quantity=0.75, seed=seed)
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=[30, 32, 16, 1],
                            learningRate=0.0001, epochs=10, seed=seed, verbose=True)
    network.setActivation(reLu, d_reLu)
    network.setOutputActivation(softMax, d_softMax)
    network.setLoss(mse, d_mse)
    print("Training neural network...")
    network.train(xTrain, yTrain)
    # * DEBUG Compare trained model against the test dataset
    errors = 0
    for x, y in zip(xTest, yTest):
        # print(network.predict(x), y)
        predictedX = np.ravel(np.round(network.predict(x)))[0]
        # print(predictedX, y)
        if predictedX != y:
            errors += 1
    errorRate = (errors / len(xTest)) * 100
    print("Errors: {}/{} {:.2f}% ({:.2f}% correct)".format(errors,
          len(xTest), errorRate, 100 - errorRate))
    # * Per loop stats (epoch, cost, accuracy ?) + Graph
    # *   epoch 39/70 - loss: 0.0750 - val_loss: 0.0406
    # * Save weights to weights.csv (with network topology)
