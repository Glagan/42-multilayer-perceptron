import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Dataset used as 32 columns and 569 rows.
1 column is the output, and another column might be the ID, which means we have 30 features we will be using.
Here are some layout that could be used and tested "(Input) -> Layers -> Output (binary)":
(Input) 30 -> 16 -> 8 -> 1 (Output) (Nearly half every layer)
(Input) 30 -> 30 -> 16 -> 1 (Output) (Half after first)
(Input) 30 -> 20 -> 10 -> 1 (Output) (2/3)
Data is normalized for performances and to be numerically stable, since some features have a much bigger range than others.
"""


def minMaxNormalize(val, min, max):
    return (val - min) / (max - min)


def normalize(df: pd.DataFrame):
    """
    Apply a min-max normalization to all features columns in a Pandas DataFrame.
    """
    normalized = df.copy()
    features = normalized.columns.to_list()
    for (name, data) in normalized[features].iteritems():
        normalized[name] = normalized[name].apply(minMaxNormalize, args=(data.min(), data.max()))
    return normalized


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
    # * Normalize data
    print(df)
    normalized = normalize(df.loc[:, df.columns != 1])
    # * Initialize neural network
    # *     Weights:
    # *         https://cs231n.github.io/neural-networks-2/#weight-initialization
    # *         Random number with variance sqrt(2 / n)
    # *         w = np.random.randn(n) / sqrt(2 / n)
    # * In loop
    # *     Forward propagation
    # *     Calculate cost
    # *     Backward propagation
    # * Per loop stats (epoch, cost, accuracy ?)
    # * Save weights to weights.csv (with dataset reference ?)
    pass
