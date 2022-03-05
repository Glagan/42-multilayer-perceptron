import io
import random
import numpy as np
from src.Normalizer import minMaxNormalize
from src.NeuralNetwork import NeuralNetwork
from src.Dataset import openDataset, selectDataset, splitDataset

if __name__ == "__main__":
    # * Open and read dataset
    print("Loading dataset...")
    dataset_path = selectDataset("datasets/correction.csv")
    df = openDataset(dataset_path)
    # * Normalize data
    normalized = minMaxNormalize(df.loc[:, df.columns > 0])
    xPredict = normalized.loc[:, normalized.columns > 1].to_numpy()
    yPredict = normalized[1].to_numpy(dtype="uint8")
    # * Get saved weights and network size
    print("Loading saved weights...")
    try:
        with open('weights.res') as file:
            lines = file.readlines()
            if len(lines) != 3:
                print("Invalid saved weights !")
            size = [int(value) for value in lines[0].split(',')]
            # Load weights and biases as separate files
            # Remove final separator and newline before split
            weights = []
            current_layer = []
            for layer in lines[1][:-2].split(','):
                weights_file = io.StringIO(layer)
                row_weights = np.loadtxt(weights_file)
                if len(current_layer) > 0 and current_layer[0].shape != row_weights.shape:
                    weights.append(np.asarray(current_layer))
                    current_layer = []
                current_layer.append(row_weights)
            weights.append(np.asarray(current_layer))
            # Bias
            biases = []
            current_layer = []
            for layer in lines[2][:-2].split(','):
                biases_file = io.StringIO(layer)
                row_biases = np.loadtxt(biases_file)
                if len(current_layer) > 0 and current_layer[0].shape != row_biases.shape:
                    biases.append(np.asarray(current_layer))
                    current_layer = []
                current_layer.append(row_biases)
            biases.append(np.asarray(current_layer))
    except IOError:
        print('No weights saved, use train.py first !')
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=size)
    network.weights = weights
    network.biases = biases
    network.accuracy(xPredict, yPredict)
