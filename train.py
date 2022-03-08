import random
import sys
from matplotlib import pyplot as plt
import numpy as np
from src.Normalizer import minMaxNormalize
from src.NeuralNetwork import NeuralNetwork
from src.Dataset import openDataset, selectDataset, splitDataset

if __name__ == "__main__":
    # Manually set seed if needed
    # Good seeds: [4172509286, 2862616662, 3380935500, 283079681, 1657489538, 168233142]
    # seed = random.randrange(2**32 - 1)
    seed = 1657489538
    print("Using seed [{}]".format(seed))

    # * Open and read dataset
    print("Loading dataset...")
    dataset_path = selectDataset()
    df = openDataset(dataset_path)
    # * Normalize data
    normalized = minMaxNormalize(df.loc[:, df.columns > 0])
    # * Split dataset in one training and one result set
    # * If there was no --no-split parameter (to test complete datasets)
    split = len(sys.argv) < 2 or sys.argv[1] != '--no-split'
    if split:
        print("Using split dataset [75%]")
        xTrain, yTrain, xTest, yTest = splitDataset(
            normalized, quantity=0.75, seed=seed)
    else:
        xTrain = normalized.loc[:, normalized.columns > 1].to_numpy()
        yTrain = normalized[1].to_numpy(dtype="uint8")
        xTest = np.zeros((1))
        yTest = np.zeros((1))
    # * Initialize neural network
    print("Initializing neural network...")
    network = NeuralNetwork(size=[30, 256, 128, 64, 32, 2],
                            learning_rate=0.001,
                            epochs=10000, seed=seed, verbose=True)
    print("Training neural network...")
    network.train(xTrain, yTrain, xTest, yTest)
    # * Calculate accuracy for test set if there was one
    if xTest.any() and yTest.any():
        network.accuracy(xTest, yTest)
    # * Save weights (with network topology)
    try:
        file = open("weights.res", "w")
        file.write("{}\n".format(','.join(str(value)
                   for value in network.size)))
        for layer in network.weights:
            np.savetxt(file, layer, newline=",")
        file.write("\n")
        for layer in network.biases:
            np.savetxt(file, layer, newline=",")
        file.write("\n")
        file.close()
        print('Saved trained model')
    except IOError:
        print('Failed to save trained model')
    # * Show history graph last
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(network.loss_over_epoch, label="Training set")
    if len(network.test_loss_over_epoch):
        plt.plot(network.test_loss_over_epoch, label="Test set")
    plt.legend(loc="upper right")
    plt.show()
