import random
from matplotlib import pyplot as plt
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
                            learning_rate=0.01, epochs=1000, seed=seed, verbose=False)
    # * Test multiple sizes
    loss_per_learning_rate = []
    # * Each new training use the same seed
    # * but since the size change the weights are different
    sizes = [
        [30, 16, 8, 2],
        [30, 64, 32, 2],
        [30, 64, 64, 2],
        [30, 128, 64, 2],
        [30, 128, 128, 2],
        [30, 256, 128, 64, 2],
        [30, 256, 128, 128, 2],
    ]
    for size in sizes:
        network.size = size
        network.initialize()
        print("Training neural network with size {}".format(size))
        network.train(xTrain, yTrain)
        network.accuracy(xTest, yTest)
        loss_per_learning_rate.append(network.loss_over_epoch)
    # * Show results, one line per sizes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for index, loss in enumerate(loss_per_learning_rate):
        plt.plot(loss, label=sizes[index])
    plt.legend(loc="upper right")
    plt.show()
