import sys
import pandas as pd


def selectDataset():
    argc = len(sys.argv)
    if argc > 2:
        print("There should be only one argument, the dataset path !")
        exit()
    dataset_path = "datasets/data.csv"
    if argc == 2:
        dataset_path = sys.argv[1]
    return dataset_path


def openDataset(file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, header=None)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
        exit(1)
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
        exit(1)
    # * Set result column as a value
    df[1] = df[1].apply(lambda value: 1 if value == 'M' else 0)
    return df


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
