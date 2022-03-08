import sys
import pandas as pd


def selectDataset(default="datasets/data.csv"):
    argc = len(sys.argv)
    dataset_path = default
    if argc >= 2 and sys.argv[argc - 1] != "--no-split":
        dataset_path = sys.argv[argc - 1]
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
    return [
        train.loc[:, cpy.columns > 1].to_numpy(),
        train[1].to_numpy(dtype="uint8"),
        test.loc[:, cpy.columns > 1].to_numpy(),
        test[1].to_numpy(dtype="uint8"),
    ]
