import pandas as pd


def minMaxNormalize(df: pd.DataFrame):
    try:
        print("Normalizing data...")
        cpy = df.copy()
        features = cpy.columns.to_list()
        for (name, data) in cpy[features].iteritems():
            cpy[name] = cpy[name].apply(lambda val, min, max: (
                val - min) / (max - min), args=(data.min(), data.max()))
        return cpy
    except ValueError as err:
        print(err)
        exit(1)
