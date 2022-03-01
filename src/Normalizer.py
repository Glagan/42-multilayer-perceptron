import pandas as pd

class Normalizer:
    def __init__(self, method='minMax'):
        self.method = method

    def minMaxNormalize(self, df: pd.DataFrame, features):
        for (name, data) in df[features].iteritems():
            df[name] = df[name].apply(lambda val, min, max: (val - min) / (max - min), args=(data.min(), data.max()))
        return df
        
    def normalize(self, df: pd.DataFrame):
        """
        Apply the selected normalization to all features columns in a Pandas DataFrame.
        """
        cpy = df.copy()
        features = cpy.columns.to_list()
        if self.method == 'minMax':
            return self.minMaxNormalize(cpy, features)
        raise ValueError("Invalid normalization method '{}'".format(self.method))
