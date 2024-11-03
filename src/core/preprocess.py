import numpy as np
import pandas as pd
from pandas import DataFrame

from core.consts import URL_RESOURCE

np.random.seed(42)


class Preprocess:

    def __init__(self):
        self.df: DataFrame = pd.read_csv(URL_RESOURCE)

    async def preprocess(self):
        # todo 1: if any of the values is null/0/None drop na)
        df = self.df.loc[:, self.df.notna().all(axis=0) & (self.df != 0).all(axis=0)]

        # todo 2: drop all duplicates
        df = df[df.duplicated(keep=False) == False]

        return df
