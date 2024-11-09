import numpy as np
import pandas as pd
from fontTools.subset import subset
from pandas import DataFrame

from loguru import logger

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from core.consts import URL_RESOURCE

np.random.seed(42)


class Preprocess:

    def __init__(self):
        self.df: DataFrame = pd.read_csv(URL_RESOURCE)

    @staticmethod
    async def preprocess(data):
        try:
            df: DataFrame = data.loc[:, data.notna().all(axis=0) & (data != 0).all(axis=0)]
            duplicates = df[df.duplicated(subset='text', keep=False)]
            if duplicates.empty:
                logger.info('Seems like there are no duplicates')
            else:
                logger.info(f'Duplicates: {duplicates}')
            df: DataFrame = df.drop_duplicates(subset='text', keep='first')
            return df.dropna()
        except ValueError as e:
            logger.error(f'Here is the error: {e!r}')

    @staticmethod
    async def fuck_punctuation():
        try:
            ...
        except ValueError as e:
            logger.error(f'Ooops, something went wrong in fuck_punctuation. Mama told ya not to use profanity, '
                         f'but you didnt listen')


