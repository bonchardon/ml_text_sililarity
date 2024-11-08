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

    async def preprocess(self):
        try:
            df: DataFrame = self.df.loc[:, self.df.notna().all(axis=0) & (self.df != 0).all(axis=0)]

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
    async def words_visualisation():
        try:
            df = await Preprocess().preprocess()
            text_combined = ' '.join(df['text'].dropna())

            wordcloud = WordCloud(max_words=1000, background_color='white').generate(text_combined)
            plt.figure(figsize=(8, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
        except TypeError as e:
            logger.error(f'Here is the problem: {e!r}')
