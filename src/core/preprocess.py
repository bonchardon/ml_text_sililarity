from string import punctuation

import numpy as np
import pandas as pd
from pandas import DataFrame

from loguru import logger

from core.consts import URL_RESOURCE

np.random.seed(42)


class Preprocess:

    def __init__(self):
        self.df: DataFrame = pd.read_csv(URL_RESOURCE)

    @staticmethod
    async def preprocess(data):
        try:
            df: DataFrame = data.loc[:, data.notna().all(axis=0) & (data != 0).all(axis=0)]
            duplicates: DataFrame | None = df[df.duplicated(subset='text', keep=False)]
            if duplicates.empty:
                logger.info('Seems like there are no duplicates')
            else:
                logger.info(f'Duplicates: {duplicates}')
            df: DataFrame = df.drop_duplicates(subset='text', keep='first')
            return df.dropna()
        except ValueError as e:
            logger.error(f'Here is the error: {e!r}')

    @staticmethod
    async def delete_stop_words():
        try:
            ...
        except Exception as e:
            logger.error(f'The error is here: {e!r}')

    @staticmethod
    async def fuck_punctuation(text_data):
        try:
            return text_data.translate(str.maketrans('', punctuation))
        except ValueError as e:
            logger.error(f'Oops, something went wrong in fuck_punctuation. Mama told ya not to use profanity, '
                         f'but you didnt listen. Maybe, this would help: {e!r}')

    @classmethod
    async def fully_preprocessed_data(cls, data: DataFrame):
        try:
            df: DataFrame = await cls.preprocess(data)
            df['text'] = df['text'].apply(lambda x: cls.fuck_punctuation(x))
            if not df:
                logger.debug('Something is wrong with df.')
            return df
        except (ValueError, IndexError) as e:
            logger.error(f'Having issues here: {e!r}')
