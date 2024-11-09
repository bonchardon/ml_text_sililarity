from typing import Any
from string import punctuation
from asyncio import gather

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    async def preprocess(data) -> DataFrame:
        try:
            df: DataFrame = data.loc[:, data.notna().all(axis=0) & (data != 0).all(axis=0)]
            duplicates: DataFrame | None = df[df.duplicated(subset='text', keep=False)]
            if duplicates.empty:
                logger.info('Seems like there are no duplicates')
            else:
                logger.info(f'Duplicates: \n'
                            f'{duplicates[0:5]}')
            df: DataFrame = df.drop_duplicates(subset='text', keep='first')
            return df.dropna()
        except ValueError as e:
            logger.error(f'Here is the error: {e!r}')

    @staticmethod
    async def lemmas(text_data) -> DataFrame | None:
        try:
            word_lemmas: WordNetLemmatizer = WordNetLemmatizer()
            words = text_data.split()
            return [word_lemmas.lemmatize(word) for word in words]
        except ValueError as e:
            logger.error(f'Getting issues here, namely ==> {e!r}')

    @staticmethod
    async def fuck_punctuation(text_data) -> DataFrame:
        try:
            return text_data.translate(str.maketrans('', '', punctuation)).lower()
        except ValueError as e:
            logger.error(f'Oops, something went wrong in fuck_punctuation. Mama told ya not to use profanity, '
                         f'but you didnt listen. Maybe, this would help: {e!r}')

    async def fully_preprocessed_data(self) -> DataFrame:
        try:
            df: DataFrame = await self.preprocess(self.df)
            tasks: list = [
                self.fuck_punctuation(text)
                for text in df['text']
            ]
            cleaned_texts: Any = await gather(*tasks)
            lemmatized_texts: list = [await self.lemmas(text) for text in cleaned_texts]
            df['text'] = [' '.join(text) for text in lemmatized_texts]
            if not df.empty:
                logger.debug('Something is wrong with df. Maybe, its empty just like your heart.')
            return df
        except (ValueError, IndexError) as e:
            logger.error(f'Having issues here: {e!r}')
