from wordcloud import WordCloud
import matplotlib.pyplot as plt

from loguru import logger

from core.preprocess import Preprocess


class Visualization:
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
