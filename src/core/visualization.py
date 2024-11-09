from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from loguru import logger

from core.preprocess import Preprocess


class Visualization:

    @staticmethod
    async def words_visualisation():
        try:
            df = await Preprocess().fully_preprocessed_data()
            vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(df['text'])
            feature_names = vectorizer.get_feature_names_out()
            summed_tfidf = tfidf_matrix.sum(axis=0).A1
            tfidf_scores = dict(zip(feature_names, summed_tfidf))

            wordcloud = WordCloud(max_words=100, background_color='white').generate_from_frequencies(tfidf_scores)
            plt.figure(figsize=(8, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()

        except TypeError as e:
            logger.error(f'Error in words_visualisation: {e!r}')
