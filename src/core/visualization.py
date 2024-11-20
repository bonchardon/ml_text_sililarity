import numpy as np

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from loguru import logger

from text_similarity_algos.preprocess import Preprocess


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

    @staticmethod
    def visualize(distances, figsize=(10, 5), titles=None):
        ncols = len(distances)
        fig, ax = plt.subplots(ncols=ncols, figsize=figsize)

        for i in range(ncols):
            axes = ax[i] if ncols > 1 else ax
            distance = distances[i]
            axes.imshow(distance)
            axes.set_xticks(np.arange(distance.shape[0]))
            axes.set_yticks(np.arange(distance.shape[1]))
            axes.set_xticklabels(np.arange(distance.shape[0]))
            axes.set_yticklabels(np.arange(distance.shape[1]))
            for j in range(distance.shape[0]):
                for k in range(distance.shape[1]):
                    text = axes.text(k, j, str(round(distance[j, k].item(), 3)),
                                     ha="center", va="center", color="w")
            title = titles[i] if titles and len(titles) > i else "Text Distance"
            axes.set_title(title, fontsize="x-large")

        fig.tight_layout()
        plt.show()
