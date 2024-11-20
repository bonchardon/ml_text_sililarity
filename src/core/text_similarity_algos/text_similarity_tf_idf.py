from typing import Any

from numpy import fill_diagonal, max, argmax

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfIdfSimilarity:

    def __init__(self, data_texts: dict[str]):
        self.data_texts: dict[str] = data_texts

    async def tf_idf_similarity_check(self) -> list | Any:
        vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data_texts['text'])
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        fill_diagonal(similarity, -1)
        most_similar_indices = argmax(similarity, axis=1)
        most_similar_texts = [list(self.data_texts.values())[i] for i in most_similar_indices]
        most_similar_scores = max(similarity, axis=1)
        return most_similar_texts, most_similar_scores
