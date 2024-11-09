''' Trial of similarity algorithms '''

from numpy import fill_diagonal, max, argmax

# from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.preprocess import Preprocess


class TextSimilarity:

    def __init__(self, data_texts):
        self.data_texts: dict[str] = data_texts

    async def tf_idf_similarity_check(self):
        vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data_texts['text'])
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        fill_diagonal(similarity, -1)
        most_similar_indices = argmax(similarity, axis=1)
        most_similar_texts = [self.data_texts.iloc[i] for i in most_similar_indices]
        most_similar_scores = max(similarity, axis=1)
        return most_similar_texts, most_similar_scores

    # async def bert_similarity(self):
        # todo: consider and check most reliable models
        #model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        #embeddings = model.encode(self.data_texts)
        #similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
        #print(similarity)
