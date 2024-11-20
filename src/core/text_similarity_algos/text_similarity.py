''' Trial of similarity algorithms '''
from typing import Any

import numpy
from numpy import fill_diagonal, max, argmax

from loguru import logger

import torch
import torch.nn.functional as f

from transformers import BertTokenizer, BertModel, PreTrainedModel, BatchEncoding
from transformers import BertTokenizer, BertModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.models.distilbert.modeling_distilbert import Embeddings

from core.consts import BERT_MODEL
from core.visualization import Visualization


class TextSimilarity:

    # def __init__(self, data_texts):
    #    self.data_texts: dict[str] = data_texts

    async def tf_idf_similarity_check(self) -> list | Any:
        vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data_texts['text'])
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        fill_diagonal(similarity, -1)
        most_similar_indices = argmax(similarity, axis=1)
        most_similar_texts = [self.data_texts.iloc[i] for i in most_similar_indices]
        most_similar_scores = max(similarity, axis=1)
        return most_similar_texts, most_similar_scores

    def bert_similarity(self) -> Any:
        # todo: consider and check most reliable models
        tokenizer: BertTokenizer | None = BertTokenizer.from_pretrained(BERT_MODEL)
        model: PreTrainedModel = BertModel.from_pretrained(BERT_MODEL)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if not model:
            logger.warning('Seems like there is an issue with model itself. Check it out.')

        text_1: str = 'Hate the fact that you hate me'
        text_2: str = 'i love the fact that you love that you hate me'
        text_3: str = 'bitches gonna be bitches'
        text_4: str = 'i hate that you hate me'

        texts: list = [text_1, text_2, text_3, text_4]

        encodings: BatchEncoding = tokenizer(texts, padding=True, return_tensors='pt')
        logger.info(f'Here are the keys ==> {encodings.keys()}')
        logger.info([f'{tokens} ==> {tokenizer.convert_ids_to_tokens(tokens)} \n '
                     for tokens in encodings['input_ids']])

        with torch.no_grad():
            embeddings: Embeddings = model(**encodings)[0]

        logger.info(f'The shape is: {embeddings.shape}')

        # 1st solution
        CLSs = embeddings[:, 0, :]
        normalized_cls = f.normalize(CLSs, p=2, dim=1)
        cls_dist = normalized_cls.matmul(normalized_cls.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.numpy()

        cls_similarity = cosine_similarity(normalized_cls.cpu().numpy())

        # 2nd solution
        MEANS = embeddings.mean(dim=1)
        normalized = f.normalize(MEANS, p=2, dim=1)
        mean_dist = normalized.matmul(normalized.T)
        mean_dist = mean_dist.new_ones(mean_dist.shape) - mean_dist
        mean_dist = mean_dist.numpy()

        # 3rd solution
        MAXS, _ = embeddings.max(dim=1)
        normalized = f.normalize(MAXS, p=2, dim=1)
        max_dist = normalized.matmul(normalized.T)
        max_dist = max_dist.new_ones(max_dist.shape) - max_dist
        max_dist = max_dist.numpy()

        # visualization part
        dist = [cls_dist, mean_dist, max_dist]
        titles = ["CLS", "MEAN", "MAX"]
        Visualization().visualize(dist, titles=titles)


if __name__ == '__main__':
    test = TextSimilarity()
    print(test.bert_similarity())
#
# from sklearn.metrics.pairwise import cosine_similarity
#
# import torch
# import torch.nn.functional as f
#
# from transformers import BertTokenizer, BertModel, PreTrainedModel, BatchEncoding
#
# from loguru import logger
#
# from core.visualization import Visualization
# from core.consts import BERT_MODEL
#
#
# class TextSimilarity:
#
#     def __init__(self, data_texts):
#         self.data_texts: dict[str] = data_texts
#
#     async def bert_similarity(self):
#         tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
#         model = BertModel.from_pretrained(BERT_MODEL)
#         model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#
#         encodings: BatchEncoding = tokenizer(self.data_texts, padding=True, return_tensors='pt')
#         with torch.no_grad():
#             embeddings = model(**encodings)[0]
#
#         CLSs = embeddings[:, 0, :]  # CLS token embeddings
#         MEANS = embeddings.mean(dim=1)  # Mean pooling
#         MAXS, _ = embeddings.max(dim=1)  # Max pooling
#
#         # Normalize embeddings (important for cosine similarity)
#         normalized_cls = f.normalize(CLSs, p=2, dim=1)
#         normalized_mean = f.normalize(MEANS, p=2, dim=1)
#         normalized_max = f.normalize(MAXS, p=2, dim=1)
#
#         # Calculate cosine similarities for each strategy
#         cls_similarity = cosine_similarity(normalized_cls.cpu().numpy())
#         mean_similarity = cosine_similarity(normalized_mean.cpu().numpy())
#         max_similarity = cosine_similarity(normalized_max.cpu().numpy())
#
#         # Find most similar sentences based on the cosine similarity
#         most_similar_cls = self._find_most_similar(cls_similarity, texts=self.data_texts)
#         most_similar_mean = self._find_most_similar(mean_similarity, texts=self.data_texts)
#         most_similar_max = self._find_most_similar(max_similarity, texts=self.data_texts)
#
#         # Visualization
#         dist = [cls_similarity, mean_similarity, max_similarity]
#         titles = ["CLS", "MEAN", "MAX"]
#         Visualization().visualize(dist, titles=titles)
#
#         # Return the results
#         return {
#             "most_similar_cls": most_similar_cls,
#             "most_similar_mean": most_similar_mean,
#             "most_similar_max": most_similar_max,
#         }
#
#     def _find_most_similar(self, similarity_matrix, texts):
#         most_similar_indices = similarity_matrix.argmax(axis=1)
#         most_similar_scores = similarity_matrix.max(axis=1)
#
#         most_similar_texts = [
#             (texts[i], most_similar_scores[i])
#             for i in most_similar_indices
#         ]
#         return most_similar_texts
#
#
# # if __name__ == '__main__':
# #     test = TextSimilarity()
# #     result = test.bert_similarity()
# #     logger.info(result)
#
# # means works the best

