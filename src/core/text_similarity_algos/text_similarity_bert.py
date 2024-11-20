from typing import Any

from loguru import logger

import torch
import torch.nn.functional as f

from transformers import BertTokenizer, BertModel, PreTrainedModel, BatchEncoding

from sklearn.metrics.pairwise import cosine_similarity
from transformers.models.distilbert.modeling_distilbert import Embeddings

from core.consts import BERT_MODEL
from core.visualization import Visualization


class TextSimilarityBert:

    def __init__(self, data_texts: dict[str]):
        self.data_texts: dict[str] = data_texts

    async def bert_similarity(self) -> Any:
        # todo: consider and check most reliable models
        tokenizer: BertTokenizer | None = BertTokenizer.from_pretrained(BERT_MODEL)
        model: PreTrainedModel = BertModel.from_pretrained(BERT_MODEL)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if not model:
            logger.warning('Seems like there is an issue with model itself. Check it out.')

        encodings: BatchEncoding = tokenizer(self.data_texts['text'], padding=True, return_tensors='pt')
        logger.info(f'Here are the keys ==> {encodings.keys()}')
        logger.info([f'{tokens} ==> {tokenizer.convert_ids_to_tokens(tokens)} \n '
                     for tokens in encodings['input_ids']])

        with torch.no_grad():
            embeddings: Embeddings = model(**encodings)[0]

        logger.info(f'The shape is: {embeddings.shape}')
        return embeddings

    async def distance_computation(self) -> list:
        # 1st solution
        CLSs = await self.bert_similarity()
        CLSs = CLSs[:, 0, :]
        normalized_cls = f.normalize(CLSs, p=2, dim=1)
        cls_dist = normalized_cls.matmul(normalized_cls.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.numpy()
        cls_similarity = cosine_similarity(cls_dist.cpu().numpy())

        # 2nd solution
        MEANS = await self.bert_similarity()
        MEANS = MEANS.mean(dim=1)
        normalized = f.normalize(MEANS, p=2, dim=1)
        mean_dist = normalized.matmul(normalized.T)
        mean_dist = mean_dist.new_ones(mean_dist.shape) - mean_dist
        mean_dist = mean_dist.numpy()

        # 3rd solution
        MAXS, _ = await self.bert_similarity()
        MAXS, _ = MAXS, _.max(dim=1)
        normalized = f.normalize(MAXS, p=2, dim=1)
        max_dist = normalized.matmul(normalized.T)
        max_dist = max_dist.new_ones(max_dist.shape) - max_dist
        max_dist = max_dist.numpy()
        return [cls_similarity, mean_dist, max_dist]

    async def combine_visualization(self) -> None:
        dist: list = await self.distance_computation()
        if not dist:
            logger.error('There is an error here somewhere.')
            return
        titles: list = ['CLS', 'MEAN', 'MAX']
        Visualization().visualize([dist[0], dist[1], dist[2]], titles=titles)
